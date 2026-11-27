"""Roblox controller support: whether a game's UI and actions can be driven with a gamepad, read from a live Studio.

A game adds gamepad support itself; this module is the check. Roblox's console guidelines
(https://create.roblox.com/the internal notes, read 2026-11-20) ask that every UI element be reachable
with the four directions, select and back, key actions in a few moves, and UI inside TV-safe areas. They give no button
or text size in pixels, so sizes are reported, and a threshold the caller passes is advisory only: an uncited number never
fails an audit.

What a plugin can reach: GuiService.SelectedObject, GuiObject.Selectable and
NextSelection*, GuiBase2d.SelectionGroup and SelectionBehavior*, ScreenGui.ScreenInsets, ContextActionService:
GetAllBoundActionInfo (in a running client), ProximityPrompt.GamepadKeyCode, and the device simulator
(StudioDeviceSimulatorService, PluginSecurity), whose xbox and ps5 devices turn GuiService:IsTenFootInterface true and
ViewportDisplaySize Large in edit mode. VirtualInputManager's gamepad members and StudioDeviceEmulatorService are
RobloxScriptSecurity; UserInputService:CreateVirtualInput's SendKey makes Keyboard-type input even for a gamepad KeyCode
(RB1441), so Studio's automatic directional navigation cannot be driven from here: neighbours a NextSelection* link does
not name are MODELED (nearest selectable in that direction) and every answer says which.

Every tool is a composite over tools.py's generic layer; nothing here has an op of its own. NEVER RAISE.
"""
import functools
import json
import re
import time
import uuid
from typing import Any

from mif_roblox import values as V

__all__ = []

DIRS = ("Up", "Down", "Left", "Right")
VEC = {"Up": (0, -1), "Down": (0, 1), "Left": (-1, 0), "Right": (1, 0)}
GAMEPAD_PREFIXES = ("Button", "DPad", "Thumbstick")      # Enum.KeyCode gamepad items: ButtonA..ButtonSelect, DPad*, Thumbstick*
GUIDELINE = "https://create.roblox.com/docs/production/publishing/console-guidelines"
CONSOLES = ("xbox", "ps4", "ps5")                        # device simulator ids whose DeviceForm is Console (RB1443)
PROBE_SCRIPT = "MifControllerProbe"
PROBE_MARK = "MIFPAD:"

# The tail a client script ends with to hand play_client a table r: JSON in numbered parts that join_parts puts back
# together. The Output does not carry every character as printed (a backslash is lost, RB1451), so every byte that is not
# a letter, digit, space or one of , : { } [ ] . _ - " is sent as ~ and two hex digits - about 1.1 times the JSON, where
# plain hex was 2 times and a 2950-element answer then never crossed the client's report (RB2073). Needs HS =
# HttpService. A string in r that is not UTF-8 stops JSONEncode ("Can't convert to JSON", RB2070): then every such
# string is sent with its high bytes as "?" and r.notUtf8 says so. Each part names its run ("i/n@RUN:", found
# 2026-11-22): play_client puts a fresh nonce where __MIFRUN__ stands, and join_parts takes only that run's parts, so
# parts an earlier run printed into the same Output are never joined into this one's answer.
CLIENT_EMIT = r'''
local okJ, json = pcall(function() return HS:JSONEncode(r) end)
if not okJ then
	local function clean(v)
		if type(v) == "string" then
			return if utf8.len(v) == nil then (string.gsub(v, "[\128-\255]", "?")) else v
		elseif type(v) == "table" then
			for k, x in v do v[k] = clean(x) end
		end
		return v
	end
	r = clean(r)
	r.notUtf8 = tostring(json)
	json = HS:JSONEncode(r)
end
local text = string.gsub(json, '[^%%w ,:{}%%[%%]%%._%%-"]', function(c) return string.format("~%%02x", string.byte(c)) end)
local parts = math.ceil(#text / %(chunk)s)
for i = 1, parts do
	print("%(mark)s" .. i .. "/" .. parts .. "@__MIFRUN__:" .. string.sub(text, (i - 1) * %(chunk)s + 1, i * %(chunk)s))
end
'''


def _T():
    from mif_roblox import tools as T
    return T


def _tool(fn):
    @functools.wraps(fn)
    def wrapper(*a, **kw):
        try:
            out = fn(*a, **kw)
        except Exception as exc:  # noqa: BLE001 - never raise
            out = {"ok": False, "error": "%s failed inside the bridge: %s: %s" % (fn.__name__, type(exc).__name__, exc)}
        _T()._post_verdict(fn.__name__, out)
        return out
    __all__.append(fn.__name__)
    return wrapper


def _no(error, **kw):
    return dict({"ok": False, "ran": False, "error": error}, **kw)


def _xy(v):
    if isinstance(v, dict) and "x" in v and "y" in v:
        return [float(v["x"]), float(v["y"])]
    if isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
        return [float(v[0]), float(v[1])]
    return None


def _path(v):
    """An Instance property's decoded value as a dot path, or None."""
    if v is None:
        return None
    if isinstance(v, str):
        return v
    if isinstance(v, list) and all(isinstance(x, str) for x in v):
        return V.join_path(v)
    if isinstance(v, dict) and v.get("path") is not None:
        return _path(v["path"])
    return None


def is_gamepad_input(token):
    """An input type or KeyCode name (as GetAllBoundActionInfo or a prompt gives it) that a gamepad sends."""
    s = str(token or "")
    if s.startswith(("Enum.PlayerActions.", "PlayerActions.")):
        return True          # a character action: Roblox maps it to every device, the gamepad included
    for pre in ("Enum.KeyCode.", "Enum.UserInputType.", "KeyCode.", "UserInputType."):
        if s.startswith(pre):
            s = s[len(pre):]
    return s.startswith(GAMEPAD_PREFIXES) or s.startswith("Gamepad")


# ---------------------------------------------------------------- the selection graph (pure: tested offline)
def _center(r):
    return (r[0] + r[2] / 2.0, r[1] + r[3] / 2.0)


def _group_stop(el, d, by_path):
    """The nearest SelectionGroup ancestor (or the element) whose SelectionBehavior<d> is Stop: navigation that way stays inside it."""
    p = el["path"]
    while p:
        g = by_path.get(p)
        if g and g.get("selectionGroup") and (g.get("behavior") or {}).get(d) == "Stop":
            return p
        p = p.rsplit(".", 1)[0] if "." in p else None
    return None


def modeled_neighbour(el, d, cands, by_path):
    """Studio's automatic navigation is not reachable from a plugin (RB1441), so this is a model: the nearest candidate
    whose center lies in direction d (inside a 45-degree cone either side), scored by distance along d plus twice the
    sideways offset. A SelectionGroup with SelectionBehavior Stop that way keeps the search inside the group."""
    ex, ey = _center(el["rect"])
    vx, vy = VEC[d]
    stop = _group_stop(el, d, by_path)
    best, best_score = None, None
    for c in cands:
        if c["path"] == el["path"]:
            continue
        if stop and not (c["path"] == stop or c["path"].startswith(stop + ".")):
            continue
        cx, cy = _center(c["rect"])
        along = (cx - ex) * vx + (cy - ey) * vy
        side = abs((cx - ex) * vy - (cy - ey) * vx)
        if along <= 0.5 or side > along:
            continue
        score = along + 2.0 * side
        if best_score is None or score < best_score or (score == best_score and c["path"] < best["path"]):
            best, best_score = c, score
    return best["path"] if best else None


def selection_graph(elements, start=None):
    """elements: [{path, rect [x,y,w,h], visible (effective), selectable, interactable?, next {Up..}: path|None,
    selectionOrder, selectionGroup?, behavior?}] for every interactive element (GuiButton, TextBox) plus any group
    containers. Answers the edges per element (explicit link or modeled neighbour), the first selected object, what is
    reachable from it, and the findings."""
    by_path = {e["path"]: e for e in elements}
    inter = [e for e in elements if e.get("interactive")]
    shown = [e for e in inter if e.get("visible") and e.get("rect") and e["rect"][2] > 0 and e["rect"][3] > 0]
    cands = [e for e in shown if e.get("selectable")]
    cand_paths = {e["path"] for e in cands}
    edges, dead = {}, []
    for e in cands:
        row = {}
        for d in DIRS:
            link = (e.get("next") or {}).get(d)
            if link:
                if link in cand_paths:
                    row[d] = {"to": link, "how": "explicit"}
                else:
                    tgt = by_path.get(link)
                    why = ("not found among the audited elements" if tgt is None else
                           "hidden" if not tgt.get("visible") else "not Selectable" if not tgt.get("selectable") else
                           "not interactive")
                    dead.append({"from": e["path"], "direction": d, "to": link, "why": why})
                    row[d] = {"to": None, "how": "explicit-dead", "link": link}
            else:
                n = modeled_neighbour(e, d, cands, by_path)
                row[d] = {"to": n, "how": "modeled"} if n else {"to": None, "how": "modeled-none"}
        edges[e["path"]] = row
    first, first_how = None, None
    if start and start in cand_paths:
        first, first_how = start, "GuiService.SelectedObject"
    elif cands:
        ordered = sorted(cands, key=lambda e: (e.get("selectionOrder") or 0, e["rect"][1], e["rect"][0], e["path"]))
        first, first_how = ordered[0]["path"], "lowest SelectionOrder, then top-left (modeled)"
    reach, seen, frontier, depth = [], set(), [first] if first else [], {}
    if first:
        seen.add(first)
        depth[first] = 0
    while frontier:
        p = frontier.pop(0)
        reach.append(p)
        for d in DIRS:
            to = edges.get(p, {}).get(d, {}).get("to")
            if to and to not in seen:
                seen.add(to)
                depth[to] = depth[p] + 1
                frontier.append(to)
    unreachable = sorted(cand_paths - seen)
    # a dead end: reached, but no way back to the first object (a SelectionGroup that Stops every way, say)
    back = set()
    if first:
        back.add(first)
        changed = True
        while changed:
            changed = False
            for p in reach:
                if p not in back and any(edges.get(p, {}).get(d, {}).get("to") in back for d in DIRS):
                    back.add(p)
                    changed = True
    dead_ends = [p for p in reach if p not in back]
    not_selectable = sorted(e["path"] for e in shown if not e.get("selectable"))
    any_modeled = any(v["how"].startswith("modeled") and v.get("to") for row in edges.values() for v in row.values())
    return {"edges": edges, "first": first, "firstBy": first_how, "reachable": reach, "moves": depth,
            "unreachable": unreachable, "deadEnds": dead_ends, "notSelectable": not_selectable, "deadLinks": dead,
            "modeled": any_modeled, "selectable": len(cands), "shown": len(shown)}


# ---------------------------------------------------------------- reading the UI
_EL_PROPS = ["Visible", "AbsolutePosition", "AbsoluteSize", "Selectable", "Interactable", "Active", "SelectionOrder",
             "SelectionGroup", "NextSelectionUp", "NextSelectionDown", "NextSelectionLeft", "NextSelectionRight",
             "SelectionBehaviorUp", "SelectionBehaviorDown", "SelectionBehaviorLeft", "SelectionBehaviorRight"]
_TEXT_PROPS = ["TextSize", "TextScaled", "TextBounds", "Text"]
_SG_PROPS = ["Enabled", "ScreenInsets", "IgnoreGuiInset", "ClipToDeviceSafeArea", "SafeAreaCompatibility", "AbsoluteSize",
             "DisplayOrder"]


def _get(path, props):
    g = _T()._rbx("get_props", target=V.split_path(path), props=props)
    if not g.get("ok"):
        return None, g.get("error")
    vals = g.get("values") if isinstance(g.get("values"), dict) else {}
    return {k: V.decode(v) for k, v in vals.items()}, None


def _query(root, cls, cap):
    out, page = [], 0
    while True:
        q = _T().rbx_query(root=root, class_name=cls, page=page, page_size=500)
        if not q.get("ok"):
            return None, q.get("error")
        out += list(q.get("matches") or [])
        if not q.get("more") or len(out) >= cap:
            return out[:cap], None
        page += 1


def read_ui(root, cap):
    """Every ScreenGui under root, and every GuiObject under each, read. ([screen guis], [elements], error)."""
    cat, err = _T()._catalog()
    if err:
        return None, None, err.get("error")
    rs = _T()._rbx("resolve", target=V.split_path(root))
    if not rs.get("ok"):
        return None, None, rs.get("error") or "%s not found" % root
    if rs.get("className") == "ScreenGui":
        guis = [{"path": V.join_path(V.split_path(root)), "className": "ScreenGui"}]
    else:
        guis, e = _query(root, "ScreenGui", 200)
        if e:
            return None, None, e
    sg_list, elements = [], []
    for g in guis:
        vals, e = _get(g["path"], _SG_PROPS)
        row = {"path": g["path"], "error": e}
        if vals:
            row.update(enabled=vals.get("Enabled"), screenInsets=vals.get("ScreenInsets"),
                       ignoreGuiInset=vals.get("IgnoreGuiInset"), clipToDeviceSafeArea=vals.get("ClipToDeviceSafeArea"),
                       safeAreaCompatibility=vals.get("SafeAreaCompatibility"), size=_xy(vals.get("AbsoluteSize")),
                       displayOrder=vals.get("DisplayOrder"))
        sg_list.append(row)
    for sg in sg_list:
        objs, e = _query(sg["path"], "GuiObject", cap)
        if e:
            sg["error"] = e
            continue
        for m in objs:
            cls = m.get("className")
            interactive = bool(cls) and (cat.is_a(cls, "GuiButton") or cat.is_a(cls, "TextBox"))
            props = list(_EL_PROPS) + ([p for p in _TEXT_PROPS if cat.find_member(cls, p)[1] is not None] if cls else [])
            props += ["ClipsDescendants"] + (["ScrollingDirection"] if cls == "ScrollingFrame" else [])
            vals, e = _get(m["path"], props)
            vals = vals or {}
            pos, size = _xy(vals.get("AbsolutePosition")), _xy(vals.get("AbsoluteSize"))
            el = {"path": m["path"], "className": cls, "screenGui": sg["path"], "interactive": interactive,
                  "ownVisible": vals.get("Visible"), "selectable": vals.get("Selectable"),
                  "interactable": vals.get("Interactable"), "selectionOrder": vals.get("SelectionOrder"),
                  "selectionGroup": vals.get("SelectionGroup"),
                  "behavior": {d: vals.get("SelectionBehavior" + d) for d in DIRS},
                  "next": {d: _path(vals.get("NextSelection" + d)) for d in DIRS},
                  "rect": (pos + size) if pos and size else None}
            if "TextSize" in vals:
                el.update(textSize=vals.get("TextSize"), textScaled=vals.get("TextScaled"),
                          textBounds=_xy(vals.get("TextBounds")), text=vals.get("Text"))
            if vals.get("ClipsDescendants"):
                el["clips"] = True
            if cls == "ScrollingFrame":
                el["scrollDirection"] = vals.get("ScrollingDirection")
            if e:
                el["error"] = e
            elements.append(el)
        for comp, props in (("UIPadding", _PAD_PROPS), ("UIStroke", _STROKE_PROPS)):
            found, e = _query(sg["path"], comp, cap)
            for m in found or []:
                vals, _e = _get(m["path"], props)
                if vals is not None:
                    sg.setdefault("_" + comp, []).append(dict(vals, path=m["path"]))
    effective_visibility(sg_list, elements)
    return sg_list, elements, None


_PAD_PROPS = ["PaddingLeft", "PaddingRight", "PaddingTop", "PaddingBottom"]
_STROKE_PROPS = ["Thickness", "ApplyStrokeMode", "BorderStrokePosition", "StrokeSizingMode", "Enabled"]


def rims_from_edit(sg_list, elements):
    """(pads, strokes) for judge_rims from read_ui's UIPadding and UIStroke reads."""
    cls = {e["path"]: e.get("className") for e in elements}
    pads, strokes = [], []
    for sg in sg_list:
        for p in sg.pop("_UIPadding", []):
            pads.append({"owner": _parent(p["path"]), **{k[7:]: p.get(k) for k in _PAD_PROPS}})
        for s in sg.pop("_UIStroke", []):
            owner = _parent(s["path"])
            strokes.append({"owner": owner, "ownerClass": cls.get(owner), "thickness": s.get("Thickness"),
                            "mode": s.get("ApplyStrokeMode"), "position": s.get("BorderStrokePosition"),
                            "sizing": s.get("StrokeSizingMode"), "enabled": s.get("Enabled")})
    return pads, strokes


def effective_visibility(sg_list, elements):
    """Set each element's visible: it and every GuiObject between it and its ScreenGui Visible, the ScreenGui Enabled."""
    by_path = {el["path"]: el for el in elements}
    sg_on = {sg["path"]: sg.get("enabled") is not False for sg in sg_list}
    for el in elements:
        vis, p = el.get("ownVisible") is not False and sg_on.get(el["screenGui"], True), el["path"]
        while vis and "." in p:
            p = p.rsplit(".", 1)[0]
            a = by_path.get(p)
            if a is None:
                break
            vis = a.get("ownVisible") is not False
        el["visible"] = vis


def _sizes(shown, min_button_px, min_text_px):
    rows, advisory = [], []
    for e in shown:
        r = e["rect"]
        row = {"path": e["path"], "size": [round(r[2], 1), round(r[3], 1)]}
        if "textSize" in e:
            row.update(textSize=e.get("textSize"), textScaled=e.get("textScaled"),
                       textHeight=(e.get("textBounds") or [None, None])[1])
        rows.append(row)
        if min_button_px and min(r[2], r[3]) < min_button_px:
            advisory.append({"path": e["path"], "what": "button smaller than %g px" % min_button_px, "size": row["size"]})
        th = row.get("textHeight") if e.get("textScaled") else e.get("textSize")
        if min_text_px and isinstance(th, (int, float)) and e.get("text") and th < min_text_px:
            advisory.append({"path": e["path"], "what": "text smaller than %g px" % min_text_px, "textSize": th})
    return rows, advisory


def _safe_area(sg_list, elements):
    """ScreenInsets per ScreenGui; buttons partly or wholly off the ScreenGui's area (any insets); and buttons a None-inset
    ScreenGui puts on the screen's very edge (the guideline asks for TV-safe areas; Roblox gives no percentage, so only an
    element touching the edge is listed)."""
    rows, edge, off = [], [], []
    for sg in sg_list:
        rows.append({k: sg.get(k) for k in ("path", "screenInsets", "ignoreGuiInset", "clipToDeviceSafeArea", "size")})
        if not sg.get("size"):
            continue
        w, h = sg["size"]
        for e in elements:
            if e["screenGui"] != sg["path"] or not e.get("interactive") or not e.get("visible") or not e.get("rect"):
                continue
            x, y, ew, eh = e["rect"]
            if x < -0.5 or y < -0.5 or x + ew > w + 0.5 or y + eh > h + 0.5:
                off.append({"path": e["path"], "rect": e["rect"], "screenGui": sg["path"], "screenSize": [w, h]})
            elif sg.get("screenInsets") == "None" and (x <= 0.5 or y <= 0.5 or x + ew >= w - 0.5 or y + eh >= h - 0.5):
                edge.append({"path": e["path"], "rect": e["rect"], "screenGui": sg["path"]})
    return rows, edge, off


# ---------------------------------------------------------------- rims and badges
TEXT_CLASSES = ("TextLabel", "TextButton", "TextBox")
EDGES = {"X": ("Left", "Right"), "Y": ("Top", "Bottom"), "XY": ("Left", "Right", "Top", "Bottom")}


def _parent(p):
    return p.rsplit(".", 1)[0] if p and "." in p else None


def stroke_overhang(s):
    """How far a UIStroke draws outside its owner's border, in px: Thickness for BorderStrokePosition Outer (the
    default), half for Center, 0 for Inner. None when it draws no border rim: disabled, Contextual on a text object (it
    strokes the text), or ScaledSize (its pixels are not Thickness)."""
    if s.get("enabled") is False:
        return None
    if s.get("mode") == "Contextual" and s.get("ownerClass") in TEXT_CLASSES:
        return None
    if s.get("sizing") == "ScaledSize":
        return None
    t = s.get("thickness")
    if not isinstance(t, (int, float)) or t <= 0:
        return None
    return {"Inner": 0.0, "Center": t / 2.0}.get(s.get("position"), float(t))


def _pad_px(pad, edge, size):
    u = (pad or {}).get(edge)
    if not isinstance(u, (list, tuple)) or len(u) != 2:
        return 0.0
    along = size[0] if edge in ("Left", "Right") else size[1]
    return float(u[0]) * float(along or 0) + float(u[1])


def judge_rims(elements, pads, strokes):
    """(scrollPadding rows, badge rows). A row per scrolling edge of a visible ScrollingFrame whose padding there (its
    UIPadding child, scale against the frame's size) is less than the widest rim a UIStroke draws outside one of its
    children (the edge's first and last cards are cut there); a row per visible element that pokes outside its parent
    (a parent that does not clip it) and covers a sibling of that parent (one that does not hold the parent inside it,
    like a backdrop)."""
    by = {e["path"]: e for e in elements if e.get("path")}
    pad_of = {p.get("owner"): p for p in pads or []}
    rim = {}
    for s in strokes or []:
        o = stroke_overhang(s)
        holder = _parent(s.get("owner"))
        if o and holder and (holder not in rim or o > rim[holder][0]):
            rim[holder] = (o, s.get("owner"), s.get("thickness"), s.get("position") or "Outer")
    scroll = []
    for e in elements:
        if e.get("className") != "ScrollingFrame" or not e.get("visible") or e["path"] not in rim:
            continue
        o, owner, thick, pos = rim[e["path"]]
        size = (e.get("rect") or [0, 0, 0, 0])[2:]
        for edge in EDGES.get(e.get("scrollDirection") or "Y", ()):
            px = round(_pad_px(pad_of.get(e["path"]), edge, size), 2)
            if px < o - 0.01:
                scroll.append({"path": e["path"], "edge": edge, "padding": px, "rim": o, "stroke": owner,
                               "thickness": thick, "position": pos,
                               "what": "%s padding %g px is under the %g px rim of %s's UIStroke: the end cards' rims "
                                       "are cut" % (edge, px, o, owner.rsplit(".", 1)[-1] if owner else "a child")})
    badges = []
    for e in elements:
        r, par = e.get("rect"), by.get(_parent(e["path"]))
        if not e.get("visible") or not r or r[2] <= 0 or r[3] <= 0 or not par or not par.get("rect") or par.get("clips"):
            continue
        pr = par["rect"]
        if not (r[0] < pr[0] - 0.5 or r[1] < pr[1] - 0.5 or r[0] + r[2] > pr[0] + pr[2] + 0.5
                or r[1] + r[3] > pr[1] + pr[3] + 0.5):
            continue
        gp = _parent(par["path"])
        for s in elements:
            sr = s.get("rect")
            if s is par or _parent(s["path"]) != gp or not s.get("visible") or not sr or sr[2] <= 0 or sr[3] <= 0:
                continue
            if sr[0] <= pr[0] + 0.5 and sr[1] <= pr[1] + 0.5 and sr[0] + sr[2] >= pr[0] + pr[2] - 0.5 \
                    and sr[1] + sr[3] >= pr[1] + pr[3] - 0.5:
                continue                         # it holds the parent inside it: a backdrop, not a neighbour
            w = min(r[0] + r[2], sr[0] + sr[2]) - max(r[0], sr[0])
            h = min(r[1] + r[3], sr[1] + sr[3]) - max(r[1], sr[1])
            if w > 0.5 and h > 0.5:
                badges.append({"path": e["path"], "parent": par["path"], "covers": s["path"],
                               "overlap": [round(w, 1), round(h, 1)]})
    return scroll, badges


def judge_ui(sg_list, elements, start, min_button_px, min_text_px, cap, screen_fails=True, pads=None, strokes=None):
    """The UI findings for one set of ScreenGuis and their elements: (report, problems, shown). screen_fails False (the
    play client, whose window the probe minimizes so it stays off the person's screen: its ScreenGuis lay out at that
    small size, RB1451) reports off-screen and edge buttons without failing on them."""
    graph = selection_graph(elements, start)
    shown = [e for e in elements if e.get("interactive") and e.get("visible") and e.get("rect")
             and e["rect"][2] > 0 and e["rect"][3] > 0]
    sizes, advisory = _sizes(shown, min_button_px, min_text_px)
    safe, edge, off = _safe_area(sg_list, elements)
    laid = any(e.get("rect") and (e["rect"][2] > 0 or e["rect"][3] > 0) for e in elements)
    rep = {"screenGuis": safe, "elements": len(elements), "buttons": len(shown), "selection": graph, "sizes": sizes,
           "advisory": advisory, "edgeButtons": edge, "offScreen": off, "layoutRan": laid,
           "notInteractable": sorted(e["path"] for e in shown if e.get("interactable") is False),
           "truncated": len(elements) >= cap}
    scroll, badges = judge_rims(elements, pads, strokes)
    rep.update(scrollPadding=scroll, badgesOverSiblings=badges)
    problems = []
    if scroll:
        problems.append("%d scrolling edge(s) padded under their children's UIStroke rim" % len(scroll))
    if badges and screen_fails:
        problems.append("%d element(s) poking out of their parent over a sibling of it (a badge over the next button)"
                        % len(badges))
    if shown and not laid:
        problems.append("every element reads AbsoluteSize 0: Studio has not laid this UI out")
    if graph["unreachable"]:
        problems.append("%d button(s) unreachable from %s" % (len(graph["unreachable"]), graph["first"]))
    if graph["deadEnds"]:
        problems.append("%d button(s) reached with no way back to %s" % (len(graph["deadEnds"]), graph["first"]))
    if off and screen_fails:
        problems.append("%d button(s) partly or wholly off screen" % len(off))
    if graph["notSelectable"]:
        problems.append("%d visible button(s) not Selectable" % len(graph["notSelectable"]))
    if graph["deadLinks"]:
        problems.append("%d NextSelection link(s) to nothing selectable" % len(graph["deadLinks"]))
    if edge and screen_fails:
        problems.append("%d button(s) on the screen edge under ScreenInsets None" % len(edge))
    if not screen_fails:
        rep["screenNote"] = ("positions are at the play window's size, which the probe minimizes: off-screen and edge "
                             "buttons and badges over siblings are reported, not failed - audit them in edit mode "
                             "(console sets 1920 x 1080)")
    return rep, problems, shown


# ---------------------------------------------------------------- the play half: a client reads its actions and its UI
# ScreenGuis Roblox's own scripts put in PlayerGui (0.748 ExtraContent/scripts: FreeCameraInstaller, ControlModule,
# VRBaseCamera; the legacy chat's Chat and BubbleChat, measured in a play client, RB1451; ProximityPrompts, where
# ProximityPromptService draws a prompt's default button - found 2026-11-23: a publish preflight failed consoleReady on
# PlayerGui.ProximityPrompts.Prompt.TextButton): listed apart, never audited.
ROBLOX_GUIS = frozenset(("Freecam", "TouchGui", "VRBlurScreen", "Chat", "BubbleChat", "ProximityPrompts"))
CHUNK = 20000           # hex characters in one printed line of the client's answer, joined in order: a play client's
                        # Output carried lines of 3000-60000 intact (RB2072); 3000 put a 3000-element UI past its 500 + 500 kept lines
PROBE_SOURCE = r'''-- MifBridge controller audit probe: written by rbx_controller_audit for one play test, removed after. Read only.
local CAS = game:GetService("ContextActionService")
local GS = game:GetService("GuiService")
local UIS = game:GetService("UserInputService")
local HS = game:GetService("HttpService")
local player = game:GetService("Players").LocalPlayer
-- said at once, so an answer that never comes is told apart: never started, errored (-ERR) or out of time (RB2070)
print("%(base)s-RUN: " .. os.clock())
local t00 = os.clock()
local okRun, errRun = xpcall(function()
task.wait(%(wait)s)
-- a client blocked at its start (a heavy game script) runs this wait out before the game built its UI (RB2074): up to
-- %(grace)s s more for a ScreenGui that is not Roblox's own, then half a second for it to fill
local ROBLOX = { %(roblox)s }
local function gameGui(pg)
	for _, sg in (pg and pg:GetChildren() or {}) do
		if sg:IsA("ScreenGui") and not ROBLOX[sg.Name] then return true end
	end
	return false
end
local tg = os.clock()
while not gameGui(player:FindFirstChildOfClass("PlayerGui")) and os.clock() - tg < %(grace)s do task.wait(0.25) end
local uiWaited = os.clock() - tg
-- then until the game's UI stops being built: its descendant count the same for %(stable)s s, for at most %(settle)s s.
-- A HUD a client script builds behind WaitForChild calls was read half-built on a slow start, and one run of three on
-- an unchanged place failed (found 2026-11-22); a UI still changing at the deadline is reported unsettled, never judged.
local function uiCount(pg)
	local c = 0
	for _, sg in (pg and pg:GetChildren() or {}) do
		if sg:IsA("ScreenGui") and not ROBLOX[sg.Name] then c += 1 + #sg:GetDescendants() end
	end
	return c
end
local ts, last, since, settled = os.clock(), -1, os.clock(), false
if gameGui(player:FindFirstChildOfClass("PlayerGui")) then
	while os.clock() - ts < %(settle)s do
		local c = uiCount(player:FindFirstChildOfClass("PlayerGui"))
		if c ~= last then
			last, since = c, os.clock()
		elseif os.clock() - since >= %(stable)s then
			settled = true
			break
		end
		task.wait(0.25)
	end
end
local r = { actions = {}, prompts = {}, screenGuis = {}, elements = {}, pads = {}, strokes = {}, uiWaited = uiWaited,
	uiSettled = settled, uiSettleSeconds = os.clock() - ts, uiCount = last }
local ok, bound = pcall(function() return CAS:GetAllBoundActionInfo() end)
if ok then
	for name, info in bound do
		local types = {}
		for _, it in (info.inputTypes or {}) do table.insert(types, tostring(it)) end
		table.insert(r.actions, { name = name, inputTypes = types, priority = info.priorityLevel,
			stackOrder = info.stackOrder, createTouchButton = info.createTouchButton })
	end
else
	r.actionsError = tostring(bound)
end
for _, d in workspace:GetDescendants() do
	if d:IsA("ProximityPrompt") then
		table.insert(r.prompts, { path = d:GetFullName(), gamepadKeyCode = d.GamepadKeyCode.Name,
			keyboardKeyCode = d.KeyboardKeyCode.Name, enabled = d.Enabled, holdDuration = d.HoldDuration })
	end
end
local function full(o) return o and o:GetFullName() or nil end
local function r1(x) return math.floor(x * 10 + 0.5) / 10 end
-- the first 40 characters, never 40 bytes: a cut inside an emoji is not UTF-8 and stops JSONEncode (RB2070)
local function clip(s)
	if utf8.len(s) == nil then return (string.gsub(s, "[\128-\255]", "?")) end
	local stop = utf8.offset(s, 41)
	return if stop then string.sub(s, 1, stop - 1) else s
end
local cap, n = %(cap)s, 0
local pg = player:FindFirstChildOfClass("PlayerGui")
for _, sg in (pg and pg:GetChildren() or {}) do
	if sg:IsA("ScreenGui") then
		table.insert(r.screenGuis, { path = sg:GetFullName(), name = sg.Name, enabled = sg.Enabled,
			screenInsets = sg.ScreenInsets.Name, ignoreGuiInset = sg.IgnoreGuiInset,
			clipToDeviceSafeArea = sg.ClipToDeviceSafeArea, size = { sg.AbsoluteSize.X, sg.AbsoluteSize.Y } })
		for _, g in sg:GetDescendants() do
			if g:IsA("GuiObject") and n < cap then
				n += 1
				-- compact (a client's whole report must cross one RemoteEvent, RB2073): the ScreenGui as its index sgi,
				-- positions to 0.1 px, and behavior and next only when one is not the default
				local e = { path = g:GetFullName(), className = g.ClassName, sgi = #r.screenGuis,
					interactive = g:IsA("GuiButton") or g:IsA("TextBox"), ownVisible = g.Visible, selectable = g.Selectable,
					interactable = g.Interactable, selectionOrder = g.SelectionOrder, selectionGroup = g.SelectionGroup,
					rect = { r1(g.AbsolutePosition.X), r1(g.AbsolutePosition.Y), r1(g.AbsoluteSize.X), r1(g.AbsoluteSize.Y) } }
				local b = { Up = g.SelectionBehaviorUp.Name, Down = g.SelectionBehaviorDown.Name,
					Left = g.SelectionBehaviorLeft.Name, Right = g.SelectionBehaviorRight.Name }
				if b.Up ~= "Escape" or b.Down ~= "Escape" or b.Left ~= "Escape" or b.Right ~= "Escape" then e.behavior = b end
				local nx = { Up = full(g.NextSelectionUp), Down = full(g.NextSelectionDown),
					Left = full(g.NextSelectionLeft), Right = full(g.NextSelectionRight) }
				if next(nx) ~= nil then e.next = nx end
				if g:IsA("TextButton") or g:IsA("TextBox") or g:IsA("TextLabel") then
					e.textSize = g.TextSize
					e.textScaled = g.TextScaled
					e.textBounds = { r1(g.TextBounds.X), r1(g.TextBounds.Y) }
					e.text = clip(g.Text)
				end
				if g.ClipsDescendants then e.clips = true end
				if g:IsA("ScrollingFrame") then e.scrollDirection = g.ScrollingDirection.Name end
				table.insert(r.elements, e)
			elseif g:IsA("UIPadding") and #r.pads < cap then
				local function u(x) return { x.Scale, x.Offset } end
				table.insert(r.pads, { owner = full(g.Parent), Left = u(g.PaddingLeft), Right = u(g.PaddingRight),
					Top = u(g.PaddingTop), Bottom = u(g.PaddingBottom) })
			elseif g:IsA("UIStroke") and #r.strokes < cap then
				local s = { owner = full(g.Parent), ownerClass = g.Parent.ClassName, thickness = g.Thickness,
					mode = g.ApplyStrokeMode.Name, enabled = g.Enabled }
				pcall(function() s.position = g.BorderStrokePosition.Name end)
				pcall(function() s.sizing = g.StrokeSizingMode.Name end)
				table.insert(r.strokes, s)
			end
		end
	end
end
r.truncated = n >= cap
r.tenFoot = GS:IsTenFootInterface()
r.displaySize = GS.ViewportDisplaySize.Name
r.gamepadEnabled = UIS.GamepadEnabled
r.viewport = { workspace.CurrentCamera.ViewportSize.X, workspace.CurrentCamera.ViewportSize.Y }
pcall(function() r.preferredInput = UIS.PreferredInput.Name end)
r.selected = full(GS.SelectedObject)
-- holds (N58): each input the game's hook marks as repeating is held, a screen opened over it through the hook, and
-- released there; the hook's tick count says whether the action still repeats a second later (HOLD_STEPS in Python)
if %(holds)s then
	local hd = { rows = {} }
	r.holds = hd
	local hook = game
	for name in string.gmatch("%(hook)s", "[^%%.]+") do
		hook = hook and (hook:FindFirstChild(name) or (hook == game and game:FindService(name)) or nil)
	end
	if hook == nil or not hook:IsA("BindableFunction") then
		hd.error = "no BindableFunction at %(hook)s"
	else
		local function ask(what)
			local okA, v = pcall(function() return hook:Invoke(what) end)
			if not okA then return nil, tostring(v) end
			return v, nil
		end
		local list, lerr = ask("repeating")
		local okV, vi = pcall(function() return UIS:CreateVirtualInput() end)
		if lerr then
			hd.error = "the hook failed on repeating: " .. lerr
		elseif type(list) ~= "table" then
			hd.error = "the hook answered repeating with " .. typeof(list) .. ", not a list"
		elseif not okV or vi == nil then
			hd.error = "UserInputService:CreateVirtualInput failed: " .. tostring(vi)
		else
			local inset = GS:GetGuiInset()
			local cam = workspace.CurrentCamera
			local function ticks()
				local v = ask("ticks")
				return if type(v) == "number" then v else nil
			end
			for i, item in list do
				if i > %(maxholds)s then hd.cut = #list; break end
				local row = { input = if typeof(item) == "Instance" then item:GetFullName() else tostring(item) }
				table.insert(hd.rows, row)
				local press, release
				local key = if type(item) == "string" then string.match(item, "^key:(.+)$") else nil
				local btn = if typeof(item) == "Instance" then item else nil
				if btn == nil and type(item) == "string" and string.sub(item, 1, 7) == "button:" then
					btn = pg
					for name in string.gmatch(string.sub(item, 8), "[^%%.]+") do btn = btn and btn:FindFirstChild(name) end
				end
				if key then
					local okK, kc = pcall(function() return Enum.KeyCode[key] end)
					if not okK or kc == nil then row.error = "no KeyCode " .. key; continue end
					row.kind = "key"
					press = function() vi:SendKey(true, kc, false) end
					release = function() vi:SendKey(false, kc, false) end
				elseif item == "mouse" or btn ~= nil then
					local at
					if btn ~= nil then
						if not btn:IsA("GuiObject") then row.error = "not a GuiObject"; continue end
						row.kind = "button"
						-- SendMouseButton takes a screen position, inset included: the InputObject reads it less the
						-- inset (RB2740)
						at = btn.AbsolutePosition + btn.AbsoluteSize / 2 + inset
					else
						row.kind = "mouse"
						at = cam.ViewportSize / 2
					end
					row.at = { at.X, at.Y }
					press = function() vi:SendMouseButton(at, Enum.UserInputType.MouseButton1, true, 0) end
					release = function() vi:SendMouseButton(at, Enum.UserInputType.MouseButton1, false, 0) end
				else
					row.error = if type(item) == "string" and string.sub(item, 1, 7) == "button:"
						then "no button at PlayerGui." .. string.sub(item, 8) else "not key:<KeyCode>, button:<path>, mouse or a GuiButton"
					continue
				end
				local t0 = ticks()
				if t0 == nil then row.error = "the hook answered ticks with no number"; continue end
				local okP, eP = pcall(press)
				if not okP then row.error = "press failed: " .. tostring(eP); continue end
				task.wait(%(held)s)
				local t1 = ticks()
				local function topAt()
					if row.kind == "key" then return nil end
					return pg:GetGuiObjectsAtPosition(row.at[1] - inset.X, row.at[2] - inset.Y)[1]
				end
				local topBefore = topAt()
				row.ticksHeld = (t1 or t0) - t0
				local _o, oerr = ask("open_screen")
				row.openError = oerr
				task.wait(0.3)
				if row.kind ~= "key" then
					-- the screen must lie over the release point, or nothing was tested (a minimized client's small view
					-- left a button outside it, RB2743)
					local top = topAt()
					row.under = if top then top:GetFullName() else nil
					row.underBefore = if topBefore then topBefore:GetFullName() else nil
					-- the screen is what is on top now, and was not before it opened
					row.covered = top ~= nil and top ~= topBefore and top ~= btn and not (btn ~= nil and top:IsDescendantOf(btn))
				end
				pcall(release)
				task.wait(0.2)
				local ta = ticks()
				task.wait(%(after)s)
				local tb = ticks()
				row.ticksAfter = (tb or ta or 0) - (ta or 0)
				row.runningAfter = row.ticksAfter > 0
				ask("close_screen")
				task.wait(0.3)
				if row.runningAfter then
					-- the player's own way out: press and release again with no screen open
					pcall(press); task.wait(0.2); pcall(release); task.wait(0.5)
					local tc = ticks(); task.wait(0.5)
					row.stoppedByRepress = (ticks() or 0) - (tc or 0) == 0
				end
			end
		end
	end
end
-- presses (N75): each visible GuiButton inside a panel (a frame under its ScreenGui, not the ScreenGui's own child) is
-- clicked through CreateVirtualInput, and what changed is told apart: inside the button's own panel, or only in another
-- ScreenGui or another of its ScreenGui's top frames (a toast elsewhere is extra, never the only answer)
if %(presses)s then
	local pr = { rows = {} }
	r.presses = pr
	local okV, vi = pcall(function() return UIS:CreateVirtualInput() end)
	if not okV or vi == nil then
		pr.error = "UserInputService:CreateVirtualInput failed: " .. tostring(vi)
	else
		local inset = GS:GetGuiInset()
		local function sig(d)
			local t = { tostring(d.Visible), tostring(d.Position), tostring(d.Size), tostring(d.BackgroundColor3),
				tostring(d.BackgroundTransparency), tostring(d.ZIndex) }
			if d:IsA("TextLabel") or d:IsA("TextButton") or d:IsA("TextBox") then
				table.insert(t, d.Text); table.insert(t, tostring(d.TextColor3)); table.insert(t, tostring(d.TextTransparency))
			end
			if d:IsA("ImageLabel") or d:IsA("ImageButton") then
				table.insert(t, tostring(d.Image)); table.insert(t, tostring(d.ImageColor3))
				table.insert(t, tostring(d.ImageTransparency))
			end
			return table.concat(t, "|")
		end
		-- every game ScreenGui's GuiObjects and UIStrokes by full name: {path = signature}
		local function snap()
			local s = {}
			for _, sg in (pg and pg:GetChildren() or {}) do
				if sg:IsA("ScreenGui") and not ROBLOX[sg.Name] then
					s[sg:GetFullName()] = "E" .. tostring(sg.Enabled)
					for _, d in sg:GetDescendants() do
						if d:IsA("GuiObject") then
							s[d:GetFullName()] = sig(d)
						elseif d:IsA("UIStroke") then
							s[d:GetFullName()] = tostring(d.Enabled) .. "|" .. tostring(d.Color) .. "|" .. tostring(d.Thickness)
						end
					end
				end
			end
			return s
		end
		local function diff(a, b)
			local out = {}
			for k, v in a do if b[k] ~= v then table.insert(out, k) end end
			for k, _ in b do if a[k] == nil then table.insert(out, k) end end
			return out
		end
		local function shown(g)
			local o = g
			while o and not o:IsA("LayerCollector") do
				if o:IsA("GuiObject") and not o.Visible then return false end
				o = o.Parent
			end
			return o ~= nil and o:IsA("ScreenGui") and o.Enabled
		end
		-- what changes with no press (a ticking clock, a spinning icon) is left out of every press's count
		local idle0 = snap()
		task.wait(%(presswait)s)
		local volatile = {}
		for _, k in diff(idle0, snap()) do volatile[k] = true end
		pr.volatile = 0
		for _ in volatile do pr.volatile += 1 end
		local list = {}
		for _, sg in (pg and pg:GetChildren() or {}) do
			if sg:IsA("ScreenGui") and not ROBLOX[sg.Name] then
				for _, d in sg:GetDescendants() do
					if d:IsA("GuiButton") and d.Parent ~= sg then table.insert(list, d) end
				end
			end
		end
		pr.found = #list
		for i, btn in list do
			if i > %(maxpresses)s then pr.cut = #list; break end
			local sg = btn:FindFirstAncestorWhichIsA("ScreenGui")
			local panel = btn
			while panel.Parent ~= sg do panel = panel.Parent end
			local row = { button = btn:GetFullName(), panel = panel:GetFullName(), layer = sg:GetFullName() }
			table.insert(pr.rows, row)
			if btn.Parent == nil or not shown(btn) then row.skipped = "hidden"; continue end
			local at = btn.AbsolutePosition + btn.AbsoluteSize / 2
			local top = pg:GetGuiObjectsAtPosition(at.X, at.Y)[1]
			if top == nil or not (top == btn or top:IsDescendantOf(btn)) then
				row.skipped = "covered"
				row.under = if top then top:GetFullName() else nil
				continue
			end
			local before = snap()
			local p = at + inset
			local okP, eP = pcall(function()
				vi:SendMouseButton(p, Enum.UserInputType.MouseButton1, true, 0)
				task.wait(0.05)
				vi:SendMouseButton(p, Enum.UserInputType.MouseButton1, false, 0)
			end)
			if not okP then row.error = "press failed: " .. tostring(eP); continue end
			task.wait(%(presswait)s)
			local pp, own, other, where = row.panel, 0, 0, {}
			for _, k in diff(before, snap()) do
				if not volatile[k] then
					if k == pp or string.sub(k, 1, #pp + 1) == pp .. "." then
						own += 1
					else
						other += 1
						if #where < 5 then table.insert(where, k) end
					end
				end
			end
			row.inPanel, row.elsewhere, row.changedElsewhere = own, other, where
		end
	end
end
r.readSeconds = os.clock() - t00
''' + CLIENT_EMIT + r'''
print("%(base)s-DONE: " .. (os.clock() - t00))
end, function(e) return tostring(e) .. " | " .. debug.traceback("", 2) end)
if not okRun then print("%(base)s-ERR: " .. string.gsub(tostring(errRun), "\n", " | ")) end
'''


def _find_mark(x, mark, found):
    if isinstance(x, str) and mark in x:
        found.append(x[x.index(mark) + len(mark):])
    elif isinstance(x, dict):
        for v in x.values():
            _find_mark(v, mark, found)
    elif isinstance(x, list):
        for v in x:
            _find_mark(v, mark, found)
    return found


def join_parts(found, run=None):
    """The client's answer from its printed parts ("i/n@run:text", or an older "i/n:text"), in CLIENT_EMIT's ~XX text or
    as hex: (text, error). With run, when any part names a run, only the parts that name this one are taken (another
    run's, and a part naming none, are left out and counted in the error when nothing of this run came back); an Output
    whose parts name no run at all (a script from before the run nonce) is read as before."""
    parts, total, foreign = {}, None, 0
    tagged = any("@" in f.partition(":")[0] for f in found if isinstance(f, str))
    for f in found:
        head, sep, body = f.partition(":")
        if not sep or "/" not in head:
            continue
        head, _at, who = head.partition("@")
        if run is not None and tagged and who != run:
            foreign += 1
            continue
        i, _s, n = head.partition("/")
        if not (i.strip().isdigit() and n.strip().isdigit()):
            continue
        parts.setdefault(int(i), body.split("\n")[0].rstrip("\r"))
        total = int(n)
    if not total:
        return None, "the client printed no audit line" + (
            " of this run (%d part(s) of another run were left out)" % foreign if foreign else "")
    missing = [i for i in range(1, total + 1) if i not in parts]
    if missing:
        return None, "the client's answer arrived without part(s) %s of %d" % (missing[:5], total)
    body = "".join(parts[i] for i in range(1, total + 1))
    try:
        if body.startswith("{"):
            # CLIENT_EMIT's text: every byte outside its plain set is ~ and two hex digits
            raw = re.sub(rb"~([0-9a-f]{2})", lambda m: bytes([int(m.group(1), 16)]), body.encode("latin-1"))
            return raw.decode("utf-8"), None
        return bytes.fromhex(body).decode("utf-8"), None          # an earlier probe's all-hex answer
    except ValueError as exc:
        return None, "the client's answer did not decode as UTF-8: %s" % exc


def _device_before(T):
    """{"device", "resolution"} the device simulator had before the audit set its console; device None when it could not
    be read, resolution None for the default device (Studio refuses a resolution with no device set)."""
    svc = "StudioDeviceSimulatorService"
    g = T.rbx_call(svc, "GetDeviceAsync", timeout=30)
    dev = g.get("value") if g.get("ok") and isinstance(g.get("value"), str) and g.get("value") else None
    res = None
    if dev and dev != "default":
        r = T.rbx_call(svc, "GetResolutionAsync", timeout=30)
        v = r.get("value") if r.get("ok") else None
        if isinstance(v, dict) and all(isinstance(v.get(k), int) and not isinstance(v.get(k), bool) for k in ("x", "y")):
            res = [v["x"], v["y"]]
    return {"device": dev, "resolution": res}


def _device_restore(T, before):
    """Put the device simulator back as _device_before read it: stopped when it was the default or could not be read,
    else that device and its resolution. -> what the answer's console says (stopped, restoredTo, restored)."""
    dev = (before or {}).get("device")
    if not dev or dev == "default":
        stop = T.rbx_device_sim("stop")
        return {"stopped": bool(stop.get("ok")), "restoredTo": "default" if dev else "default (the device before the "
                "audit could not be read)", "restored": bool(stop.get("ok")) and dev == "default"}
    s = T.rbx_device_sim("set", device=dev)
    out = {"stopped": False, "restoredTo": dev, "restored": bool(s.get("ok"))}
    res = before.get("resolution")
    if s.get("ok") and res:
        r = T.rbx_device_sim("resolution", width=res[0], height=res[1])
        out.update(restoredResolution=res, restored=bool(r.get("ok")))
    if not out["restored"]:
        out["restoreError"] = (s.get("error") if not s.get("ok") else "the resolution %s was not put back" % (res,))
    return out


def play_half(seconds=None, wait=3, cap=1000, players=1, diag=None, hook=None, presses=False):
    """A LocalScript in StarterPlayerScripts for one play test (full mode or MifBridge's scratch place), removed after;
    the client prints its bound actions and prompts. players 1 is a one-player multiplayer test, None a solo test (the
    console run: its client is the simulated device, RB1881). hook, a dot path to the game's BindableFunction, adds the
    hold check (N58); presses the press check (N75). (result, error, wrote)."""
    # holds and presses need the client's view at its own size: a minimized client's view is a few hundred px, so a
    # screen opened over it never reached a button outside it (RB2743); the window is kept behind the person's, unfocused
    return play_client(probe_source(wait, cap, hook, presses), PROBE_MARK, PROBE_SCRIPT,
                       seconds or window(wait, cap, hook, presses), players=players, diag=diag,
                       windows="behind" if hook or presses else None)


# The hold check: each input is held HOLD_HELD s (its ticks must rise: the hold started), a screen is
# opened over it through the hook, it is released 0.3 s later, and the ticks are counted over HOLD_AFTER s from 0.2 s
# after the release: any tick there is a hold still running. At most MAX_HOLDS inputs per run.
HOLD_HELD, HOLD_AFTER, MAX_HOLDS = 0.6, 1.0, 8
HOLD_SECONDS = 4.5      # one input's steps, the re-press of a runaway hold included
# The press check: each visible GuiButton inside a panel is clicked (press, 0.05 s, release) and the
# client's game ScreenGuis are compared PRESS_WAIT s later; what changed in a PRESS_WAIT s window with no press is left
# out. At most MAX_PRESSES buttons per run, in PlayerGui order.
PRESS_WAIT, MAX_PRESSES = 0.6, 24
PRESS_SECONDS = 1.0     # one button's steps: the snapshot, the click, the wait and the second snapshot


def probe_source(wait, cap, hook=None, presses=False):
    """The audit's client probe for one run."""
    return PROBE_SOURCE % {"wait": wait, "mark": PROBE_MARK, "base": PROBE_MARK.rstrip(":"), "cap": int(cap),
                           "chunk": CHUNK, "grace": UI_GRACE, "stable": UI_STABLE, "settle": UI_SETTLE,
                           "roblox": ", ".join('["%s"] = true' % n for n in sorted(ROBLOX_GUIS)),
                           "holds": "true" if hook else "false", "hook": hook or "", "maxholds": MAX_HOLDS,
                           "held": HOLD_HELD, "after": HOLD_AFTER, "presses": "true" if presses else "false",
                           "presswait": PRESS_WAIT, "maxpresses": MAX_PRESSES}


UI_GRACE = 4            # seconds past play_wait the probe waits for the game's first ScreenGui (RB2074)
UI_STABLE = 2           # seconds the game's UI must hold the same descendant count to count as built
UI_SETTLE = 12          # seconds at most the probe waits for that, after the first ScreenGui


def window(wait, cap, hook=None, presses=False):
    """The play window the audit asks for: the probe's wait, up to UI_GRACE more for the game's first ScreenGui, then
    time to read the UI and print it, with a hook the holds and with presses the press check. The window starts
    once the client probe said it runs (the play probe's client wait, plugin 0.17.8), which is when this script starts
    too; reading and printing 1505 elements took 0.11 s after the wait live (RB2071)."""
    return (wait + UI_GRACE + UI_SETTLE + 5 + int(cap) // 500 + (int(MAX_HOLDS * HOLD_SECONDS) + 2 if hook else 0)
            + (int(MAX_PRESSES * PRESS_SECONDS + PRESS_WAIT) + 2 if presses else 0))


def judge_holds(h):
    """(report, problems) from the client's holds table: a hold still running after its release over a screen fails,
    as do a hold that never started (nothing was tested), an input the probe could not use, a missing hook and a hook
    that marks nothing as repeating (an empty check is not a clean one)."""
    if not isinstance(h, dict):
        return {"error": "the client sent no holds table"}, ["holds: the client sent no holds table"]
    rows = [x for x in h.get("rows") or [] if isinstance(x, dict)]
    rep = {"rows": rows, "hook": h.get("hook")}
    if h.get("error"):
        return dict(rep, error=h["error"]), ["holds: " + h["error"]]
    if not rows:
        return rep, ["holds: the hook marked no input as repeating: nothing was held"]
    probs = []
    running = [x["input"] for x in rows if x.get("runningAfter")]
    never = [x["input"] for x in rows if not x.get("error") and not x.get("runningAfter") and not x.get("ticksHeld")]
    bad = ["%s (%s)" % (x["input"], x["error"]) for x in rows if x.get("error")]
    bad += ["%s (the screen the hook opened did not cover it: %s was on top)" % (x["input"], x.get("under") or "nothing")
            for x in rows if not x.get("error") and x.get("covered") is False]
    if running:
        probs.append("holds: %d hold(s) still running %g s after their release over a screen: %s" % (
            len(running), HOLD_AFTER, ", ".join(running)))
    if never:
        probs.append("holds: %d hold(s) never started (the hook's ticks did not rise while held): %s" % (
            len(never), ", ".join(never)))
    if bad:
        probs.append("holds: %d input(s) could not be held: %s" % (len(bad), "; ".join(bad)))
    if h.get("cut"):
        rep["cut"] = "the hook marked %s inputs; the first %d were held" % (h["cut"], MAX_HOLDS)
    rep.update(stillRunning=running, neverStarted=never)
    return rep, probs


def judge_presses(p):
    """(report, problems) from the client's presses table: a button whose press changed nothing in its own panel and
    something in another ScreenGui or another of its ScreenGui's top frames is answeredElsewhere and fails (the internal notes
    L142); one whose press changed nothing anywhere is listed as noVisibleAnswer, not failed (a server's answer can take
    longer than the wait); a run that pressed nothing fails, as does a client that could not press."""
    if not isinstance(p, dict):
        return {"error": "the client sent no presses table"}, ["presses: the client sent no presses table"]
    rows = [x for x in p.get("rows") or [] if isinstance(x, dict)]
    rep = {"rows": rows, "found": p.get("found"), "volatile": p.get("volatile")}
    if p.get("error"):
        return dict(rep, error=p["error"]), ["presses: " + p["error"]]
    pressed = [x for x in rows if not x.get("skipped") and not x.get("error") and isinstance(x.get("inPanel"), int)]
    elsewhere = [x["button"] for x in pressed if x["inPanel"] == 0 and (x.get("elsewhere") or 0) > 0]
    silent = [x["button"] for x in pressed if x["inPanel"] == 0 and not x.get("elsewhere")]
    rep.update(pressed=len(pressed), answeredElsewhere=elsewhere, noVisibleAnswer=silent,
               skipped=[{"button": x.get("button"), "why": x["skipped"], "under": x.get("under")}
                        for x in rows if x.get("skipped")])
    if p.get("cut"):
        rep["cut"] = "%s buttons sit in panels; the first %d were pressed" % (p["cut"], MAX_PRESSES)
    probs = []
    if elsewhere:
        where = ["%s -> %s" % (x["button"], ", ".join(x.get("changedElsewhere") or []))
                 for x in pressed if x["button"] in elsewhere]
        probs.append("presses: %d button(s) answer only outside their own panel, where a player pressing them may not "
                     "look: %s" % (len(elsewhere), "; ".join(where[:5])))
    bad = ["%s (%s)" % (x.get("button"), x["error"]) for x in rows if x.get("error")]
    if bad:
        probs.append("presses: %d button(s) could not be pressed: %s" % (len(bad), "; ".join(bad)))
    if not pressed:
        probs.append("presses: no button inside a panel was pressed (%s found, %d skipped): nothing was checked" % (
            p.get("found") or 0, len(rep["skipped"])))
    return rep, probs


def _messages(x, rows):
    """Every Output row ({message, type, t}) under a probe's clients, in order."""
    if isinstance(x, dict):
        if isinstance(x.get("message"), str):
            rows.append((x["message"], x.get("type"), x.get("t")))
        else:
            for v in x.values():
                _messages(v, rows)
    elif isinstance(x, list):
        for v in x:
            _messages(v, rows)
    return rows


def silence(p, mark, name, seconds):
    """Why a client printed no answer under mark, read from the play probe's answer p: (code, sentence). Codes: probe-
    failed (no client reported and the test failed), no-client (no client reported: the probe's clientsWarning says
    why), report-lost (the client said it runs and its report never arrived), script-error (the script errored in the client: its -ERR line, or Roblox's error row naming the script),
    out-of-time (the script said -RUN but printed nothing before the client reported), output-cut (the client's Output
    was cut to its first and last lines and the answer's lines were in the middle), never-ran (the client reported and
    the script never said -RUN)."""
    p = p if isinstance(p, dict) else {}
    clients = [c for c in p.get("clients") or [] if isinstance(c, dict)]
    if not clients:
        if not p.get("ok"):
            return "probe-failed", "the play test failed before any client reported: %s" % (p.get("error") or "no error given")
        missing = (p.get("clientsMissing") or {}).get("players") or []
        if missing and all(isinstance(m, dict) and m.get("ready") for m in missing):
            # it said it runs, so it existed; its report (its whole Output, the answer in it) never reached the server
            return "report-lost", ("the client started (it said it runs) but its report never reached the play server: "
                                   "the report carries the client's whole Output, and one too large to send is lost "
                                   "(RB2073) - lower max_elements")
        return "no-client", "no client reported: %s" % (p.get("clientsWarning") or "the probe says nothing of its clients")
    rows = _messages(clients, [])
    base = mark.rstrip(":")
    for m, _ty, _t in rows:
        if base + "-ERR:" in m:
            return "script-error", "the probe script errored in the client: %s" % m.split(base + "-ERR:", 1)[1].strip()[:500]
    for i, (m, _ty, _t) in enumerate(rows):
        if "." + name + "'" in m or "." + name + ":" in m:
            err = next((rows[j][0] for j in range(i, max(-1, i - 5), -1) if rows[j][1] == "MessageError"), m)
            return "script-error", "the probe script errored in the client: %s" % err[:500]
    ran = [t for m, _ty, t in rows if base + "-RUN:" in m]
    cut = [c.get("logsCut") for c in clients if isinstance(c.get("logsCut"), dict)]
    if cut and (ran or not any(base in m for m, _ty, _t in rows)):
        return "output-cut", ("the client's Output came back cut (%s): the answer's lines were among those dropped - lower "
                              "max_elements, or the game's own printing" % cut[0].get("keptAs", "first and last lines"))
    if ran:
        return "out-of-time", ("the probe script started in the client but printed nothing before the client reported at "
                               "the end of the %gs window: the UI took longer to read and print - lower max_elements" % seconds)
    return "never-ran", ("the client reported, but the probe script never started in it (no %s-RUN line): a script in "
                         "StarterPlayerScripts did not reach this client" % base)


def clear_leftover(T, path, mark):
    """A probe LocalScript already at path (found 2026-11-26: a caller killed mid play test left MifControllerProbe in
    StarterPlayerScripts, so every later audit refused, consoleReady read unknown and rbx_place_publish was blocked).
    One whose source is ours - it prints mark, and carries CLIENT_EMIT's run tag - is removed when no play test of this
    daemon's port holds the machine play lock (an audit running now would lose its probe); a script of that name that is
    not ours is never touched. -> (what was cleared, None) or (None, the refusal with the next step)."""
    from mif_roblox import launch as L
    got = T.rbx_get_props(path, ["Source"]) if hasattr(T, "rbx_get_props") else {}
    src = ((got or {}).get("values") or {}).get("Source") if (got or {}).get("ok") else None
    if not isinstance(src, str) or mark not in src or not re.search(r"@[0-9a-f]{12}:", src):
        return None, ("%s is already there and is not MifBridge's own probe (its source does not print %s), so it is left "
                      "alone and nothing was run: if it is a leftover, remove it with rbx_destroy('%s') and run the audit "
                      "again" % (path, mark.rstrip(":"), path))
    lock = L.read_play_lock() or {}
    port = getattr(T, "PORT", None)
    if lock and port is not None and lock.get("port") == port and L.holder_alive(lock):
        return None, ("%s is MifBridge's probe and a play test of this Studio's daemon (port %s, pid %s) holds the play "
                      "lock now, so it may be that audit's: nothing was run. Wait for it, or if no audit runs, remove it "
                      "with rbx_destroy('%s')" % (path, port, lock.get("pid"), path))
    d = T.rbx_destroy(path)
    if not (d or {}).get("ok"):
        return None, ("%s is a probe an earlier audit left (killed mid play test) and it could not be removed (%s): "
                      "nothing was run; remove it with rbx_destroy('%s')" % (path, (d or {}).get("error"), path))
    return {"path": path, "why": "a probe an earlier audit left (its caller ended mid play test) was removed first"}, None


def play_client(source, mark, name, seconds, players=1, diag=None, windows=None):
    """Run source as a LocalScript in StarterPlayerScripts for one play test (full mode or MifBridge's scratch place) and
    read back the table it printed with CLIENT_EMIT under mark; the script is removed after. players 1 is a one-player
    multiplayer test (its client is a separate desktop window: the device simulator does not carry into it, RB1444);
    players None is solo play in the edit Studio, whose client IS the simulated device. When nothing comes back, the
    error says why (silence) and diag, a dict, gets why and the probe's client facts. (result, error, wrote)."""
    T = _T()
    diag = diag if isinstance(diag, dict) else {}
    run = uuid.uuid4().hex[:12]                  # this run's parts name it (CLIENT_EMIT), and only they are joined
    source = source.replace("__MIFRUN__", run)
    parent = "StarterPlayer.StarterPlayerScripts"
    path = parent + "." + name
    cleared = None
    if _T()._rbx("resolve", target=V.split_path(path)).get("ok"):
        cleared, why = clear_leftover(T, path, mark)
        if why:
            return None, why, None
    mk = T.rbx_create("LocalScript", parent, name=name, props={})
    if not mk.get("ok"):
        return None, "the probe LocalScript could not be written (%s): the play half needs full mode" % mk.get("error"), None
    wrote = {"path": path, "removed": False}
    if cleared:
        wrote["clearedLeftover"] = cleared
    try:
        w = T.rbx_script_write(path, source)
        if not w.get("ok"):
            return None, "the probe source was not written: %s" % w.get("error"), wrote
        kw = {"players": players} if players is not None else {}
        if windows:
            kw["client_windows"] = windows
        p = T.rbx_play_probe("log", seconds=seconds, clients=True, timeout=seconds + 120, **kw)
        p = p if isinstance(p, dict) else {}
        diag.update(seconds=seconds, clientWait=p.get("clientWait"), clientsWarning=p.get("clientsWarning"),
                    pluginWarning=p.get("pluginWarning"))
        text, err = join_parts(_find_mark(p.get("clients"), mark, []), run=run)
        if err:
            if err.startswith("the client printed no audit line"):
                diag["why"], why = silence(p, mark, name, seconds)
                return None, "the client printed no audit line: %s" % why, wrote
            diag["why"] = "parts-missing"
            return None, err, wrote
        try:
            return json.loads(text), None, wrote
        except ValueError as exc:
            diag["why"] = "bad-json"
            return None, "the client's audit line did not parse: %s" % exc, wrote
    finally:
        d = T.rbx_destroy(path)
        wrote["removed"] = bool(d.get("ok"))


# What Roblox itself binds through ContextActionService: every action name its own client scripts pass to BindAction or
# BindActionAtPriority, read from Studio 0.748's ExtraContent/scripts (CoreScripts and StarterPlayerScripts' PlayerModule;
# literal names, named constants and the camera module's Rbx* names), so a touch, VR, vehicle or Freecam one is known
# before a client happens to bind it (RB1442, RB1450). A name starting with one of BUILTIN_PREFIXES is Roblox's too. Listed
# apart as robloxActions, never judged: a keyboard-only one (FreecamToggle on P) or a touch-only one
# (DynamicThumbstickAction) is Roblox's, not the game's.
BUILTIN_ACTIONS = frozenset((
    "DynamicThumbstickAction", "EmotesMenuActivateEmoteSlotAction", "EmotesMenuCloseAction",
    "EmotesMenuLeaveMenuDontSinkInputAction", "EmotesMenuPlaySelectedAction", "EmotesMenuSelectionAction",
    "EmotesMenuToggleAction", "EnableKeyboardUINavigation", "MouseLockSwitchAction", "MoveThumbstick", "OpenIGMenu",
    "OpenVRMenu", "ScreenshotHudGamepadCamera", "ScreenshotHudGamepadClose", "ScrollSelectedElement", "VRJumpAction",
    "arrowSteerLeft", "arrowSteerRight", "jumpAction", "moveBackwardAction", "moveDirectionalButton", "moveForwardAction",
    "moveLeftAction", "moveRightAction", "moveThumbstick", "throttleAccel", "throttleDeccel",
    # named constants in the core scripts (menus, prompts, the player list, the avatar context menu)
    "AvatarContextMenuStopInput", "CloseMoreMenuAction", "ContextActionMenuRenderArrow", "CoreScriptPromptCreatorCancel",
    "CoreScriptPromptCreatorSelect", "DontLeaveAction", "EducationalPopupConfirm", "ErrorPromptAction",
    "EscapeAvatarContextMenu", "FirstPersonToggleAction", "GamepadOpenAvatarContextMenu", "JumpAction", "LeaveGame",
    "LeaveGameAction", "LeaveGameCancelAction", "LeavePromptConfirm", "LeftStickTogglesMoreMenuAction",
    "ModalDialogDismiss", "NavigateBack", "Open Dev Console", "ReporSentDialogSinkAction", "ResetCharacter",
    "ResetCharacterAction", "Show Quick Profiler", "SwitchPageAvatarContextMenu", "ToggleDeveloperConsole",
    "ToggleInGameMenu", "TogglePerformanceStats", "doNothingAButtonPromptCreator", "player_contextual_menu_close_action",
    "settings_dropdown_close_action", "sidesheet_leave", "sidesheet_respawn"))
BUILTIN_PREFIXES = ("Rbx", "RBX", "Freecam", "TopBarGamepad", "TopBarDialog", "InGameMenu", "EmotesMenu")


def is_builtin_action(name):
    n = str(name or "")
    return n in BUILTIN_ACTIONS or n.startswith(BUILTIN_PREFIXES)


def judge_actions(res):
    """(the game's actions no gamepad input reaches, enabled prompts with no GamepadKeyCode, Roblox's own action names)."""
    actions, prompts = res.get("actions") or [], res.get("prompts") or []
    builtin = sorted(a["name"] for a in actions if is_builtin_action(a.get("name")))
    no_pad = [a["name"] for a in actions if not is_builtin_action(a.get("name"))
              and not any(is_gamepad_input(t) for t in a.get("inputTypes") or [])]
    prompt_no_pad = [p["path"] for p in prompts if p.get("enabled") is not False
                     and not is_gamepad_input(p.get("gamepadKeyCode"))]
    return no_pad, prompt_no_pad, builtin


# ---------------------------------------------------------------- the tool
@_tool
def rbx_controller_audit(root: Any = "StarterGui", play: bool = False, console: str = None, screenshot: bool = False,
                         min_button_px: float = None, min_text_px: float = None, max_elements: int = 1000,
                         play_wait: float = 3, hold_hook: str = None, press_buttons: bool = False) -> dict:
    "Roblox Studio: can a gamepad drive this game's UI and actions? For every ScreenGui under root (default StarterGui) and each visible GuiButton and TextBox: Selectable, its NextSelectionUp/Down/Left/Right links (a link to a hidden, missing or non-Selectable object is a deadLink), and for a direction with no link the neighbour Studio's automatic navigation would pick - MODELED (nearest selectable that way, a SelectionGroup with SelectionBehavior Stop kept), because a plugin cannot drive that navigation; then every button reachable from the first selected object (GuiService.SelectedObject, else lowest SelectionOrder then top-left) and the moves each takes. Sizes and text sizes are reported; Roblox's console guidelines give no pixel numbers, so min_button_px and min_text_px only add advisory rows and never fail. ScreenInsets per ScreenGui (DeviceSafeInsets reads IgnoreGuiInset true, RB1416) and buttons a ScreenInsets None gui puts on the screen edge. console 'xbox', 'ps4' or 'ps5' sets the device simulator first (IsTenFootInterface true, ViewportDisplaySize Large, 1920 x 1080: RB1443), measures under it, captures with screenshot true, and stops the simulator after. play true also runs a play test - one player, or solo with console (below) - (full mode or MifBridge's scratch place; a LocalScript written to StarterPlayerScripts for that test and removed after; one an earlier audit left, its caller ended mid play test, is removed first when its source is MifBridge's own probe and no play test of this port holds the play lock, and clearedLeftover says so - a script of that name that is not ours is never touched, and the refusal names rbx_destroy) whose client, play_wait seconds in (default 3: time for the game's own LocalScripts to build their UI), reports what it shows: the same checks on every ScreenGui in its PlayerGui (playerGui - a UI a LocalScript builds at run time is audited there; Roblox's own Freecam, TouchGui, VRBlurScreen, Chat, BubbleChat and ProximityPrompts (the default prompt's button) are listed apart), every ContextActionService bound action (GetAllBoundActionInfo) with the game's actions no gamepad input reaches (Roblox's own, BindAction names from its client scripts, are listed apart as robloxActions), and every ProximityPrompt by the GamepadKeyCode it holds (a new prompt holds ButtonX, RB1473, so one that sets only KeyboardKeyCode still passes); with console the test is a SOLO play test instead, whose client runs as the simulated console - IsTenFootInterface true, ViewportDisplaySize Large, a 1919 x 1079 view under an 87 px top bar, but GamepadEnabled false and PreferredInput KeyboardAndMouse, so a UI shown only when GamepadEnabled is true stays hidden there (RB1881) - and clientTenFoot, clientDisplaySize, clientViewport and clientPreferredInput say what it read; a client that does not read ten-foot fails the audit (a one-player multiplayer test's client is a desktop window the simulator never reaches, RB1444). 'Nothing was audited' fails only when neither root nor the play client's PlayerGui has a ScreenGui: a game whose UI is all built at run time leaves StarterGui empty, which is normal - editTree says so and the client's PlayerGui is audited. The play half needs MifBridge plugin 0.17.8 or later (its play server waits for the client, RB1850), read from the connected plugin: an older one - a running Studio keeps the plugin it loaded (RB1970) - is named with the restart and nothing is run. The client probe says it runs as it starts, waits play_wait and then up to 4 s more for a ScreenGui that is not Roblox's own (a client blocked at its start runs play_wait out before the game builds its UI, RB2074), then until that UI stops being built - its descendant count the same for 2 s, for at most 12 s - and answers uiSettled (a HUD a client script builds behind WaitForChild calls was read half-built on a slow start; a UI still changing at the deadline is answered uiSettled false), and is given that plus time to read and print max_elements; when no answer comes back, play.why and the error say which: no-client (the probe's clientsWarning), report-lost (the client ran but its report, too large, never arrived: lower max_elements), script-error (with the client's own error message), out-of-time, output-cut, never-ran or probe-failed. A label's text is kept to its first 40 characters, never cut inside one (a cut emoji stopped the JSON, RB2070). Rims and badges (both halves): a ScrollingFrame whose UIPadding on a scrolling edge (ScrollingDirection X: Left and Right, Y: Top and Bottom) is less than the widest rim a UIStroke draws outside one of its children (BorderStrokePosition Outer draws Thickness, Center half, Inner none; a Contextual stroke on text draws none) is a scrollPadding row per edge and fails - the first and last cards' rims are cut there; an element that pokes outside its parent (one that does not clip it) over a sibling of that parent - a tab's badge over the next tab - is a badgesOverSiblings row, failed in the edit tree and reported only in the play client (its minimized window's layout). hold_hook (with play; a dot path from game to a BindableFunction the game's LocalScript answers, e.g. ReplicatedStorage.MifAuditHook) checks that a held input ends when a screen opens over it: the hook answers 'repeating' with a list of the inputs the game repeats while held - 'key:<KeyCode>', 'button:<path under PlayerGui>' or the GuiButton itself, 'mouse' (MouseButton1 at the view's center) - 'ticks' with a number that rises each time a repeating action runs, and 'open_screen' and 'close_screen' by opening and closing one of its screens. Each input (at most 8) is held through UserInputService:CreateVirtualInput (SendKey, or SendMouseButton at the button's center plus the GUI inset, RB2740) for 0.6 s - its ticks must rise - then a screen is opened over it and it is released 0.3 s later; a tick in the 1 s from 0.2 s after the release is a hold still running (holds.stillRunning: a screen took the release, RB2741), and the probe then presses and releases it again with no screen open, as a player would (stoppedByRepress). A hold that never started, a missing hook or one that marks nothing fails too. press_buttons (with play) checks that a press answers where it was pressed: after a 0.6 s look at what changes with no press at all (a ticking label, left out), each visible GuiButton inside a panel - a frame under its ScreenGui; a button that is the ScreenGui's own child has no panel and is not pressed - is clicked through CreateVirtualInput's SendMouseButton at its center plus the GUI inset (at most 24, in PlayerGui order; one hidden or covered by then is skipped and named), and the game's ScreenGuis are compared 0.6 s later (Visible, Position, Size, colors and transparency, Text, Image, ScreenGui Enabled, UIStroke): a button whose press changed nothing inside its own panel - the ScreenGui's child that holds it - but something in another ScreenGui or another of its ScreenGui's top frames (a toast) is presses.answeredElsewhere and fails; one that changed nothing visible anywhere is presses.noVisibleAnswer, reported only (a server's answer can take longer). The presses are real clicks in the play test: the game's own handlers run. A run that pressed nothing fails. ok is false when a button is unreachable or not Selectable, a link is dead, an action or prompt has no gamepad input, a scroll edge cuts a rim, a hold runs on, a press answers only in another layer, or nothing was audited. Read only otherwise."
    if not isinstance(play, bool) or not isinstance(screenshot, bool) or not isinstance(press_buttons, bool):
        return _no("play, screenshot and press_buttons are true or false")
    if press_buttons and not play:
        return _no("press_buttons presses the play client's buttons: give play true too", refusedRule="type")
    if console is not None and console not in CONSOLES:
        return _no("console is one of %s (the device simulator's console devices), or omitted" % ", ".join(CONSOLES))
    for nm, v in (("min_button_px", min_button_px), ("min_text_px", min_text_px)):
        if v is not None and (not isinstance(v, (int, float)) or isinstance(v, bool) or not 0 < v <= 1000):
            return _no("%s is a number of pixels, 0-1000, or omitted" % nm)
    if not isinstance(max_elements, int) or isinstance(max_elements, bool) or not 1 <= max_elements <= 5000:
        return _no("max_elements is 1-5000")
    if not isinstance(play_wait, (int, float)) or isinstance(play_wait, bool) or not 0 <= play_wait <= 60:
        return _no("play_wait is 0-60 seconds")
    if not V.split_path(root):
        return _no("root is a dot path from game, e.g. StarterGui", refusedRule="type")
    if hold_hook is not None:
        hp = V.split_path(hold_hook) if isinstance(hold_hook, str) else None
        if not hp or not play or any(not re.match(r"^[A-Za-z0-9_ ]+$", n) for n in hp):
            return _no("hold_hook is a dot path from game to the game's BindableFunction (e.g. "
                       "ReplicatedStorage.MifAuditHook), with play true", refusedRule="type")
        hold_hook = ".".join(hp)
    T = _T()
    out = {"root": V.join_path(V.split_path(root)), "guideline": GUIDELINE}
    sim = None
    before = None
    try:
        if console:
            # the person's own device and resolution, read first, so the audit puts them back (found 2026-11-22: it
            # stopped the simulator, so a person simulating a phone was left at the default device)
            before = _device_before(T)
            s = T.rbx_device_sim("set", device=console)
            if not s.get("ok"):
                return _no("the device simulator did not take %s: %s" % (console, s.get("error")))
            r = T.rbx_device_sim("resolution", width=1920, height=1080)
            sim = {"device": console, "set": True, "resolution": [1920, 1080] if r.get("ok") else r.get("error")}
            time.sleep(1.0)
            ten = T.rbx_call("GuiService", "IsTenFootInterface")
            gs, _e = _get("GuiService", ["ViewportDisplaySize"])
            ui, _e = _get("UserInputService", ["GamepadEnabled", "PreferredInput"])
            sim.update(tenFoot=ten.get("value"), viewportDisplaySize=(gs or {}).get("ViewportDisplaySize"),
                       gamepadEnabled=(ui or {}).get("GamepadEnabled"), preferredInput=(ui or {}).get("PreferredInput"))
        sg_list, elements, err = read_ui(root, max_elements)
        if err:
            return dict(out, ok=False, error=err, console=sim)
        sel, _e = _get("GuiService", ["SelectedObject"])
        pads, strokes = rims_from_edit(sg_list, elements)
        rep, problems, shown = judge_ui(sg_list, elements, _path((sel or {}).get("SelectedObject")),
                                        min_button_px, min_text_px, max_elements, pads=pads, strokes=strokes)
        out.update(rep, console=sim)
        if screenshot:
            shot = T.rbx_screenshot(max_width=1920, name="controller_audit")
            out["screenshot"] = {k: shot.get(k) for k in ("ok", "file", "width", "height", "error")}
        client_guis = []
        if play:
            # with console the test is SOLO: its client runs as the simulated console (ten-foot, 1919 x 1079, RB1881); a
            # one-player multiplayer test's client is a desktop window the simulator never reaches (RB1444)
            # the play half needs a plugin whose play server waits for its clients (0.17.8, RB1850): an older one, read
            # from rbx_status (a running Studio keeps the plugin it loaded, RB1970), is named with the restart, not run
            have, older = T._plugin_older(T.CLIENT_WAIT_PLUGIN)
            if older:
                res, wrote, diag = None, None, {"why": "plugin-old"}
                perr = ("this Studio runs MifBridge plugin %s, and the play half needs %s or later (its play server waits "
                        "for the client before its window, and the client keeps the lines it printed late): %s. Nothing "
                        "was run" % (have, ".".join(map(str, T.CLIENT_WAIT_PLUGIN)), T.PLUGIN_RESTART_STEP))
            else:
                diag = {}
                res, perr, wrote = play_half(wait=play_wait, cap=max_elements, players=None if console else 1, diag=diag,
                                             hook=hold_hook, presses=press_buttons)
            out["play"] = {"wrote": wrote, "solo": bool(console), "plugin": have}
            if perr:
                out["play"].update(error=perr, why=diag.get("why"),
                                   **{k: v for k, v in diag.items() if k != "why" and v is not None})
                if older:
                    out["play"]["steps"] = [T.PLUGIN_RESTART_STEP]
                problems.append("play half: " + perr)
            else:
                no_pad, prompt_no_pad, builtin = judge_actions(res)
                out["play"].update(actions=[a for a in res.get("actions") or [] if not is_builtin_action(a.get("name"))],
                                   robloxActions=builtin, prompts=res.get("prompts"), actionsError=res.get("actionsError"),
                                   actionsWithoutGamepad=no_pad, promptsWithoutGamepad=prompt_no_pad,
                                   clientTenFoot=res.get("tenFoot"), clientDisplaySize=res.get("displaySize"),
                                   clientGamepadEnabled=res.get("gamepadEnabled"), clientViewport=res.get("viewport"),
                                   clientPreferredInput=res.get("preferredInput"), uiWaited=res.get("uiWaited"),
                                   uiSettled=res.get("uiSettled"), uiSettleSeconds=res.get("uiSettleSeconds"))
                if console and res.get("tenFoot") is not True:
                    problems.append("the solo play client under %s read IsTenFootInterface %s: its UI was not seen as a "
                                    "console's" % (console, res.get("tenFoot")))
                if res.get("actionsError"):
                    problems.append("GetAllBoundActionInfo failed: %s" % res["actionsError"])
                if no_pad:
                    problems.append("%d bound action(s) take no gamepad input: %s" % (len(no_pad), ", ".join(no_pad[:10])))
                if prompt_no_pad:
                    problems.append("%d prompt(s) have no GamepadKeyCode" % len(prompt_no_pad))
                # what the client shows: its PlayerGui, judged like the edit tree
                all_guis = res.get("screenGuis") or []
                client_guis = [g for g in all_guis if g.get("name") not in ROBLOX_GUIS]
                keep = {g["path"] for g in client_guis}
                for e in res.get("elements") or []:
                    i = e.get("sgi") if isinstance(e, dict) else None
                    if "screenGui" not in e and isinstance(i, int) and 1 <= i <= len(all_guis):
                        e["screenGui"] = all_guis[i - 1].get("path")
                cel = [e for e in res.get("elements") or [] if e.get("screenGui") in keep]
                effective_visibility(client_guis, cel)
                cpads = [p for p in res.get("pads") or [] if isinstance(p, dict)]
                cstrokes = [s for s in res.get("strokes") or [] if isinstance(s, dict)]
                crep, cprob, _cshown = judge_ui(client_guis, cel, res.get("selected"), min_button_px, min_text_px,
                                                max_elements, screen_fails=False, pads=cpads, strokes=cstrokes)
                crep["robloxGuis"] = sorted(g.get("name") for g in all_guis if g.get("name") in ROBLOX_GUIS)
                crep["truncated"] = bool(res.get("truncated"))
                if console:
                    crep["screenNote"] = ("positions are the solo client's, at the simulated console's view (%s): off-"
                                          "screen and edge buttons are reported, not failed" % (res.get("viewport"),))
                out["play"]["playerGui"] = crep
                problems += ["play client: " + x for x in cprob]
                if hold_hook:
                    hrep, hprob = judge_holds(res.get("holds"))
                    out["play"]["holds"] = dict(hrep, hook=hold_hook)
                    problems += hprob
                if press_buttons:
                    prep, pprob = judge_presses(res.get("presses"))
                    out["play"]["presses"] = prep
                    problems += pprob
            if wrote and not wrote.get("removed"):
                problems.append("the probe LocalScript %s was not removed" % wrote["path"])
    finally:
        if sim is not None:
            sim.update(_device_restore(T, before))
    if not sg_list and play:
        # a UI built at run time by LocalScripts leaves the edit tree empty: normal, said, and failed only when the play
        # client shows nothing either (a failed play half is its own problem, never a second one here)
        out["editTree"] = "no ScreenGui under %s: %s" % (out["root"], (
            "the UI was audited in the play client's PlayerGui" if client_guis else
            "and the play client was not read, so no UI was audited" if "error" in out.get("play", {}) else
            "and the play client's PlayerGui had none either"))
    if not sg_list and not client_guis and not (play and "error" in out.get("play", {})):
        late = ""
        if play and isinstance((out.get("play") or {}).get("uiWaited"), (int, float)):
            late = (" (the client was read %g s after play_wait and %g s more waiting for one: a UI the game builds later "
                    "than that is not seen - raise play_wait)" % (play_wait, round(out["play"]["uiWaited"], 1)))
        problems.append("no ScreenGui under %s%s: nothing was audited%s" % (
            out["root"], " nor in the play client's PlayerGui" if play else "", late))
    elif not shown and not play:
        problems.append("no visible GuiButton or TextBox under %s: nothing was audited" % out["root"])
    out["ok"] = not problems
    if problems:
        out["problems"] = problems
        out["error"] = "; ".join(problems)
    modeled = (out.get("selection") or {}).get("modeled") or ((out.get("play") or {}).get("playerGui") or {}).get(
        "selection", {}).get("modeled")
    out["navigation"] = ("explicit links read from Studio; neighbours without a link are modeled - Studio's automatic "
                         "navigation cannot be driven from a plugin (RB1441)") if modeled else "explicit links only"
    return out
