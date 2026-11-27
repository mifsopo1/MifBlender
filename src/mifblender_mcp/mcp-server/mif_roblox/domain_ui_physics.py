"""Roblox lane 4: UI (batch 8), physics, constraints and navigation (batch 9), camera and input (batch 12).

the internal notes sections 5.5, 5.6 and 5.12 are the plan; the internal notes "UI, physics, camera and input" is what was built. Every tool here
is a composite over the generic layer in tools.py (rbx_create, rbx_set_props, rbx_get_props, rbx_call, rbx_batch, rbx_watch,
rbx_undo_group, rbx_play_probe): nothing here talks to the plugin with an op of its own, so the plugin's safety gate and
its one-recording rule hold unchanged. What a composite adds is the check before anything is sent (every class and member
against the catalog) and the read-back after: ok means Studio was asked again and holds what was meant.

tools.py imports this module's rbx_* names into its own namespace (one line, before TOOLS), so they join TOOLS and PACK the
way every other tool does. This module reaches tools.py lazily (_T()), never at import, so either may load first.

NEVER RAISE: every failure is {"ok": False, ...}.
"""
import functools
import math
import re
import time
import uuid
from typing import Any

from mif_roblox import catalog as C
from mif_roblox import values as V

__all__ = []      # filled by _tool below: only the rbx_* tools leave this module


def _T():
    from mif_roblox import tools as T
    return T


def _tool(fn):
    """Export fn and post its verdict to the panel's journal, as tools._reports does."""
    @functools.wraps(fn)
    def wrapper(*a, **kw):
        try:
            out = fn(*a, **kw)
        except _Refused as exc:
            out = dict(exc.answer)       # a read a step depends on failed: its own answer, any open recording cancelled
        except Exception as exc:  # noqa: BLE001 - never raise: a bug answers as a failure with its text
            out = {"ok": False, "error": "%s failed inside the bridge: %s: %s" % (fn.__name__, type(exc).__name__, exc)}
        _T()._post_verdict(fn.__name__, out)
        return out
    __all__.append(fn.__name__)
    return wrapper


# ---------------------------------------------------------------- small helpers
def _no(error, **kw):
    return dict({"ok": False, "ran": False, "error": error}, **kw)


def _parts(x, what="target"):
    parts = V.split_path(x)
    if not parts:
        return None, _no("%s is a dot path from game ('Workspace.Part') or a list of names, got %r" % (what, x),
                         refusedRule="type")
    return parts, None


def _join(parts):
    return V.join_path(parts)


def _show(parts):
    """A path as text, always a string: join_path answers the list itself when a name holds a dot ("Cube.001"), which a
    message built with ", ".join cannot take."""
    if isinstance(parts, str):
        return parts
    if isinstance(parts, (list, tuple)):
        return ".".join(str(p) for p in parts)
    return str(parts)


class _Refused(Exception):
    """Raised by a helper whose read a write depends on, carrying the {"ok": False, ...} answer; _tool returns it, and a
    `with _Group` it passes through cancels the recording on the way out."""

    def __init__(self, answer):
        Exception.__init__(self, answer.get("error"))
        self.answer = answer


def _num(x):
    return V._num(x)


def _vec(x, n=3):
    return isinstance(x, (list, tuple)) and len(x) == n and all(_num(v) for v in x)


def _xyz(v):
    """A decoded Vector3 ({x,y,z} or [x,y,z]) as a list, or None."""
    if isinstance(v, dict) and all(k in v for k in "xyz"):
        return [v["x"], v["y"], v["z"]]
    if _vec(v):
        return list(v)
    return None


def _xy(v):
    if isinstance(v, dict) and "x" in v and "y" in v:
        return [v["x"], v["y"]]
    if _vec(v, 2):
        return list(v)
    return None


def _close(a, b, rel=1e-4, abs_tol=1e-6):
    return _num(a) and _num(b) and abs(a - b) <= max(abs_tol, rel * max(abs(a), abs(b)))


def _same_decoded(a, b):
    """Two decoded values hold the same thing: numbers to 1e-4 relative, a color's channels to 1/255, the rest exactly.
    Studio keeps floats and adds nothing a caller sent, so a key the caller did not send is ignored on the read side."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if _num(a) and _num(b):
        return _close(a, b)
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) == {"r", "g", "b"} and set(b) >= {"r", "g", "b"}:
            return all(_num(b[k]) and abs(a[k] - b[k]) <= 1.0 / 255 + 1e-9 for k in "rgb")
        return all(k in b and _same_decoded(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same_decoded(x, y) for x, y in zip(a, b))
    return a == b


def _sub(a, b):
    return [x - y for x, y in zip(a, b)]


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _norm(a):
    n = math.sqrt(_dot(a, a))
    return [x / n for x in a] if n > 1e-9 else None


def _basis_x(axis):
    """A CFrame rotation (9 numbers, row-major) whose X column (RightVector) is axis: a hinge or prismatic constraint acts
    along its attachments' X axis (Attachment.Axis is the X column)."""
    x = _norm(axis)
    if x is None:
        return None
    up = [0, 1, 0] if abs(x[1]) < 0.99 else [0, 0, 1]
    z = _norm(_cross(x, up))
    y = _cross(z, x)
    return [x[0], y[0], z[0], x[1], y[1], z[1], x[2], y[2], z[2]]


def _look_at(pos, target):
    """CFrame.lookAt(pos, target) as {position, rotation}: Studio's camera looks down its -Z column."""
    f = _norm(_sub(target, pos))
    if f is None:
        return None
    up = [0, 1, 0] if abs(f[1]) < 0.999 else [0, 0, 1]
    r = _norm(_cross(f, up))
    u = _cross(r, f)
    return {"position": list(pos), "rotation": [r[0], u[0], -f[0], r[1], u[1], -f[1], r[2], u[2], -f[2]]}


def _cf_point(cf, local):
    """World point of a local offset under a decoded CFrame {position, rotation}."""
    p, R = cf["position"], cf["rotation"]
    return [p[i] + R[3 * i] * local[0] + R[3 * i + 1] * local[1] + R[3 * i + 2] * local[2] for i in range(3)]


def _cat():
    return _T()._catalog()


def _rbx(op, **kw):
    return _T()._rbx(op, **kw)


def _exists(parts):
    return bool(_rbx("resolve", target=parts).get("ok"))


def _class_of(parts):
    r = _rbx("resolve", target=parts)
    return (r.get("className"), None) if r.get("ok") else (None, r)


def _read(target, props):
    """rbx_get_props' values, or (None, the failure)."""
    r = _T().rbx_get_props(target, props)
    return (r.get("values") or {}, None) if r.get("ok") else (None, r)


def _call(target, method, args=None, timeout=30):
    return _T().rbx_call(target, method, args or [], timeout=timeout)


def _value(r):
    return r.get("value") if r.get("ok") else None


class _Group:
    """One undo recording for a composite's several writes (rbx_undo_group): commit on success, cancel on a failure, so a
    half-made composite is never left in the place. The answer says which, and whether the cancel was asked of Studio."""

    def __init__(self, name):
        self.name, self.open, self.began = name, False, None

    def __enter__(self):
        self.began = _T().begin_group(self.name)
        self.open = bool(self.began.get("ok"))
        return self

    def end(self, ok):
        if not self.open:
            return None
        self.open = False
        return _T().rbx_undo_group("commit" if ok else "cancel")

    def cancel(self):
        """Cancel the recording; True only when Studio answered the cancel ok (the writes were reverted)."""
        return bool((self.end(False) or {}).get("ok"))

    def refused(self):
        """The answer when begin was refused: nothing is written, since a write outside a recording of its own would
        join another one, or stay half-made with no cancel to take it back."""
        b = self.began if isinstance(self.began, dict) else {}
        return dict(b, ok=False, verified=False, ran=False, cancelled=False,
                    error="no undo group could be opened, so nothing was written: %s" % (b.get("error") or b))

    def __exit__(self, et, ev, tb):
        if self.open:
            cancelled = self.cancel()
            if isinstance(ev, _Refused):
                ev.answer = dict(ev.answer, cancelled=cancelled)
        return False


def _ensure_scratch():
    """Workspace.MifScratch, made when missing (allowed in scratch mode too). (made?, failure)."""
    if _exists(["Workspace", "MifScratch"]):
        return False, None
    mk = _rbx("create", parent=["Workspace"], className="Folder", name="MifScratch", props={}, tags=[])
    return (True, None) if mk.get("ok") else (False, mk)


CHILDREN_LIMIT = 10000   # rbx_tree's most; the tree op has no page, so a parent with more children is refused, not guessed


def _children(parts):
    """Every child name of parts. Raises _Refused when the read failed or did not reach every child: a failed or cut read
    taken as the whole set hands back names already in use."""
    t = _rbx("tree", root=parts, depth=1, limit=CHILDREN_LIMIT)
    if not t.get("ok"):
        raise _Refused(dict(_no("could not read the children of %s to pick a free name, so nothing was made: %s" % (
            _show(parts), t.get("error")), refusedRule=t.get("refusedRule")), read=t))
    root = t.get("root") if isinstance(t.get("root"), dict) else {}
    kids = root.get("children") if isinstance(root.get("children"), list) else []
    count = root.get("childCount")
    if root.get("truncated") or t.get("truncated") or (isinstance(count, int) and count > len(kids)):
        raise _Refused(_no("%s has %s children and one read reaches %d, so a free name cannot be known; nothing was made" % (
            _show(parts), count if isinstance(count, int) else "more", len(kids)), refusedRule="too-many-children"))
    return [c.get("name") for c in kids if isinstance(c, dict)]


def _free_name(parent_parts, base, taken=()):
    have = set(_children(parent_parts)) | set(taken)
    name, n = base, 1
    while name in have:
        n += 1
        name = "%s_%d" % (base, n)
    return name


# ---------------------------------------------------------------- trees: one description, one recording
UI_BASES = ("GuiBase", "UIComponent", "StyleBase", "StyleLink", "StyleDerive", "StyleQuery")
NODE_KEYS = {"class", "name", "props", "tags", "children"}
MAX_TREE = 100          # rbx_batch's limit: a tree is one recording or nothing


def _plan_tree(cat, roots, parent_parts, allowed_bases=None, what="element"):
    """Flatten [{class, name?, props?, tags?, children?}] into rbx_batch create ops, parents before children. Every class and
    property is checked here, before anything is sent: an unknown class or member, a NotCreatable class, a class outside
    allowed_bases, a value of the wrong shape, and two siblings with one name (a path must name one instance) are all
    refused with every problem listed. (ops, paths, refusal)."""
    if isinstance(roots, dict):
        roots = [roots]
    if not isinstance(roots, list) or not roots:
        return None, None, _no("the description is an element {class, name, props, children} or a list of them")
    ops, paths, problems = [], [], []

    def walk(node, pparts, where):
        if len(ops) > MAX_TREE:
            return
        if not isinstance(node, dict) or not isinstance(node.get("class"), str):
            problems.append({"at": where, "error": "an %s is {class, name?, props?, tags?, children?}" % what})
            return
        extra = sorted(set(node) - NODE_KEYS)
        if extra:
            problems.append({"at": where, "error": "unknown keys %s; an %s has %s" % (extra, what, sorted(NODE_KEYS))})
        cls = node["class"]
        bad = cat.create_refusal(cls)
        if bad:
            problems.append({"at": where, "refusedRule": bad.get("refusedRule"), "error": bad.get("error"),
                             "didYouMean": bad.get("didYouMean")})
            return
        if allowed_bases and not any(cat.is_a(cls, b) for b in allowed_bases):
            problems.append({"at": where, "refusedRule": "wrong-class",
                             "error": "%s is not one of %s" % (cls, ", ".join(allowed_bases))})
            return
        name = node.get("name", cls)
        if not isinstance(name, str) or not name:
            problems.append({"at": where, "error": "name is a non-empty string"})
            return
        wire = {}
        if node.get("props") is not None:
            wire, err = _T()._encode_props(cat, cls, node["props"])
            if err:
                problems.append({"at": where, "refused": err.get("refused"), "error": err.get("error")})
                return
        tags = node.get("tags") or []
        if not (isinstance(tags, list) and all(isinstance(t, str) and t for t in tags)):
            problems.append({"at": where, "error": "tags is a list of non-empty strings"})
            return
        me = pparts + [name]
        ops.append({"tool": "rbx_create", "args": {"class_name": cls, "parent": list(pparts), "name": name,
                                                   "props": node.get("props") or None, "tags": tags}})
        paths.append((me, cls, wire))
        kids = node.get("children") or []
        if not isinstance(kids, list):
            problems.append({"at": where, "error": "children is a list"})
            return
        seen = {}
        for i, k in enumerate(kids):
            kn = k.get("name", k.get("class")) if isinstance(k, dict) else None
            if kn in seen:
                problems.append({"at": "%s.children[%d]" % (where, i), "refusedRule": "duplicate-name",
                                 "error": "two children of %s are named %s: a path must name one instance" % (name, kn)})
            seen[kn] = i
            walk(k, me, "%s.children[%d]" % (where, i))

    names = {}
    for i, r in enumerate(roots):
        rn = r.get("name", r.get("class")) if isinstance(r, dict) else None
        if rn in names:
            problems.append({"at": "[%d]" % i, "refusedRule": "duplicate-name", "error": "two roots are named %s" % rn})
        names[rn] = i
        walk(r, list(parent_parts), "[%d]" % i)
    if len(ops) > MAX_TREE:
        return None, None, _no("the description has more than %d elements; one recording takes at most %d" % (MAX_TREE, MAX_TREE),
                               refusedRule="too-many")
    if problems:
        return None, None, _no("%d problem(s) in the description, nothing was made: %s" % (
            len(problems), "; ".join("%s: %s" % (p["at"], p["error"]) for p in problems[:8])), problems=problems)
    return ops, paths, None


def _build_tree(roots, parent, allowed_bases, what, replace=False):
    cat, err = _cat()
    if err:
        return err
    pparts, err = _parts(parent, "parent")
    if err:
        return err
    ops, paths, err = _plan_tree(cat, roots, pparts, allowed_bases, what)
    if err:
        return err
    if not _exists(pparts):
        return _no("the parent %s does not exist" % _join(pparts), refusedRule="not-found")
    taken = set(_children(pparts))
    clash = [_join(p) for p, _c, _w in paths if len(p) == len(pparts) + 1 and p[-1] in taken]
    if clash and not replace:
        return _no("already there: %s. Pass replace true to destroy and rebuild them in the same recording" % ", ".join(
            _show(c) for c in clash), refusedRule="exists", existing=clash)
    pre = [{"tool": "rbx_destroy", "args": {"target": c}} for c in clash] if replace else []
    if len(pre) + len(ops) > MAX_TREE:
        return _no("replace needs %d destroy(s) and %d create(s), %d ops, and one recording (rbx_batch) takes at most %d: "
                   "nothing was changed. Destroy the old elements first (rbx_destroy), or build fewer at once" % (
                       len(pre), len(ops), len(pre) + len(ops), MAX_TREE),
                   refusedRule="too-many", existing=clash)
    r = _T().rbx_batch(pre + ops)
    out = {"ok": bool(r.get("ok")), "verified": bool(r.get("verified")), "elements": len(paths),
           "paths": [_join(p) for p, _c, _w in paths], "requestId": r.get("requestId")}
    if replace and clash:
        out["replaced"] = clash
    if not r.get("ok"):
        return dict(out, ok=False, verified=False, error=r.get("error"), batch=r)
    # The read-back beyond each create's own: the whole tree again, by name and class, in one call.
    depth = max(len(p) for p, _c, _w in paths) - len(pparts)
    tree = _rbx("tree", root=pparts, depth=min(depth, 20), limit=10000)
    found = {}

    def walk(node, at):
        for c in node.get("children") or []:
            found[tuple(at + [c.get("name")])] = c.get("className")
            walk(c, at + [c.get("name")])
    walk(tree.get("root") or {}, list(pparts))
    missing = [_join(p) for p, c, _w in paths if found.get(tuple(p)) != c]
    out["foundByPath"] = len(paths) - len(missing)
    if missing:
        out.update(ok=False, verified=False, missing=missing,
                   error="%d element(s) not found by path and class after the build: %s" % (
                       len(missing), ", ".join(_show(m) for m in missing[:10])))
    return out


# ================================================================ batch 8: UI
# ScreenInsets and IgnoreGuiInset are one setting seen two ways (RB1416): DeviceSafeInsets reads IgnoreGuiInset true,
# CoreUISafeInsets reads it false. A description that sets both another way gets the other one silently overwritten.
INSET_PAIRS = {"DeviceSafeInsets": True, "CoreUISafeInsets": False}


def _udim2(v):
    v = v.get("v") if isinstance(v, dict) else v
    return v if isinstance(v, list) and len(v) == 4 and all(isinstance(x, (int, float)) for x in v) else None


def _ui_spec_checks(spec):
    """Before anything is made: a ScreenGui whose IgnoreGuiInset and ScreenInsets disagree (refusals), and each element
    sized or placed in Scale under a UIScale, which multiplies Offset only (RB1415: warnings). (refusals, warnings)."""
    refusals, warnings = [], []

    def walk(node, where, scaled_by):
        if not isinstance(node, dict):
            return
        props = node.get("props") if isinstance(node.get("props"), dict) else {}
        kids = [k for k in node.get("children") or [] if isinstance(k, dict)] if isinstance(node.get("children"), list) else []
        name = node.get("name", node.get("class"))
        if node.get("class") == "ScreenGui" and "IgnoreGuiInset" in props and "ScreenInsets" in props:
            si, ig = props["ScreenInsets"], props["IgnoreGuiInset"]
            si = si.get("v") if isinstance(si, dict) else si
            want = INSET_PAIRS.get(si)
            if want is None or want != ig:
                use = ("ScreenInsets %s alone (it reads IgnoreGuiInset %s)" % (si, str(want).lower()) if want is not None
                       else "ScreenInsets alone; how %s reads IgnoreGuiInset is not measured" % si)
                refusals.append({"at": where, "name": name, "refusedRule": "inset-contradiction",
                                 "sent": {"IgnoreGuiInset": ig, "ScreenInsets": si},
                                 "error": "ScreenInsets %s and IgnoreGuiInset %s are one setting seen two ways (RB1416): "
                                          "Studio keeps one and overwrites the other. Set %s" % (si, str(ig).lower(), use)})
        if scaled_by:
            for prop in ("Size", "Position"):
                u = _udim2(props.get(prop))
                if u and (u[0] or u[2]):
                    warnings.append({"at": where, "name": name, "property": prop, "value": u, "uiScale": scaled_by,
                                     "warning": "%s uses Scale (%g, %g) under the UIScale at %s, which multiplies Offset "
                                                "only (RB1415): the Scale part still follows the real screen" % (
                                                    prop, u[0], u[2], scaled_by)})
        mine = next((k.get("name", "UIScale") for k in kids if k.get("class") == "UIScale"), None)
        inner = scaled_by or ("%s.%s" % (where, mine) if mine else None)
        for i, k in enumerate(kids):
            if k.get("class") != "UIScale":
                walk(k, "%s.children[%d]" % (where, i), inner)
    roots = spec if isinstance(spec, list) else [spec]
    for i, r in enumerate(roots):
        walk(r, "[%d]" % i, None)
    return refusals, warnings


@_tool
def rbx_ui_build(spec: Any, parent: Any = "StarterGui", replace: bool = False) -> dict:
    "Roblox Studio: a whole GUI tree from one description, made in one undo recording. spec is an element {class, name?, props?, tags?, children?} or a list of them, under parent (default StarterGui; a ScreenGui is the usual root). Every class must be a GUI class (GuiBase, UIComponent, StyleBase, StyleLink, StyleDerive, StyleQuery) and every property is typed by the catalog: an unknown class or property, a wrong-shaped value or two siblings with one name refuse the whole tree before anything is made. Read back: each create's properties, then the tree read again so every element is found by path and class. At most 100 elements. A root already there is refused unless replace is true. Also refused before anything is made: a ScreenGui given both IgnoreGuiInset and ScreenInsets that disagree (they are one setting, RB1416: DeviceSafeInsets reads IgnoreGuiInset true, CoreUISafeInsets false; the refusal names the one to set). Each element sized or placed in Scale under a UIScale in the description is answered in scaleUnderUIScale: a UIScale multiplies Offset only (RB1415)."
    if not isinstance(replace, bool):
        return _no("replace is true or false")
    refusals, warnings = _ui_spec_checks(spec)
    if refusals:
        return _no("; ".join("%s: %s" % (r["at"], r["error"]) for r in refusals), refusedRule="inset-contradiction",
                   problems=refusals, scaleUnderUIScale=warnings)
    out = _build_tree(spec, parent, UI_BASES, "element", replace)
    if warnings and isinstance(out, dict):
        out["scaleUnderUIScale"] = warnings
    return out


_RECT_PROPS = ("AbsolutePosition", "AbsoluteSize")


def _rect(vals):
    p, s = _xy(vals.get("AbsolutePosition")), _xy(vals.get("AbsoluteSize"))
    return [p[0], p[1], s[0], s[1]] if p and s else None


def _overlap(a, b):
    w = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    h = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    return w * h if w > 0.5 and h > 0.5 else 0


def _inside(child, parent, slack=0.5):
    return (child[0] >= parent[0] - slack and child[1] >= parent[1] - slack
            and child[0] + child[2] <= parent[0] + parent[2] + slack and child[1] + child[3] <= parent[1] + parent[3] + slack)


def _query_all(root_parts, class_name, cap):
    out, page = [], 0
    while True:
        q = _T().rbx_query(root=root_parts, class_name=class_name, page=page, page_size=500)
        if not q.get("ok"):
            return None, q
        out += [m for m in q.get("matches") or []]
        if not q.get("more") or len(out) >= cap:
            return out[:cap], None
        page += 1


def _findings(elements, by_path):
    """Findings of laid-out elements: overflow out of the parent, overlap between visible siblings, text that does not
    fit. -> (overflow, overlaps, textOverflow, siblings grouped by parent path)."""
    overflow, overlaps, text = [], [], []
    groups = {}
    for e in elements:
        ep = tuple(V.split_path(e["path"]))
        par = by_path.get(ep[:-1])
        if e["rect"] and par and par.get("rect") and par["rect"][2] > 0 and not _inside(e["rect"], par["rect"]):
            overflow.append({"path": e["path"], "rect": e["rect"], "parent": par["path"], "parentRect": par["rect"]})
        if e.get("textFits") is False:
            text.append({"path": e["path"], "textBounds": e.get("textBounds"), "rect": e["rect"]})
        if e["rect"] and e.get("visible", True) is not False:
            groups.setdefault(ep[:-1], []).append(e)
    for sib in groups.values():
        for i in range(len(sib)):
            for j in range(i + 1, len(sib)):
                a = _overlap(sib[i]["rect"], sib[j]["rect"])
                if a:
                    overlaps.append({"a": sib[i]["path"], "b": sib[j]["path"], "area": round(a, 1)})
    return overflow, overlaps, text, groups


# The play route (found 2026-11-26: a UI review had to print every shown GuiObject's rect from the client by hand, and a
# dump taken as the client started recorded a 1 x 1 viewport: the test window is sized a moment after its client starts,
# the internal notes L159). A LocalScript in StarterPlayerScripts (domain_controller.play_client, removed after) waits until the
# camera's ViewportSize.Y is PLAY_MIN_VIEWPORT or more and the root is there, settles, and prints every GuiBase2d under
# it with its rect, Visible and TextFits. Roblox's own guis directly under PlayerGui are left out.
PLAY_MIN_VIEWPORT = 300
PLAY_LAYOUT_MARK = "MIFLAYOUT:"
PLAY_LAYOUT_SCRIPT = "MifLayoutProbe"
PLAY_LAYOUT_SOURCE = r'''-- MifBridge play layout probe: written by rbx_ui_layout(play=true) for one play test, removed after. Read only.
local Players = game:GetService("Players")
local HS = game:GetService("HttpService")
local pg = Players.LocalPlayer:WaitForChild("PlayerGui")
local cam = workspace.CurrentCamera
local t0 = os.clock()
while cam.ViewportSize.Y < %(minvp)s and os.clock() - t0 < %(wait)s do
	task.wait(0.1)
end
local vpWaited = os.clock() - t0
local root = pg
for _, n in ipairs({ %(names)s }) do
	root = root and root:WaitForChild(n, %(wait)s)
end
task.wait(%(settle)s)
local ROBLOX = { %(roblox)s }
local r = { viewport = { cam.ViewportSize.X, cam.ViewportSize.Y }, viewportWaited = vpWaited, elements = {} }
if root == nil then
	r.missing = true
else
	local list = { root }
	for _, d in ipairs(root:GetDescendants()) do
		table.insert(list, d)
	end
	local n = 0
	for _, g in ipairs(list) do
		if g:IsA("GuiBase2d") and not (g.Parent == pg and ROBLOX[g.Name]) then
			n += 1
			if n > %(max)s then
				r.truncated = true
				break
			end
			local e = { path = g:GetFullName(), className = g.ClassName,
				rect = { g.AbsolutePosition.X, g.AbsolutePosition.Y, g.AbsoluteSize.X, g.AbsoluteSize.Y } }
			if g:IsA("GuiObject") then
				e.visible = g.Visible
			end
			pcall(function()
				e.textFits = g.TextFits
				e.textBounds = { g.TextBounds.X, g.TextBounds.Y }
			end)
			table.insert(r.elements, e)
		end
	end
end
'''


def _play_root(parts):
    """The names under PlayerGui a root names: StarterGui.X... (copied into PlayerGui at spawn) or PlayerGui.X...;
    (names, error)."""
    if parts and parts[0] in ("StarterGui", "PlayerGui", "@playergui"):
        return list(parts[1:]), None
    return None, "with play, root is under StarterGui or PlayerGui (StarterGui.Hud reads the play client's PlayerGui.Hud)"


def _play_layout(root, max_elements=500, wait=20, settle=1.0):
    """The play client's PlayerGui laid out: the same rows and findings as the edit route, read in a one-player play test
    once the client's viewport is sized (L159)."""
    from mif_roblox import domain_controller as DC
    parts, err = _parts(root, "root")
    if err:
        return err
    names, err = _play_root(parts)
    if err:
        return _no(err)
    src = PLAY_LAYOUT_SOURCE % {
        "minvp": PLAY_MIN_VIEWPORT, "wait": wait, "settle": settle, "max": int(max_elements),
        "names": ", ".join('"%s"' % n.replace("\\", "").replace('"', "") for n in names),
        "roblox": ", ".join('["%s"] = true' % n for n in sorted(DC.ROBLOX_GUIS))}
    src += DC.CLIENT_EMIT % {"mark": PLAY_LAYOUT_MARK, "chunk": DC.CHUNK}
    diag = {}
    # the client window is kept behind the person's windows, never minimized: a minimized client lays out a small view
    # (RB2562; live 2026-11-26 the viewport went from sized to 296 x 264 once the window was minimized)
    res, perr, wrote = DC.play_client(src, PLAY_LAYOUT_MARK, PLAY_LAYOUT_SCRIPT, wait + settle + 8, players=1, diag=diag,
                                      windows="behind")
    out = {"ok": False, "root": "PlayerGui" + "".join("." + n for n in names), "play": True, "probe": wrote, "diag": diag}
    if perr:
        return dict(out, error="the play client's layout did not come back: %s" % perr)
    vp = res.get("viewport") or [0, 0]
    elements = []
    for e in res.get("elements") or []:
        full = str(e.get("path") or "")
        k = full.find(".PlayerGui")
        e = dict(e, path=full[k + 1:] if k >= 0 else full)
        elements.append(e)
    out.update(viewport=vp, viewportWaited=res.get("viewportWaited"), elements=elements, count=len(elements),
               truncated=bool(res.get("truncated")))
    if res.get("missing"):
        return dict(out, error="%s was not in the play client's PlayerGui within %s s" % (out["root"], wait))
    if not vp or vp[1] < PLAY_MIN_VIEWPORT:
        return dict(out, error="the client's viewport was %s after %s s, under %d high: not laid out yet, so nothing was "
                               "judged" % (vp, wait, PLAY_MIN_VIEWPORT))
    if not elements:
        return dict(out, error="no GUI element under %s in the play client: nothing was laid out" % out["root"])
    by_path = {tuple(V.split_path(e["path"])): e for e in elements}
    overflow, overlaps, text, _groups = _findings(elements, by_path)
    out.update(ok=True, layoutRan=any(e.get("rect") and (e["rect"][2] > 0 or e["rect"][3] > 0) for e in elements),
               overflow=overflow, overlaps=overlaps[:200], textOverflow=text)
    if wrote and not wrote.get("removed"):
        out.update(ok=False, error="the probe LocalScript %s was not removed" % wrote.get("path"))
    return out


def _layout(root, max_elements=500, glyphs=False):
    cat, err = _cat()
    if err:
        return err
    parts, err = _parts(root, "root")
    if err:
        return err
    cls, err = _class_of(parts)
    if err:
        return err
    found, err = _query_all(parts, "GuiBase2d", max_elements)
    if err:
        return err
    items = ([{"path": _join(parts), "className": cls}] if cat.is_a(cls, "GuiBase2d") else []) + found
    if not items:
        return {"ok": False, "elements": 0, "error": "no GUI element under %s: nothing was laid out" % _join(parts)}
    elements, by_path = [], {}
    for m in items:
        mp = V.split_path(m["path"])
        want = list(_RECT_PROPS)
        for p in ("Visible", "TextBounds", "TextFits", "ClipsDescendants", "Text") + (("FontFace",) if glyphs else ()):
            if cat.find_member(m["className"], p)[1] is not None:
                want.append(p)
        g = _rbx("get_props", target=mp, props=want)
        vals = {k: V.decode(v) for k, v in T_obj(g.get("values")).items()} if g.get("ok") else {}
        e = {"path": m["path"], "className": m["className"], "rect": _rect(vals)}
        if "Visible" in vals:
            e["visible"] = vals["Visible"]
        if "TextFits" in vals:
            e["textFits"] = vals["TextFits"]
            e["textBounds"] = _xy(vals.get("TextBounds"))
        if not g.get("ok"):
            e["error"] = g.get("error")
        if glyphs:
            e["_vals"] = vals
        elements.append(e)
        by_path[tuple(mp)] = e
    sized = [e for e in elements if e["rect"] and (e["rect"][2] > 0 or e["rect"][3] > 0)]
    out = {"ok": True, "root": _join(parts), "elements": elements, "count": len(elements),
           "truncated": len(found) >= max_elements, "layoutRan": bool(sized)}
    if not sized:
        out.update(ok=False, error="every element under %s reads AbsoluteSize 0: Studio has not laid this tree out (a "
                                   "StarterGui tree in edit mode may not be; measure with StarterGui.ShowDevelopmentGui on, or "
                                   "build under a ScreenGui that renders)" % _join(parts))
        return out
    overflow, overlaps, text, groups = _findings(elements, by_path)
    out.update(overflow=overflow, overlaps=overlaps[:200], textOverflow=text)
    # Layouts: a UIListLayout's siblings stacked along its axis with the padding it holds.
    lists, err = _query_all(parts, "UIListLayout", 200)
    out["lists"] = []
    for L in lists or []:
        lp = V.split_path(L["path"])
        g = _rbx("get_props", target=lp, props=["FillDirection", "Padding", "AbsoluteContentSize", "Wraps"])
        vals = {k: V.decode(v) for k, v in T_obj(g.get("values")).items()}
        par = by_path.get(tuple(lp[:-1]))
        kids = [e for e in groups.get(tuple(lp[:-1]), []) if e["className"] not in ("UIListLayout",)]
        vertical = vals.get("FillDirection") != "Horizontal"
        ax = 1 if vertical else 0
        pad = vals.get("Padding")
        pad_px = None
        if isinstance(pad, list) and len(pad) == 2 and par and par.get("rect"):
            pad_px = pad[0] * par["rect"][2 + ax] + pad[1]
        kids = sorted((k for k in kids if k["rect"] and cat.is_a(k["className"], "GuiObject")), key=lambda k: k["rect"][ax])
        gaps = [round(b["rect"][ax] - (a["rect"][ax] + a["rect"][2 + ax]), 3) for a, b in zip(kids, kids[1:])]
        row = {"layout": L["path"], "fillDirection": vals.get("FillDirection"), "padding": pad, "paddingPixels": pad_px,
               "absoluteContentSize": vals.get("AbsoluteContentSize"), "children": [k["path"] for k in kids], "gaps": gaps}
        if pad_px is not None and gaps and not vals.get("Wraps"):
            row["stackedWithPadding"] = all(abs(g_ - pad_px) <= 1.0 for g_ in gaps)
        out["lists"].append(row)
    if glyphs:
        audit = _glyph_audit(elements)
        out["glyphs"] = audit
        out["missingGlyphs"] = [{"path": e["path"], "missing": e["missingGlyphs"]} for e in elements if e.get("missingGlyphs")]
        if not audit.get("ok"):
            out.update(ok=False, error="glyphs: %s" % audit.get("error"))
    for e in elements:
        e.pop("_vals", None)
        e.pop("_glyphKeys", None)
    return out


def T_obj(x):
    return x if isinstance(x, dict) else {}


# ---------------------------------------------------------------- glyphs a font cannot draw
# Measured on 0.748 (RB1600): TextService:GetTextBoundsAsync gives a missing character an advance (U+2630 in FredokaOne
# measures 9 x 20 at size 20, in SourceSansPro 11 x 20) and no error, so a width cannot tell. What tells is the drawing:
# in an edit-mode capture U+2630 in FredokaOne draws no ink at all (blank), and in SourceSansPro draws the same box as a
# noncharacter (U+10FFFD) in that font: the missing-glyph box. Emoji, CJK, arrows and shapes draw in every font tried
# (Roblox's fallback fonts), so they are not missing. So a character is missing when its one-character label draws no
# ink, or draws its font's U+10FFFD box.
GLYPH_CELL = 90          # pixels per probe label, at TextSize 60
GLYPH_MAX = 400          # distinct (font, character) pairs per call
TOFU = "\U0010fffd"


def _ink(rgba, w, h, rect):
    """The dark pixels (any channel under 200 on white) inside rect [x, y, w, h], as a set of (dx, dy) from their own
    top-left corner, so two drawings compare by shape wherever they sit."""
    x0, y0, rw, rh = (int(round(v)) for v in rect)
    pts = []
    for y in range(max(0, y0), min(h, y0 + rh)):
        row = y * w
        for x in range(max(0, x0), min(w, x0 + rw)):
            p = (row + x) * 4
            if min(rgba[p], rgba[p + 1], rgba[p + 2]) < 200:
                pts.append((x, y))
    if not pts:
        return set()
    mx, my = min(p[0] for p in pts), min(p[1] for p in pts)
    return {(x - mx, y - my) for x, y in pts}


def _same_drawing(a, b):
    if not a or not b:
        return False
    return len(a & b) / len(a | b) >= 0.85


def _font_key(f):
    f = f if isinstance(f, dict) else {}
    return (f.get("family"), str(f.get("weight") or "Regular"), str(f.get("style") or "Normal"))


def _glyph_audit(elements):
    """For each text element: the characters its FontFace cannot draw, from one probe ScreenGui per page of
    one-character labels captured on white. Adds e["missingGlyphs"]; answers the audit summary."""
    from mif_roblox import domain_showcase as SH
    from mif_roblox import png as P
    T = _T()
    pairs, by_el = [], {}
    for e in elements:
        if "Text" not in (e.get("_vals") or {}):
            continue
        vals = e["_vals"]
        text, font = vals.get("Text"), vals.get("FontFace")
        if not isinstance(text, str) or not text or not isinstance(font, dict):
            continue
        keys = []
        for ch in dict.fromkeys(text):
            if ch.isspace() or ord(ch) < 33:
                continue
            k = (_font_key(font), ch)
            keys.append(k)
            if k not in by_el:
                by_el[k] = []
                pairs.append((k, font))
            by_el[k].append(e)
        e["_glyphKeys"] = keys
    if not pairs:
        return {"ok": False, "error": "no text element with a FontFace and a visible character under the root: no glyph "
                                      "was checked"}
    if len(pairs) > GLYPH_MAX:
        return {"ok": False, "refusedRule": "budget", "error": "%d distinct (font, character) pairs; at most %d a call"
                                                               % (len(pairs), GLYPH_MAX)}
    fonts = {}
    for (fk, _ch), font in pairs:
        fonts.setdefault(fk, font)
    jobs = [(fk, TOFU, font, True) for fk, font in fonts.items()] + [(k[0], k[1], f, False) for k, f in pairs]
    was, err = SH._dev_gui(True)
    if err:
        return dict(err, ok=False, step="ShowDevelopmentGui")
    verdict, ref, shots = {}, {}, []
    name = "MifGlyphProbe_" + uuid.uuid4().hex[:6]
    try:
        with SH._Window() as win:
            per_page = None
            i = 0
            while i < len(jobs):
                # the page: as many labels as the viewport holds, found from the first capture's size
                cols, rows = (20, 6) if per_page is None else per_page
                page = jobs[i:i + cols * rows]
                kids = [{"class": "Frame", "name": "Bg", "props": {"Size": [1, 0, 1, 0], "BackgroundColor3": [1, 1, 1],
                                                                   "BorderSizePixel": 0}}]
                for n, (_fk, ch, font, _isref) in enumerate(page):
                    kids.append({"class": "TextLabel", "name": "G%03d" % n, "props": {
                        "Size": [0, GLYPH_CELL - 10, 0, GLYPH_CELL - 10],
                        "Position": [0, 10 + (n % cols) * GLYPH_CELL, 0, 10 + (n // cols) * GLYPH_CELL],
                        "BackgroundTransparency": 1, "Text": ch, "TextSize": 60, "TextColor3": [0, 0, 0],
                        "FontFace": font, "ZIndex": 2}})
                b = rbx_ui_build({"class": "ScreenGui", "name": name, "props": {"DisplayOrder": 1000000,
                                                                               "IgnoreGuiInset": True}, "children": kids},
                                 parent="StarterGui", replace=True)
                if not b.get("ok"):
                    return dict(b, ok=False, step="glyph probe gui")
                lay = _layout("StarterGui." + name, max_elements=len(kids) + 2)
                rects = {x["path"].split(".")[-1]: x["rect"] for x in lay.get("elements") or []}
                bg = rects.get("Bg")
                if not bg or not bg[2]:
                    return {"ok": False, "error": "the glyph probe was not laid out (%s)" % lay.get("error")}
                if per_page is None:
                    fit = (max(1, int((bg[2] - 10) // GLYPH_CELL)), max(1, int((bg[3] - 10) // GLYPH_CELL)))
                    if fit[0] < cols or fit[1] < rows:
                        per_page = fit
                        continue                    # rebuild the page at the size the viewport holds
                    per_page = (cols, rows)
                shot = T.rbx_screenshot(max_width=4096, name="glyph_probe")
                if not shot.get("ok"):
                    return dict(shot, ok=False, step="glyph capture", window=win.info())
                shots.append(shot["file"])
                w, h, rgba = P.decode_rgba(open(shot["file"], "rb").read())
                k = w / bg[2]
                for n, (fk, ch, _font, isref) in enumerate(page):
                    r = rects.get("G%03d" % n)
                    ink = _ink(rgba, w, h, [(r[0] - bg[0]) * k, (r[1] - bg[1]) * k, r[2] * k, r[3] * k]) if r else None
                    if isref:
                        ref[fk] = ink
                    else:
                        verdict[(fk, ch)] = ink
                i += len(page)
    finally:
        T.rbx_destroy("StarterGui." + name)
        if was is not True:
            SH._dev_gui(was)
    missing_total = 0
    for (fk, ch), ink in verdict.items():
        how = None
        if ink is not None and not ink:
            how = "blank"
        elif ink and _same_drawing(ink, ref.get(fk)):
            how = "box"
        if how:
            missing_total += 1
            for e in by_el[(fk, ch)]:
                e.setdefault("missingGlyphs", []).append({"char": ch, "codepoint": "U+%04X" % ord(ch), "draws": how,
                                                          "family": fk[0]})
    return {"ok": True, "pairsChecked": len(verdict), "fonts": len(fonts), "missing": missing_total, "shots": shots,
            "rule": "a character is missing when its one-character label draws no ink (blank) or draws its font's "
                    "U+10FFFD box (RB1600)"}


@_tool
def rbx_ui_layout(root: Any, max_elements: int = 500, glyphs: bool = False, play: bool = False) -> dict:
    "Roblox Studio: how a GUI tree is laid out now: every GuiBase2d under root with its AbsolutePosition and AbsoluteSize as a rectangle [x, y, w, h], Visible, and TextBounds and TextFits for text; then the findings - children that overflow their parent, visible siblings that overlap, text that does not fit, and for each UIListLayout its children's gaps against the Padding it holds (stackedWithPadding). A tree Studio has not laid out (every size 0) fails with layoutRan false rather than reporting a clean nothing. glyphs true also names, per text element, the characters its FontFace cannot draw (missingGlyphs): a width cannot tell (a missing character still measures an advance, RB1600), so each distinct (font, character) is drawn alone on white in a probe ScreenGui made under StarterGui (DisplayOrder 1,000,000, removed after; ShowDevelopmentGui held on for it and put back) and captured: no ink is blank, the font's own U+10FFFD box is box; emoji and other characters Roblox's fallback fonts draw count as drawn. At most 400 pairs a call; the probe is a write to StarterGui, so it needs Full mode. Read only otherwise. play true reads the PLAY CLIENT's PlayerGui instead (root under StarterGui or PlayerGui: StarterGui.Hud reads PlayerGui.Hud): a LocalScript in StarterPlayerScripts for one one-player play test (full mode or MifBridge's scratch place; removed after) waits until the client's camera ViewportSize.Y is 300 or more - a test client starts with a 1 x 1 viewport and is sized a moment later - and for the root, settles a second, and prints every GuiBase2d under it (Roblox's own guis under PlayerGui left out) with its rect, Visible and TextFits; the answer has the same rows by PlayerGui path and the same overflow, overlap and text findings, viewport and viewportWaited, and is not judged when the viewport never reached 300; glyphs is edit-only."
    if not isinstance(max_elements, int) or isinstance(max_elements, bool) or not 1 <= max_elements <= 2000:
        return _no("max_elements is 1-2000")
    if not isinstance(glyphs, bool) or not isinstance(play, bool):
        return _no("glyphs and play are true or false")
    if play:
        if glyphs:
            return _no("glyphs reads an edit-mode capture: run it without play")
        return _play_layout(root, max_elements)
    return _layout(root, max_elements, glyphs)


def _style_type(cat, prop):
    """The one type a property name has on the GUI classes, so a style rule's value is typed like a property write."""
    types = set()
    for cname, c in cat.classes.items():
        row = c["m"].get(prop)
        if row and row[0] == "P" and any(cat.is_a(cname, b) for b in ("GuiBase", "UIComponent")):
            types.add(row[1])
    return types


def _style_props(cat, props):
    """{name: Variant wire} for StyleRule:SetProperties, each typed by its property, or a refusal naming each problem."""
    if not isinstance(props, dict) or not props:
        return None, _no("properties is an object of property name to value")
    wire, bad = {}, {}
    for k, v in props.items():
        if isinstance(v, dict) and isinstance(v.get("$t"), str):
            w, why = V.encode("Variant", v, cat.enums)
        else:
            types = _style_type(cat, k)
            if not types:
                bad[k] = "no GUI class has a property %s" % k
                continue
            if len(types) > 1:
                bad[k] = "%s is %s on different GUI classes: give it as {\"$t\": type, ...}" % (k, " or ".join(sorted(types)))
                continue
            w, why = V.encode(types.pop(), v, cat.enums)
        if why:
            bad[k] = why
        else:
            wire[k] = w
    if bad:
        return None, _no("style properties refused, nothing was sent: %s" % "; ".join("%s: %s" % kv for kv in bad.items()),
                         refusedRule="type", refused=bad)
    return wire, None


@_tool
def rbx_ui_styles(action: str, sheet: Any = None, name: str = None, parent: Any = "ReplicatedStorage", selector: str = None,
                  properties: dict = None, priority: int = None, derives: list = None, gui: Any = None,
                  check_target: Any = None, transitions: dict = None) -> dict:
    "Roblox Studio: style sheets. action sheet makes a StyleSheet (name, under parent); rule adds a StyleRule to sheet with selector, priority and properties (each typed like a property write; a value given as {\"$t\": type, ...} is passed as that type), read back by StyleRule:GetProperties and an empty StyleRule.SelectorError, and with check_target also by Instance:GetStyled on that element for each property; link puts a StyleLink to sheet under gui; derives sets the sheet's derived sheets (StyleSheet:SetDerives) and reads them back (GetDerives); read lists the sheet's rules with their selectors, errors, priorities and properties. transitions {property: value} go through StyleRule:SetPropertyTransition, which has no getter: they answer verified false."
    cat, err = _cat()
    if err:
        return err
    if action not in ("sheet", "rule", "link", "derives", "read"):
        return _no("action is sheet, rule, link, derives or read")
    if action == "sheet":
        if not isinstance(name, str) or not name:
            return _no("sheet needs name")
        pparts, err = _parts(parent, "parent")
        if err:
            return err
        out = _T().rbx_create("StyleSheet", pparts, name=name)
        if out.get("ok") and derives:
            d = rbx_ui_styles("derives", sheet=out["path"], derives=derives)
            out = dict(out, derives=d, ok=bool(d.get("ok")), verified=bool(d.get("ok")), error=d.get("error"))
        return out
    sparts, err = _parts(sheet, "sheet")
    if err:
        return err
    scls, err = _class_of(sparts)
    if err:
        return err
    if not cat.is_a(scls, "StyleSheet") and action in ("derives", "link"):
        return _no("%s is a %s, not a StyleSheet" % (_join(sparts), scls), refusedRule="wrong-class")
    if action == "derives":
        if not isinstance(derives, list):
            return _no("derives is a list of StyleSheet paths ([] clears)")
        r = _call(sparts, "SetDerives", [derives])
        if not r.get("ok"):
            return r
        back = _call(sparts, "GetDerives")
        got = back.get("value")
        got = got.get("items") if isinstance(got, dict) else got
        want = [_join(V.split_path(d)) for d in derives]
        ok = back.get("ok") and list(got or []) == want
        return {"ok": bool(ok), "verified": bool(ok), "sheet": _join(sparts), "derives": got,
                "error": None if ok else "GetDerives reads %s, sent %s" % (got, want)}
    if action == "link":
        gparts, err = _parts(gui, "gui")
        if err:
            return err
        return _T().rbx_create("StyleLink", gparts, name=name or "StyleLink", props={"StyleSheet": _join(sparts)})
    if action == "read":
        rules = _call(sparts, "GetStyleRules")
        if not rules.get("ok"):
            return rules
        lst = rules.get("value")
        lst = lst.get("items") if isinstance(lst, dict) else (lst or [])
        out = []
        for rp in lst:
            vals, e = _read(rp, ["Selector", "SelectorError", "Priority"])
            props = _call(rp, "GetProperties")
            out.append({"rule": rp, "selector": (vals or {}).get("Selector"), "selectorError": (vals or {}).get("SelectorError"),
                        "priority": (vals or {}).get("Priority"), "properties": props.get("value"),
                        "error": (e or {}).get("error")})
        return {"ok": True, "sheet": _join(sparts), "className": scls, "rules": out, "count": len(out)}
    # rule
    if not isinstance(selector, str) or not selector:
        return _no("rule needs selector")
    wire, err = _style_props(cat, properties)
    if err:
        return err
    if priority is not None and (not isinstance(priority, int) or isinstance(priority, bool)):
        return _no("priority is a whole number")
    if transitions is not None and not isinstance(transitions, dict):
        return _no("transitions is an object of property name to value")
    rname = name or _free_name(sparts, "Rule")
    props = {"Selector": selector}
    if priority is not None:
        props["Priority"] = priority
    with _Group("MifBridge style rule") as g:
        if not g.open:
            return g.refused()
        made = _T().rbx_create("StyleRule", sparts, name=rname, props=props)
        if not made.get("ok"):
            return dict(made, step="create", cancelled=g.cancel())
        rparts = sparts + [rname]
        r = _rbx("call", target=rparts, method="SetProperties", args=[{"$t": "Dictionary", "v": wire}], argc=1,
                 readOnly=False, yields=False, callTimeout=30, timeout=45)
        if not r.get("ok"):
            return dict(r, step="SetProperties", cancelled=g.cancel())
        tr = {}
        for k, v in (transitions or {}).items():
            w, why = V.encode("Variant", v, cat.enums)
            if why:
                tr[k] = {"ok": False, "error": why}
                continue
            t = _rbx("call", target=rparts, method="SetPropertyTransition", args=[k, w], argc=2, readOnly=False,
                     yields=False, callTimeout=30, timeout=45)
            tr[k] = {"ok": bool(t.get("ok")), "verified": False, "error": t.get("error"),
                     "note": "StyleRule has no getter for a transition: sent, not read back"}
        # A selector Studio cannot use is read before the commit, so a rule that can never match is not left in the
        # sheet (live 0.748: "Text Button!!" was kept with SelectorError set until this check moved inside).
        early, _e = _read(rparts, ["SelectorError"])
        if (early or {}).get("SelectorError"):
            return {"ok": False, "verified": False, "cancelled": g.cancel(), "rule": _join(rparts), "selector": selector,
                    "selectorError": early["SelectorError"],
                    "error": "Studio reports the selector %r as: %s; the rule was not kept" % (selector, early["SelectorError"])}
        g.end(True)
    back = _rbx("call", target=rparts, method="GetProperties", args=[], argc=0, readOnly=True, yields=False,
                callTimeout=30, timeout=45)
    got = (back.get("returns") or [{}])[0] if back.get("ok") else {}
    # An empty Lua table crosses as an empty Array, so an empty answer is read as no properties at all.
    got_v = T_obj(got.get("v")) if isinstance(got, dict) else {}
    mism = {k: {"sent": V.decode(w), "readBack": V.decode((got_v or {}).get(k))} for k, w in wire.items()
            if not V.same(w, (got_v or {}).get(k, {"$t": "missing"}))}
    vals, _e = _read(rparts, ["SelectorError"])
    sel_err = (vals or {}).get("SelectorError")
    out = {"ok": not mism and not sel_err, "verified": not mism and not sel_err, "rule": _join(rparts),
           "selector": selector, "selectorError": sel_err, "properties": {k: V.decode(w) for k, w in wire.items()},
           "readBack": {k: V.decode(x) for k, x in (got_v or {}).items()}}
    if transitions:
        out["transitions"] = tr
    if mism:
        out.update(mismatch=mism, error="GetProperties holds something else for %s" % ", ".join(sorted(mism)))
    elif sel_err:
        out["error"] = "Studio reports the selector %r as: %s" % (selector, sel_err)
    if check_target is not None and out["ok"]:
        tparts, err = _parts(check_target, "check_target")
        if err:
            return err
        styled = {}
        for k, w in wire.items():
            s = _rbx("call", target=tparts, method="GetStyled", args=[k], argc=1, readOnly=True, yields=False,
                     callTimeout=30, timeout=45)
            got1 = (s.get("returns") or [{"$t": "nil"}])[0] if s.get("ok") else None
            styled[k] = {"value": V.decode(got1), "matches": s.get("ok") and V.same(w, got1), "error": s.get("error")}
        out["styled"] = {"target": _join(tparts), "properties": styled}
        if not all(v["matches"] for v in styled.values()):
            out.update(ok=False, verified=False,
                       error="the rule is held, but GetStyled on %s does not show it for %s" % (
                           _join(tparts), ", ".join(k for k, v in styled.items() if not v["matches"])))
    return out


@_tool
def rbx_ui_hit_test(gui: Any, x: int = None, y: int = None, center_of: Any = None) -> dict:
    "Roblox Studio: which GUI elements are under a screen point, topmost first (LayerCollector:GetGuiObjectsAtPosition on gui, a ScreenGui or other LayerCollector). Give x and y in pixels, or center_of an element to test its own center, which also says whether that element is the topmost there. Read only."
    gparts, err = _parts(gui, "gui")
    if err:
        return err
    target = None
    if center_of is not None:
        target, err = _parts(center_of, "center_of")
        if err:
            return err
        vals, err = _read(target, list(_RECT_PROPS))
        if err:
            return err
        rc = _rect(vals)
        if not rc:
            return _no("%s has no laid-out rectangle" % _join(target))
        x, y = int(round(rc[0] + rc[2] / 2)), int(round(rc[1] + rc[3] / 2))
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in (x, y)):
        return _no("give x and y (whole pixels) or center_of")
    r = _call(gparts, "GetGuiObjectsAtPosition", [x, y])
    if not r.get("ok"):
        return r
    hits = r.get("value")
    hits = hits.get("items") if isinstance(hits, dict) else (hits or [])
    out = {"ok": True, "gui": _join(gparts), "point": [x, y], "hits": hits, "topmost": hits[0] if hits else None}
    if target is not None:
        out["target"] = _join(target)
        out["topmostIsTarget"] = bool(hits) and hits[0] == _join(target)
    return out


@_tool
def rbx_text_measure(text: str = None, font: dict = None, size: float = None, width: float = None, legacy_font: str = None,
                     compare_to: Any = None, family_info: bool = False) -> dict:
    "Roblox Studio: the pixel size Studio gives a string. With font ({family, weight?, style?}, family an rbxasset:// or rbxassetid:// font family) it is TextService:GetTextBoundsAsync through a GetTextBoundsParams made in Workspace.MifScratch and removed after; with legacy_font (an Enum.Font name) it is TextService:GetTextSize. size is the text size in pixels, width wraps (default unbounded). compare_to a TextLabel, TextButton or TextBox fills text, font and size from it and says whether the measurement equals its TextBounds (to 1 pixel). family_info adds TextService:GetFamilyInfoAsync for the family."
    cat, err = _cat()
    if err:
        return err
    label = None
    if compare_to is not None:
        lparts, err = _parts(compare_to, "compare_to")
        if err:
            return err
        lcls, err = _class_of(lparts)
        if err:
            return err
        if cat.find_member(lcls, "TextBounds")[1] is None:
            return _no("%s is a %s, which has no TextBounds" % (_join(lparts), lcls), refusedRule="wrong-class")
        vals, err = _read(lparts, ["Text", "FontFace", "TextSize", "TextBounds", "TextWrapped", "AbsoluteSize", "TextFits"])
        if err:
            return err
        label = {"path": _join(lparts), "values": vals}
        text = vals.get("Text") if text is None else text
        font = vals.get("FontFace") if font is None and legacy_font is None else font
        size = vals.get("TextSize") if size is None else size
        if width is None and vals.get("TextWrapped"):
            width = (_xy(vals.get("AbsoluteSize")) or [None])[0]
    if not isinstance(text, str):
        return _no("text is a string (or give compare_to)")
    if not _num(size) or not 1 <= size <= 1000:
        return _no("size is the text size in pixels, 1-1000")
    if width is not None and (not _num(width) or width <= 0):
        return _no("width is a positive number of pixels")
    out = {"text": text, "size": size, "width": width}
    if legacy_font is not None:
        r = _call("TextService", "GetTextSize", [text, int(size), legacy_font, [width or 1e6, 1e6]])
        if not r.get("ok"):
            return r
        out.update(route="TextService:GetTextSize", font=legacy_font, bounds=_xy(r.get("value")))
    else:
        if not isinstance(font, dict):
            return _no("give font {family, weight?, style?} or legacy_font")
        fw, why = V.encode("Font", font, cat.enums)
        if why:
            return _no(why, refusedRule="type")
        pname = "MifTextParams_" + uuid.uuid4().hex[:8]
        props = {"Text": text, "Font": V.decode(fw), "Size": size}
        if width is not None:
            props["Width"] = width
        # Plugin 0.15.0 makes it with no parent at all (Studio locks its Parent at nil, RB403) and hands it back as a handle.
        mk = _T().rbx_create("GetTextBoundsParams", None, name=pname, props=props, outside=True)
        if mk.get("ok") and mk.get("handle"):
            h = mk["handle"]
            try:
                r = _call("TextService", "GetTextBoundsAsync", [{"handle": h}], timeout=30)
            finally:
                rel = _T().rbx_release_handle(h)
            out["paramsReleased"] = bool(rel.get("ok"))
            if not r.get("ok"):
                return dict(r, step="GetTextBoundsAsync")
            out.update(route="TextService:GetTextBoundsAsync", font=V.decode(fw), bounds=_xy(r.get("value")),
                       params="outside the place (a handle)")
            mk = None
        made_scratch = False
        if mk is not None:
            # an older plugin (or one that refused the parentless create): the scratch-folder route, which 0.748 locks
            out["outsideCreate"] = {k: mk.get(k) for k in ("refusedRule", "error")}
            made_scratch, err = _ensure_scratch()
            if err:
                return err
            mk = _T().rbx_create("GetTextBoundsParams", "Workspace.MifScratch", name=pname, props=props)
        if mk is None:
            pass
        elif not mk.get("ok"):
            if "locked" in str(mk.get("error")):
                # Live 0.748: "The Parent property of ... is locked, current parent: NULL". A GetTextBoundsParams lives
                # outside the place, and the plugin's create always parents: this route waits for a parentless create.
                return dict(mk, ok=False, refusedRule="needs-plugin", step="GetTextBoundsParams",
                            error="a GetTextBoundsParams cannot be parented (Studio: %s), and the plugin has no parentless "
                                  "create yet (a plugin request is filed); legacy_font measures "
                                  "through TextService:GetTextSize meanwhile" % mk.get("error"))
            return dict(mk, step="GetTextBoundsParams")
        else:
            try:
                r = _call("TextService", "GetTextBoundsAsync", ["Workspace.MifScratch." + pname], timeout=30)
            finally:
                gone = _T().rbx_destroy("Workspace.MifScratch." + pname)
            out["paramsRemoved"] = bool(gone.get("ok"))
            if not r.get("ok"):
                return dict(r, step="GetTextBoundsAsync")
            out.update(route="TextService:GetTextBoundsAsync", font=V.decode(fw), bounds=_xy(r.get("value")))
        if made_scratch:
            out["madeScratch"] = "Workspace.MifScratch"
    if family_info:
        fam = (font or {}).get("family") if isinstance(font, dict) else None
        if fam is None and label and isinstance(label["values"].get("FontFace"), dict):
            fam = label["values"]["FontFace"].get("family")
        if fam:
            fi = _call("TextService", "GetFamilyInfoAsync", [fam], timeout=30)
            out["familyInfo"] = fi.get("value") if fi.get("ok") else {"error": fi.get("error")}
    out["ok"] = out.get("bounds") is not None
    if label:
        tb = _xy(label["values"].get("TextBounds"))
        fits = label["values"].get("TextFits")
        out["label"] = {"path": label["path"], "textBounds": tb, "textFits": fits}
        same = bool(tb and out.get("bounds")) and all(abs(a - b) <= 1.0 for a, b in zip(tb, out["bounds"]))
        out["matchesLabel"] = same
        if text != label["values"].get("Text") or size != label["values"].get("TextSize"):
            out["matchesLabel"] = None
            out["note"] = "text or size differs from the label's own, so its TextBounds was not compared"
        elif fits is False:
            # Live 0.748: a wrapped label whose text overflows reports TextBounds for the lines that fit (300x20 in a
            # 30 px box, where GetTextSize measured 300x60), so its TextBounds is no yardstick.
            out["note"] = "the label's text does not fit (TextFits false): its TextBounds covers only what shows"
        elif not same:
            out.update(ok=False, error="measured %s, the label's TextBounds is %s" % (out.get("bounds"), tb))
    return out


_ENTRY_KEYS = ("Key", "Source", "Context", "Example", "Values")


def _entry_id(e):
    return (e.get("Key") or "", e.get("Source") or "", e.get("Context") or "")


# The strings a game shows, for rbx_localization scan (spec "Extend rbx_localization: find the strings a game shows,
# report the untranslated"): every TextLabel and TextButton's Text in a saved place (save_audit's text classes) and every
# string literal a script assigns to a .Text; a .Text built from pieces ("Coins: " .. n) is listed as joined, since a
# translation cannot reorder pieces put together at run time.
_SHOWN_TEXT = re.compile(r"\.Text\s*=\s*(.+)$")
_LITERAL = re.compile(r"""^\s*("([^"\\]|\\.)*"|'([^'\\]|\\.)*')\s*(--.*)?$""")


def shown_strings(sources):
    """(shown [{text, where, from}], joined [{where, line, text}]) from save_audit sources (scripts and text items)."""
    shown, joined = [], []
    for s in sources:
        where = ".".join(s["path"]) if isinstance(s.get("path"), list) else str(s.get("path"))
        if "text" in s:
            if (s.get("text") or "").strip():
                shown.append({"text": s["text"], "where": where, "from": s.get("className")})
            continue
        for n, ln in enumerate((s.get("source") or "").split("\n"), 1):
            m = _SHOWN_TEXT.search(ln)
            if not m or "==" in ln[m.start():m.start() + 8]:
                continue
            rhs = m.group(1).strip()
            lit = _LITERAL.match(rhs)
            if lit:
                text = lit.group(1)[1:-1]
                if text.strip():
                    shown.append({"text": text, "where": "%s line %d" % (where, n), "from": "script"})
            elif ".." in rhs and ('"' in rhs or "'" in rhs):
                joined.append({"where": where, "line": n, "text": ln.strip()[:160]})
    return shown, joined


def _scan(scan, entries, tparts):
    """rbx_localization scan: the shown strings against the table's entries, per locale. -> answer."""
    from mif_roblox import domain_logic as DL
    from mif_roblox import save_audit as SA
    if not isinstance(scan, dict) or set(scan) - {"file", "folder", "locales"}:
        return _no("scan is {file (a saved place) or folder (.lua/.luau files), locales?}")
    locales = scan.get("locales")
    if locales is not None and not (isinstance(locales, list) and all(isinstance(x, str) and x for x in locales)):
        return _no("scan.locales is a list of locale ids (fr, de, pt-br ...)")
    file, folder = scan.get("file"), scan.get("folder")
    if file is not None:
        import os
        if not isinstance(file, str) or not os.path.isfile(file):
            return _no("scan.file is an existing .rbxlx or .rbxl: %r" % (file,))
        try:
            sources, origin = SA.sources_from_file(file, text_classes=SA.TEXT_CLASSES)
        except Exception as exc:  # noqa: BLE001 - an unreadable place is named with its reason
            return _no("%s did not read as a place: %s: %s" % (file, type(exc).__name__, exc), ran=True)
    elif folder is not None:
        got, refusal = DL._audit_sources("game", None, folder, 20000)
        if got is None:
            return refusal
        sources, origin = got[0], "folder"
    else:
        return _no("scan needs file (a saved place: its GUI text and scripts) or folder (scripts)")
    if entries is None:
        got = _call(tparts, "GetEntries")
        if not got.get("ok"):
            return dict(got, ok=False, error="the table's entries did not read: %s" % got.get("error"))
        have = got.get("value")
        entries = have.get("items") if isinstance(have, dict) else (have or [])
        table_from = "studio"
    else:
        table_from = "entries"
    by_source = {}
    for e in entries:
        if isinstance(e, dict) and isinstance(e.get("Source"), str):
            by_source.setdefault(e["Source"], []).append(e)
    if locales is None:
        locales = sorted({loc for e in entries if isinstance(e, dict) for loc in (e.get("Values") or {})})
    shown, joined = shown_strings(sources)
    out = {"ran": True, "origin": origin, "tableFrom": table_from, "locales": locales, "shown": len(shown),
           "strings": sorted({s["text"] for s in shown})[:500], "joined": joined,
           "cannotSee": ["text a script builds at run time from data (names, numbers), or sets through a variable, is "
                         "not followed; a scan of a folder holds scripts only, no GUI text",
                         "whether a translation fits its box is measured live (rbx_text_measure), not here"]}
    if not shown:
        return dict(out, ok=False, error="no shown string was found in %d sources: nothing scanned is not clean"
                    % len(sources))
    missing, gaps = [], {loc: [] for loc in locales}
    for s in shown:
        rows = by_source.get(s["text"])
        if not rows:
            missing.append(s)
            continue
        for loc in locales:
            if not any((r.get("Values") or {}).get(loc) for r in rows):
                gaps[loc].append(s["text"])
    gaps = {k: sorted(set(v)) for k, v in gaps.items() if v}
    out.update(missing=missing, gaps=gaps, ok=not missing and not gaps and not joined)
    if not out["ok"]:
        out["error"] = "; ".join(x for x in (
            "%d shown string(s) not in the table (%s)" % (len(missing), ", ".join(repr(m["text"]) for m in missing[:5]))
            if missing else "",
            "untranslated in %s" % ", ".join("%s (%d)" % (k, len(v)) for k, v in sorted(gaps.items())) if gaps else "",
            "%d .Text joined from pieces (not translatable in order)" % len(joined) if joined else "") if x)
    return out


@_tool
def rbx_localization(table: Any, entries: list = None, source_locale: str = None, set_value: dict = None,
                     create: bool = True, scan: dict = None) -> dict:
    "Roblox Studio: a LocalizationTable's entries. table is its path; with create it is made there when missing. entries [{Key?, Source, Context?, Example?, Values: {localeId: text}}] replace the table's (LocalizationTable:SetEntries); set_value {key, source, context, locale, text} sets one translation (SetEntryValue); source_locale sets SourceLocaleId. Read back: GetEntries holds every entry and value sent, matched by Key, Source and Context. With nothing to write it reads the entries. With scan {file (a saved place) or folder (scripts), locales?} it writes nothing and finds the strings the game shows - every TextLabel and TextButton Text in the file and every string literal a script assigns to .Text - against the table's entries (read from Studio, or entries given in its place): missing lists a shown string the table lacks, gaps the ones untranslated per locale (locales, else every locale the table holds), joined a .Text built from pieces, which a translation cannot reorder; ok false on any, and a scan that finds no shown string fails."
    cat, err = _cat()
    if err:
        return err
    tparts, err = _parts(table, "table")
    if err:
        return err
    if scan is not None:
        if source_locale is not None or set_value is not None:
            return _no("scan reads only: give it alone, or with entries standing in for the table's")
        return dict(_scan(scan, entries, tparts), table=_join(tparts))
    out = {"table": _join(tparts)}
    if not _exists(tparts):
        if not create:
            return _no("no LocalizationTable at %s" % _join(tparts), refusedRule="not-found")
        mk = _T().rbx_create("LocalizationTable", tparts[:-1], name=tparts[-1])
        if not mk.get("ok"):
            return dict(mk, step="create")
        out["created"] = True
    else:
        cls, err = _class_of(tparts)
        if err:
            return err
        if cls != "LocalizationTable":
            return _no("%s is a %s" % (_join(tparts), cls), refusedRule="wrong-class")
    if source_locale is not None:
        r = _T().rbx_set_props(tparts, {"SourceLocaleId": source_locale})
        out["sourceLocale"] = {"ok": r.get("ok"), "after": (r.get("after") or {}).get("SourceLocaleId"), "error": r.get("error")}
        if not r.get("ok"):
            return dict(out, ok=False, error=r.get("error"))
    want = []
    if entries is not None:
        if not isinstance(entries, list) or not all(isinstance(e, dict) for e in entries):
            return _no("entries is a list of {Key?, Source, Context?, Example?, Values}")
        for i, e in enumerate(entries):
            extra = sorted(set(e) - set(_ENTRY_KEYS))
            if extra or not isinstance(e.get("Values", {}), dict) or not all(
                    isinstance(e.get(k, ""), str) for k in ("Key", "Source", "Context", "Example")):
                return _no("entry %d: keys are %s (strings, Values an object of localeId to text)%s" % (
                    i, ", ".join(_ENTRY_KEYS), "; unknown %s" % extra if extra else ""))
        r = _call(tparts, "SetEntries", [entries])
        if not r.get("ok"):
            return dict(out, ok=False, step="SetEntries", error=r.get("error"))
        want = list(entries)
    if set_value is not None:
        names = ("key", "source", "context", "locale", "text")
        if not isinstance(set_value, dict) or set(set_value) - set(names) or not all(
                isinstance(set_value.get(k, ""), str) for k in names) or "locale" not in set_value:
            return _no("set_value is {key, source, context, locale, text}, strings")
        sv = [set_value.get(k, "") for k in names]
        r = _call(tparts, "SetEntryValue", sv)
        if not r.get("ok"):
            return dict(out, ok=False, step="SetEntryValue", error=r.get("error"))
        want.append({"Key": sv[0], "Source": sv[1], "Context": sv[2], "Values": {sv[3]: sv[4]}})
    got = _call(tparts, "GetEntries")
    if not got.get("ok"):
        return dict(out, ok=False, error=got.get("error"))
    have = got.get("value")
    have = have.get("items") if isinstance(have, dict) else (have or [])
    out.update(entries=have, count=len(have))
    if not want:
        return dict(out, ok=True)
    index = {_entry_id(e): e for e in have if isinstance(e, dict)}
    missing = []
    for e in want:
        h = index.get(_entry_id(e))
        if h is None:
            missing.append({"entry": e, "readBack": None})
            continue
        bad = {k: {"sent": e[k], "readBack": h.get(k)} for k in ("Example",) if k in e and e[k] != (h.get(k) or "")}
        hv = h.get("Values") if isinstance(h.get("Values"), dict) else {}
        bad.update({"Values." + loc: {"sent": t, "readBack": hv.get(loc)} for loc, t in (e.get("Values") or {}).items()
                    if hv.get(loc) != t})
        if bad:
            missing.append({"entry": _entry_id(e), "mismatch": bad})
    out.update(ok=not missing, verified=not missing)
    if missing:
        out.update(problems=missing, error="%d entr%s not read back as sent" % (len(missing), "y" if len(missing) == 1 else "ies"))
    return out


# Phone screens in landscape points, the sizes a layout check walks with phones true: iPhone 12-16 (844 x 390), the Pro
# Max line (932 x 430), the mini (780 x 360), the 6-8 (667 x 375), the XR/11 (896 x 414) and a small Android (740 x 360).
PHONE_SIZES = [[844, 390], [932, 430], [780, 360], [667, 375], [896, 414], [740, 360]]
# A touch button's smallest side on a phone: 44 px, the touch-target floor a game lane used (found 2026-11-26: one
# UIScale on a whole ScreenGui fit the HUD to a phone and took its buttons under it). AbsoluteSize already holds every
# UIScale above an element, so the laid-out rect is the size a finger meets.
TOUCH_MIN_PX = 44
PHONE_DEVICE = re.compile(r"(?i)phone|iphone|galaxy|pixel|android|mobile")


def _small_touch_buttons(lay, run):
    """[{path, rect}] of the visible GuiButtons whose laid-out width or height is under TOUCH_MIN_PX, on a run that is a
    phone (a phone device id, or one of PHONE_SIZES); None on another run."""
    if not (PHONE_DEVICE.search(str(run.get("device") or "")) or run.get("resolution") in PHONE_SIZES):
        return None
    cat, err = _cat()
    if err:
        return None
    out = []
    for e in lay.get("elements") or []:
        r = e.get("rect")
        if not r or e.get("visible") is False or not cat.is_a(e.get("className") or "", "GuiButton"):
            continue
        if min(r[2], r[3]) < TOUCH_MIN_PX:
            out.append({"path": e["path"], "rect": r})
    return out


def _bar_rects(bar, width):
    """The floating bar's button groups as rectangles [x, y, w, h] in the gui's own coordinates: the left group from 0
    to left_end, and the right group from right_start to the gui's width when given (GuiService.TopbarInset: Max.Y the
    height, Min.X where the left buttons end, Max.X where the right ones start)."""
    out = [[0, 0, bar["left_end"], bar["height"]]]
    if bar.get("right_start") is not None and width and bar["right_start"] < width:
        out.append([bar["right_start"], 0, width - bar["right_start"], bar["height"]])
    return out


TOPBAR_SCRIPT = "MifTopbarProbe"
TOPBAR_MARK = "MIFBAR:"
# Read in the play client of a SOLO play test: under the device simulator that client is the simulated phone (touch,
# its viewport), and its top bar is drawn; edit mode has none (RB1601) and a multiplayer test's client is a desktop
# window the simulator does not reach (RB1444). TopbarInset reads 0 for about the first second (RB1800).
TOPBAR_SOURCE = '''
local GS = game:GetService("GuiService")
local UIS = game:GetService("UserInputService")
local HS = game:GetService("HttpService")
local t0 = os.clock()
while GS.TopbarInset.Max.Y <= 0 and os.clock() - t0 < %(wait)s do
	task.wait(0.1)
end
task.wait(1)
local ti = GS.TopbarInset
local a, b = GS:GetGuiInset()
local cam = workspace.CurrentCamera
local r = { min = { ti.Min.X, ti.Min.Y }, max = { ti.Max.X, ti.Max.Y }, guiInset = { { a.X, a.Y }, { b.X, b.Y } },
	viewport = { cam.ViewportSize.X, cam.ViewportSize.Y }, touch = UIS.TouchEnabled, waited = os.clock() - t0 }
pcall(function() r.preferredInput = UIS.PreferredInput.Name end)
pcall(function() r.displaySize = GS.ViewportDisplaySize.Name end)
'''


def _play_bar():
    """The floating bar as the simulated device's play client reads it: (bar, client, error, wrote)."""
    from mif_roblox import domain_controller as DC
    src = TOPBAR_SOURCE % {"wait": 6} + DC.CLIENT_EMIT % {"mark": TOPBAR_MARK, "chunk": DC.CHUNK}
    res, err, wrote = DC.play_client(src, TOPBAR_MARK, TOPBAR_SCRIPT, 12, players=None)
    if err:
        return None, None, err, wrote
    mn, mx, vp = res.get("min") or [0, 0], res.get("max") or [0, 0], res.get("viewport") or [0, 0]
    client = {k: res.get(k) for k in ("min", "max", "guiInset", "viewport", "touch", "preferredInput", "displaySize")}
    if not mx[1]:
        return None, client, "the play client read GuiService.TopbarInset max %s after %.1f s: no top bar" % (
            mx, res.get("waited") or 0), wrote
    bar = {"height": mx[1], "left_end": mn[0], "right_start": mx[0] if mx[0] < vp[0] else None, "from": "play client"}
    return bar, client, None, wrote


def _under_bar(lay, bar):
    """Elements laid out under the floating bar. A ScreenGui that keeps out of the top bar (IgnoreGuiInset false,
    CoreUISafeInsets) is laid out below it in play, so nothing in it is counted."""
    guis = {}
    for e in lay.get("elements") or []:
        if e["className"] == "ScreenGui":
            g = _T().rbx_get_props(e["path"], ["IgnoreGuiInset", "AbsoluteSize"])
            guis[e["path"]] = (g.get("values") or {}) if g.get("ok") else {}
    under, clear_guis = [], []
    for e in lay.get("elements") or []:
        if not e.get("rect") or e.get("visible") is False or e["className"] == "ScreenGui":
            continue
        gpath = next((gp for gp in guis if e["path"].startswith(gp + ".")), None)
        gv = guis.get(gpath, {})
        if gv.get("IgnoreGuiInset") is False:
            if gpath not in clear_guis:
                clear_guis.append(gpath)
            continue
        width = (_xy(gv.get("AbsoluteSize")) or [None])[0]
        for i, br in enumerate(_bar_rects(bar, width)):
            a = _overlap(e["rect"], br)
            if a:
                under.append({"path": e["path"], "rect": e["rect"], "group": "left" if i == 0 else "right", "area": round(a, 1)})
    return under, clear_guis


@_tool
def rbx_ui_device_check(root: Any, devices: list = None, resolutions: list = None, screenshots: bool = True,
                        phones: bool = False, floating_bar: Any = None) -> dict:
    "Roblox Studio: one GUI on several devices. For each device id (rbx_device_sim list gives them) the device simulator is set, the tree under root is laid out again (rbx_ui_layout) and, with screenshots, the viewport captured; resolutions [[w, h], ...] are tried on the first device. The simulator is stopped after. Answers each device's rectangles and which elements changed rectangle between devices. The simulator changes the viewport only; it is UNVERIFIED that edit-mode layout follows it, and this answers what it measured. phones true adds six landscape phone sizes (844 x 390, 932 x 430, 780 x 360, 667 x 375, 896 x 414, 740 x 360) on the first device. floating_bar {height, left_end, right_start?} in pixels models Roblox's button bar, which floats over the top-left of a phone screen: each run answers underBar, every visible element whose rectangle meets the bar's left group (0 to left_end, 0 to height) or right group; a ScreenGui with IgnoreGuiInset false is laid out below the bar in play, so its elements are named clear instead. The numbers are a play client's GuiService.TopbarInset (Max.Y, Min.X, Max.X); edit mode has no top bar (RB1601). floating_bar \"auto\" measures them per device and size: a solo play test under the simulator, whose client is that phone (RB1800), about 15 s each, full mode or MifBridge's scratch place. On a phone run (a phone device id, or a PHONE_SIZES resolution) every visible GuiButton whose laid-out width or height is under 44 px - its AbsoluteSize, which already holds every UIScale above it (RB3080) - is named in smallTouchButtons and touchWarning."
    T = _T()
    if devices is not None and (not isinstance(devices, list) or not all(isinstance(d, str) and d for d in devices)):
        return _no("devices is a list of device ids")
    if resolutions is not None and (not isinstance(resolutions, list) or not all(
            isinstance(r, list) and len(r) == 2 and all(isinstance(x, int) and 100 <= x <= 8192 for x in r) for r in resolutions)):
        return _no("resolutions is a list of [width, height], 100-8192")
    if not devices:
        return _no("give devices: ids from rbx_device_sim('list')")
    if not isinstance(phones, bool):
        return _no("phones is true or false")
    bar = None
    auto = floating_bar == "auto"            # measured per run below, in a solo play test's client (RB1601, RB1800)
    if floating_bar is not None and not auto:
        fb = floating_bar
        if not isinstance(fb, dict) or set(fb) - {"height", "left_end", "right_start"} or not all(
                isinstance(fb.get(k), (int, float)) and not isinstance(fb.get(k), bool) and fb.get(k) >= 0
                for k in ("height", "left_end")) or (fb.get("right_start") is not None and not isinstance(fb["right_start"], (int, float))):
            return _no("floating_bar is {height, left_end, right_start?} in pixels (a play client's GuiService.TopbarInset "
                       "Max.Y, Min.X, Max.X), or \"auto\"")
        bar = dict(fb, **{"from": "given"})
    if phones:
        resolutions = list(resolutions or []) + [r for r in PHONE_SIZES if r not in (resolutions or [])]
    runs = []
    try:
        plan = [("device", d, None) for d in devices] + [("resolution", devices[0], r) for r in (resolutions or [])]
        for kind, d, res in plan:
            s = T.rbx_device_sim("set", device=d, verify=False)
            if res is not None and s.get("ok"):
                s = T.rbx_device_sim("resolution", width=res[0], height=res[1], verify=False)
            run = {"device": d, "resolution": res, "set": {"ok": s.get("ok"), "error": s.get("error")}}
            if s.get("ok") and auto:
                bar, client, berr, wrote = _play_bar()
                run["floatingBar"], run["client"] = bar, client
                if berr:
                    run["barError"] = berr
                if wrote and not wrote.get("removed"):
                    run["barError"] = "the probe LocalScript %s was not removed" % wrote["path"]
                    bar = run["floatingBar"] = None
                # the simulator is set again for the edit-mode layout, whatever the play test left behind
                s = T.rbx_device_sim("set", device=d, verify=False)
                if res is not None and s.get("ok"):
                    s = T.rbx_device_sim("resolution", width=res[0], height=res[1], verify=False)
                run["set"] = {"ok": s.get("ok"), "error": s.get("error")}
            if s.get("ok"):
                time.sleep(1.0)
                lay = _layout(root)
                run["layoutRan"] = lay.get("layoutRan")
                run["rects"] = {e["path"]: e["rect"] for e in lay.get("elements") or []}
                run["findings"] = {k: len(lay.get(k) or []) for k in ("overflow", "overlaps", "textOverflow")}
                small = _small_touch_buttons(lay, run)
                if small is not None:
                    run["smallTouchButtons"] = small
                if bar:
                    under, clear_guis = _under_bar(lay, bar)
                    run["underBar"] = under
                    if clear_guis:
                        run["clearOfBar"] = clear_guis
                if screenshots:
                    shot = T.rbx_screenshot(name="device_check")
                    run["screenshot"] = {"file": shot.get("file"), "width": shot.get("width"), "height": shot.get("height"),
                                         "ok": shot.get("ok"), "error": shot.get("error")}
            runs.append(run)
    finally:
        stop = T.rbx_device_sim("stop")
    good = [r for r in runs if r.get("rects")]
    changed = []
    if len(good) > 1:
        for p in good[0]["rects"]:
            seen = [r["rects"].get(p) for r in good]
            if any(s != seen[0] for s in seen[1:]):
                changed.append({"path": p, "rects": seen})
    ok = len(good) == len(runs) and all(r.get("layoutRan") for r in good)
    out = {"ok": ok, "runs": runs, "changedBetweenDevices": changed, "simulatorStopped": bool(stop.get("ok"))}
    small = sorted({s["path"] for r in runs for s in r.get("smallTouchButtons") or []})
    if any("smallTouchButtons" in r for r in runs):
        out["smallTouchButtons"] = small
        if small:
            out["touchWarning"] = ("%d button(s) are under %d px on a phone run, after every UIScale above them: %s. Scale "
                                   "information panels with their own UIScale and keep buttons at full size" % (
                                       len(small), TOUCH_MIN_PX, ", ".join(small[:5])))
    if bar or auto:
        out["floatingBar"] = {"from": "play client, per run"} if auto else bar
        out["underBar"] = sorted({u["path"] for r in runs for u in r.get("underBar") or []})
    if not ok:
        out["error"] = "a device could not be set or its layout did not run: %s" % "; ".join(
            "%s: %s" % (r["device"], r["set"].get("error") or "layoutRan %s" % r.get("layoutRan")) for r in runs if not r.get("rects"))
    bad_bar = [r for r in runs if auto and r.get("rects") and not r.get("floatingBar")]
    if bad_bar:
        out["ok"] = False
        out["error"] = "the floating bar was not measured for %s: %s" % (
            ", ".join("%s %s" % (r["device"], r["resolution"] or "") for r in bad_bar), bad_bar[0].get("barError"))
    return out


@_tool
def rbx_world_gui(kind: str, adornee: Any, props: dict = None, name: str = None, parent: Any = None, children: list = None,
                  replace: bool = False) -> dict:
    "Roblox Studio: a BillboardGui (kind billboard) or SurfaceGui (kind surface) shown on adornee, a part, with its Adornee set and props (BillboardGui StudsOffset, MaxDistance, Size...; SurfaceGui Face, PixelsPerStud, SizingMode...) and optional children as rbx_ui_build elements, made in one recording under parent (default the adornee). Read back: every property sent, Adornee included, and every element by path."
    cls = {"billboard": "BillboardGui", "surface": "SurfaceGui"}.get(kind)
    if cls is None:
        return _no("kind is billboard or surface")
    aparts, err = _parts(adornee, "adornee")
    if err:
        return err
    if props is not None and not isinstance(props, dict):
        return _no("props is an object")
    p = dict(props or {})
    p["Adornee"] = _join(aparts)
    node = {"class": cls, "name": name or cls, "props": p, "children": children or []}
    return _build_tree(node, parent if parent is not None else aparts, UI_BASES, "element", replace)


# ================================================================ batch 9: physics, constraints, navigation
def _attachment(part_parts, point, axis, name, existing=None):
    """An Attachment under a part at a world point (default the part's pivot), its X axis along axis when given.
    Create, then WorldCFrame (a create sets Parent last, so a world value set in it would be relative to nothing)."""
    if existing is not None:
        return {"ok": True, "path": _join(existing), "made": False}, existing
    mk = _T().rbx_create("Attachment", part_parts, name=name)
    if not mk.get("ok"):
        return mk, None
    ap = part_parts + [name]
    if point is None and axis is None:
        return {"ok": True, "path": _join(ap), "made": True}, ap
    if point is None:
        pv = _call(part_parts, "GetPivot")
        point = (pv.get("value") or {}).get("position") if pv.get("ok") else None
        if point is None:
            return dict(pv, ok=False, error="the part's pivot could not be read: %s" % pv.get("error")), None
    rot = _basis_x(axis) if axis is not None else [1, 0, 0, 0, 1, 0, 0, 0, 1]
    if rot is None:
        return _no("axis is a non-zero direction"), None
    st = _T().rbx_set_props(ap, {"WorldCFrame": {"position": list(point), "rotation": rot}})
    if not st.get("ok"):
        return dict(st, step="WorldCFrame"), None
    return {"ok": True, "path": _join(ap), "made": True, "worldCFrame": (st.get("after") or {}).get("WorldCFrame")}, ap


@_tool
def rbx_constraint(class_name: str, part0: Any, part1: Any = None, point0: list = None, point1: list = None,
                   axis: list = None, props: dict = None, name: str = None, parent: Any = None,
                   attachment0: Any = None, attachment1: Any = None) -> dict:
    "Roblox Studio: any constraint (HingeConstraint, BallSocketConstraint, RopeConstraint, AlignPosition, VectorForce...) between part0 and part1 at world points, in one undo recording. An Attachment is made in each part at point0 / point1 (default each part's pivot), its X axis along axis when given (a hinge turns about it), unless attachment0 / attachment1 name existing ones; part1 may be left out for a one-attachment constraint. props are the class's own (ActuatorType, LimitsEnabled, Mode...), typed by the catalog. Read back: each attachment's WorldCFrame, every property sent, Attachment0 and Attachment1, then Enabled and Active (Active is true once Studio's solver holds both ends in the world). Anything failing cancels the recording."
    cat, err = _cat()
    if err:
        return err
    bad = cat.create_refusal(class_name)
    if bad:
        return bad
    if not cat.is_a(class_name, "Constraint"):
        return _no("%s is not a Constraint" % class_name, refusedRule="not-a-constraint")
    p0, err = _parts(part0, "part0")
    if err:
        return err
    p1 = None
    if part1 is not None:
        p1, err = _parts(part1, "part1")
        if err:
            return err
    for pt, nm in ((point0, "point0"), (point1, "point1"), (axis, "axis")):
        if pt is not None and not _vec(pt):
            return _no("%s is [x, y, z]" % nm)
    if props is not None:
        _w, err = _T()._encode_props(cat, class_name, props)
        if err:
            return err
        if set(props) & {"Attachment0", "Attachment1"}:
            return _no("Attachment0 and Attachment1 are made here: use attachment0 / attachment1 for existing ones")
    ex0 = ex1 = None
    if attachment0 is not None:
        ex0, err = _parts(attachment0, "attachment0")
        if err:
            return err
    if attachment1 is not None:
        ex1, err = _parts(attachment1, "attachment1")
        if err:
            return err
    for pp in [p0] + ([p1] if p1 else []):
        c, err = _class_of(pp)
        if err:
            return err
        if not cat.is_a(c, "BasePart"):
            return _no("%s is a %s, not a part" % (_join(pp), c), refusedRule="wrong-class")
    cname = name or class_name
    par = p0 if parent is None else V.split_path(parent)
    if not par:
        return _no("parent is a path")
    out = {"className": class_name}
    with _Group("MifBridge constraint") as g:
        if not g.open:
            return dict(out, **g.refused())
        a0, ap0 = _attachment(p0, point0, axis, _free_name(p0, cname + "_A0"), ex0)
        out["attachment0"] = a0
        ap1 = None
        if a0.get("ok") and (p1 is not None or ex1 is not None):
            a1, ap1 = _attachment(p1 or ex1[:-1], point1, axis, _free_name(p1 or ex1[:-1], cname + "_A1"), ex1)
            out["attachment1"] = a1
            if not a1.get("ok"):
                return dict(out, ok=False, verified=False, cancelled=g.cancel(), error="attachment1: %s" % a1.get("error"))
        if not a0.get("ok"):
            return dict(out, ok=False, verified=False, cancelled=g.cancel(), error="attachment0: %s" % a0.get("error"))
        full = dict(props or {})
        full["Attachment0"] = _join(ap0)
        if ap1:
            full["Attachment1"] = _join(ap1)
        mk = _T().rbx_create(class_name, par, name=_free_name(par, cname), props=full)
        out["constraint"] = {k: mk.get(k) for k in ("ok", "verified", "path", "after", "mismatch", "error")}
        if not mk.get("ok"):
            return dict(out, ok=False, verified=False, cancelled=g.cancel(), error="the constraint: %s" % mk.get("error"))
        g.end(True)
    vals, _e = _read(mk["path"], ["Enabled", "Active"])
    out.update(ok=True, verified=True, path=mk["path"], enabled=(vals or {}).get("Enabled"), active=(vals or {}).get("Active"))
    if out["enabled"] and out["active"] is False:
        out["note"] = ("Enabled but not Active: Studio's solver does not hold this constraint now (both parts must be in "
                       "the world and not welded into one assembly with nothing free)")
    return out


@_tool
def rbx_collision_groups(register: list = None, rename: dict = None, unregister: list = None, collide: list = None,
                         assign: dict = None, matrix: bool = False) -> dict:
    "Roblox Studio: collision groups. register names new groups; rename {old: new}; collide [[a, b, true|false], ...] sets whether two groups collide (PhysicsService:CollisionGroupSetCollidable); assign {part path: group} sets BasePart.CollisionGroup; unregister removes groups; in that order. Refused before anything is sent: more groups than PhysicsService:GetMaxCollisionGroups. Read back: IsCollisionGroupRegistered for each group named, CollisionGroupsAreCollidable for every pair set, each part's CollisionGroup. matrix true (or nothing to write) reads every pair of registered groups."
    for v, nm, typ in ((register, "register", list), (unregister, "unregister", list), (collide, "collide", list),
                       (rename, "rename", dict), (assign, "assign", dict)):
        if v is not None and not isinstance(v, typ):
            return _no("%s is a %s" % (nm, typ.__name__))
    if not all(isinstance(n, str) and n for n in (register or []) + (unregister or [])):
        return _no("group names are non-empty strings")
    if not all(isinstance(t, list) and len(t) == 3 and isinstance(t[0], str) and isinstance(t[1], str)
               and isinstance(t[2], bool) for t in collide or []):
        return _no("collide is a list of [group, group, true|false]")
    svc = "PhysicsService"
    mx = _call(svc, "GetMaxCollisionGroups")
    reg = _call(svc, "GetRegisteredCollisionGroups")
    if not mx.get("ok") or not reg.get("ok"):
        return mx if not mx.get("ok") else reg
    groups = reg.get("value")
    groups = groups.get("items") if isinstance(groups, dict) else (groups or [])
    names = [g.get("name") if isinstance(g, dict) else g for g in groups]
    out = {"max": mx.get("value"), "before": names}
    new = [n for n in register or [] if n not in names]
    if isinstance(mx.get("value"), int) and len(names) + len(new) > mx["value"]:
        return _no("%d groups exist, %d more asked for, and Studio allows %d" % (len(names), len(new), mx["value"]),
                   refusedRule="too-many-groups", **out)
    steps, problems, rules = [], [], set()

    def step(label, r):
        steps.append({"step": label, "ok": bool(r.get("ok")), "error": r.get("error"), "refusedRule": r.get("refusedRule")})
        if not r.get("ok"):
            problems.append("%s: %s" % (label, r.get("error")))
            rules.add(r.get("refusedRule"))
        return r.get("ok")

    writing = any((register, rename, unregister, collide, assign))
    if writing:
        with _Group("MifBridge collision groups") as g:
            if not g.open:
                return dict(out, **g.refused())
            for n in register or []:
                if n in names:
                    steps.append({"step": "register %s" % n, "ok": True, "already": True})
                    continue
                step("register %s" % n, _call(svc, "RegisterCollisionGroup", [n]))
            for a, b in (rename or {}).items():
                step("rename %s to %s" % (a, b), _call(svc, "RenameCollisionGroup", [a, b]))
            for a, b, c in collide or []:
                step("collide %s %s %s" % (a, b, c), _call(svc, "CollisionGroupSetCollidable", [a, b, c]))
            for p, grp in (assign or {}).items():
                step("assign %s" % p, _T().rbx_set_props(p, {"CollisionGroup": grp}))
            for n in unregister or []:
                step("unregister %s" % n, _call(svc, "UnregisterCollisionGroup", [n]))
            cancelled = g.cancel() if problems else False
            if not problems:
                g.end(True)
        out["steps"] = steps
        if problems:
            return dict(out, ok=False, verified=False, cancelled=cancelled, error="; ".join(problems),
                        refusedRule=rules.pop() if len(rules) == 1 else None)
    # Read back each thing named.
    checks = []
    for n in list(register or []) + list((rename or {}).values()):
        r = _call(svc, "IsCollisionGroupRegistered", [n])
        checks.append({"check": "registered %s" % n, "ok": r.get("value") is True})
    for n in list(rename or {}) + list(unregister or []):
        r = _call(svc, "IsCollisionGroupRegistered", [n])
        checks.append({"check": "gone %s" % n, "ok": r.get("ok") and r.get("value") is False})
    for a, b, c in collide or []:
        r = _call(svc, "CollisionGroupsAreCollidable", [a, b])
        checks.append({"check": "collide %s %s" % (a, b), "ok": r.get("value") is c, "readBack": r.get("value")})
    for p, grp in (assign or {}).items():
        vals, _e = _read(p, ["CollisionGroup"])
        checks.append({"check": "assign %s" % p, "ok": (vals or {}).get("CollisionGroup") == grp,
                       "readBack": (vals or {}).get("CollisionGroup")})
    bad = [c for c in checks if not c["ok"]]
    out.update(checks=checks, ok=not bad, verified=not bad if writing else None)
    if bad:
        out["error"] = "not read back: %s" % ", ".join(c["check"] for c in bad)
    if matrix or not writing:
        reg = _call(svc, "GetRegisteredCollisionGroups")
        lst = reg.get("value")
        lst = lst.get("items") if isinstance(lst, dict) else (lst or [])
        now = [g.get("name") if isinstance(g, dict) else g for g in lst]
        m = {}
        for i, a in enumerate(now):
            for b in now[i:]:
                r = _call(svc, "CollisionGroupsAreCollidable", [a, b])
                m["%s|%s" % (a, b)] = r.get("value")
        out.update(groups=now, matrix=m)
        if not now:
            out.update(ok=False, error="PhysicsService lists no collision groups (Default is always there): nothing was read")
    return out


_PP_NAMES = ("density", "friction", "elasticity", "frictionWeight", "elasticityWeight")


@_tool
def rbx_physical_properties(target: Any, density: float = None, friction: float = None, elasticity: float = None,
                            friction_weight: float = None, elasticity_weight: float = None, material: str = None,
                            massless: bool = None) -> dict:
    "Roblox Studio: a part's physical properties and the mass that results. Any of density, friction, elasticity, friction_weight, elasticity_weight sets CustomPhysicalProperties (the ones not given keep the part's current values); material sets them to that Material's defaults; massless sets Massless. Read back: CustomPhysicalProperties and Massless as sent, and CurrentPhysicalProperties, Mass and AssemblyMass before and after, with the mass ratio. With nothing given it reads."
    parts, err = _parts(target)
    if err:
        return err
    names = ["CustomPhysicalProperties", "CurrentPhysicalProperties", "Mass", "AssemblyMass", "Massless", "Material"]
    before, err = _read(parts, names)
    if err:
        return err
    given = dict(zip(_PP_NAMES, (density, friction, elasticity, friction_weight, elasticity_weight)))
    given = {k: v for k, v in given.items() if v is not None}
    if not all(_num(v) for v in given.values()):
        return _no("density, friction, elasticity and the weights are numbers")
    if given and material is not None:
        return _no("give numbers or material, not both")
    if massless is not None and not isinstance(massless, bool):
        return _no("massless is true or false")
    out = {"target": _join(parts), "before": before}
    props = {}
    if given:
        cur = before.get("CustomPhysicalProperties") or before.get("CurrentPhysicalProperties") or {}
        if not isinstance(cur, dict) or not all(k in cur for k in _PP_NAMES):
            return _no("the part's current physical properties could not be read to fill the rest: %r" % (cur,))
        props["CustomPhysicalProperties"] = {k: given.get(k, cur[k]) for k in _PP_NAMES}
        # Live 0.748: five numbers alone reset AcousticAbsorption to Studio's default 1 (it was 0.3): carry it over.
        if _num(cur.get("acousticAbsorption")):
            props["CustomPhysicalProperties"]["acousticAbsorption"] = cur["acousticAbsorption"]
    elif material is not None:
        props["CustomPhysicalProperties"] = material
    if massless is not None:
        props["Massless"] = massless
    if not props:
        return dict(out, ok=True, values=before)
    st = _T().rbx_set_props(parts, props)
    out["set"] = {k: st.get(k) for k in ("ok", "verified", "after", "mismatch", "error")}
    after, _e = _read(parts, names)
    out["after"] = after
    m0, m1 = before.get("Mass"), (after or {}).get("Mass")
    if _num(m0) and _num(m1) and m0:
        out["massRatio"] = round(m1 / m0, 6)
    d0 = (before.get("CurrentPhysicalProperties") or {}).get("density") if isinstance(before.get("CurrentPhysicalProperties"), dict) else None
    d1 = ((after or {}).get("CurrentPhysicalProperties") or {}).get("density") if isinstance((after or {}).get("CurrentPhysicalProperties"), dict) else None
    if _num(d0) and _num(d1) and d0:
        out["densityRatio"] = round(d1 / d0, 6)
    out.update(ok=bool(st.get("ok")), verified=bool(st.get("verified")), error=st.get("error"))
    return out


@_tool
def rbx_physics_step(parts: list, steps: int = 60, dt: float = 1 / 60, restore: bool = True) -> dict:
    "Roblox Studio: step physics in edit mode on a set of parts (WorldRoot:StepPhysics, the plugin's route; WorldRoot:Simulate is RobloxSecurity) and say where each went: pivot before and after, displacement and AssemblyLinearVelocity. steps 1-600 of dt seconds (at most 0.1). The steps run in one undo recording; with restore that recording is cancelled and every pivot read again, and if the cancel did not put a part back it is moved back to its whole pivot from before, position and rotation (PVInstance:PivotTo), with its linear and angular velocity zeroed. Restored means both position and rotation match. The answer says which restore worked, measured. Refused, with nothing stepped, when no undo group can be opened. With plugin 0.15.0 the steps are one step_physics op that waits a frame after each step (StepPhysics called many times in one frame does not advance by every step, RB600), allowed in scratch mode when every part is under Workspace.MifScratch; route says which ran."
    if not isinstance(parts, list) or not 1 <= len(parts) <= 100:
        return _no("parts is a list of 1-100 part paths")
    if not isinstance(steps, int) or isinstance(steps, bool) or not 1 <= steps <= 600:
        return _no("steps is 1-600", refusedRule="too-many-steps" if isinstance(steps, int) and steps > 600 else "type")
    if not _num(dt) or not 0 < dt <= 0.1:
        return _no("dt is above 0 and at most 0.1 seconds")
    pl = []
    for p in parts:
        pp, err = _parts(p, "part")
        if err:
            return err
        pl.append(pp)

    def pivots():
        """Each part's whole pivot {position, rotation}, or None when it could not be read or has no such shape."""
        out = []
        for pp in pl:
            r = _call(pp, "GetPivot")
            v = r.get("value") if r.get("ok") else None
            ok = (isinstance(v, dict) and _vec(v.get("position"))
                  and isinstance(v.get("rotation"), (list, tuple)) and len(v["rotation"]) == 9
                  and all(_num(x) for x in v["rotation"]))
            out.append({"position": list(v["position"]), "rotation": list(v["rotation"])} if ok else None)
        return out

    def same_pivot(n, b):
        """Back where it was: position within 1e-3 studs and every rotation entry within 1e-4, so a part put back on its
        spot lying on its side is not restored."""
        return bool(n and b and all(abs(x - y) <= 1e-3 for x, y in zip(n["position"], b["position"]))
                    and all(abs(x - y) <= 1e-4 for x, y in zip(n["rotation"], b["rotation"])))

    def pos(p):
        return p["position"] if p else None

    before = pivots()
    if any(b is None for b in before):
        return _no("could not read the pivot of %s" % ", ".join(_show(p) for p, b in zip(pl, before) if b is None))
    joined = [_join(p) for p in pl]
    ran, err_step = 0, None
    route = "step_physics"
    with _Group("MifBridge physics step") as g:
        if not g.open:
            return g.refused()
        # Plugin 0.15.0: every step in one op (and in scratch mode when every part is under MifScratch); older plugins
        # answer unknown-op and the steps go one call each.
        one = _T()._rbx("step_physics", parts=pl, count=steps, dt=dt, timeout=120)
        if one.get("ok"):
            ran = int(one.get("steps") or 0)
        elif one.get("refusedRule") == "unknown-op":
            route = "call"
            for _i in range(steps):
                r = _call("Workspace", "StepPhysics", [dt, joined])
                if not r.get("ok"):
                    err_step = r
                    break
                ran += 1
        else:
            err_step = one
        after = pivots()
        vel = []
        for pp in pl:
            v, _e = _read(pp, ["AssemblyLinearVelocity"])
            vel.append(_xyz((v or {}).get("AssemblyLinearVelocity")))
        if restore or err_step:
            g.end(False)
        else:
            g.end(True)
    rows = [{"part": j, "before": pos(b), "after": pos(a), "displacement": _sub(pos(a), pos(b)) if a and b else None,
             "rotationBefore": b["rotation"] if b else None, "rotationAfter": a["rotation"] if a else None, "velocity": v}
            for j, b, a, v in zip(joined, before, after, vel)]
    out = {"ok": err_step is None, "stepsRun": ran, "dt": dt, "seconds": round(ran * dt, 4), "parts": rows,
           "undoGroup": bool((g.began or {}).get("ok")), "route": route}
    if err_step:
        out["error"] = "step %d failed: %s" % (ran + 1, err_step.get("error"))
        out["refusedRule"] = err_step.get("refusedRule")
    if restore or err_step:
        now = pivots()
        back = all(same_pivot(n, b) for n, b in zip(now, before))
        out["restoredByCancel"] = back
        out["restoreMethod"] = "cancel" if back else None
        if not back:
            fixed = []
            for pp, b, n in zip(pl, before, now):
                if same_pivot(n, b):
                    continue
                # The whole pivot from before the steps: a part that tipped over is put back upright, not on its side.
                r1 = _call(pp, "PivotTo", [{"position": list(b["position"]), "rotation": list(b["rotation"])}])
                r2 = _T().rbx_set_props(pp, {"AssemblyLinearVelocity": [0, 0, 0], "AssemblyAngularVelocity": [0, 0, 0]})
                fixed.append({"part": _join(pp), "pivotTo": r1.get("verified"), "velocityZeroed": r2.get("verified")})
            now = pivots()
            back = all(same_pivot(n, b) for n, b in zip(now, before))
            out.update(restoreMethod="pivot" if back else None, pivotRestore=fixed)
        out["restored"] = back
        if not back:
            out.update(ok=False, error=(out.get("error") or "") + "; the parts are NOT back where they were")
    return out


_AGENT_KEYS = {"AgentRadius": "num", "AgentHeight": "num", "AgentCanJump": "bool", "AgentCanClimb": "bool",
               "WaypointSpacing": "num", "Costs": "costs"}


# tostring(PathWaypoint) on 0.748, measured live: "{10, 0, 0}, Walk, Plastic" - position, action, label.
_WP_TEXT = re.compile(r"^\{\s*([-+0-9.eE]+),\s*([-+0-9.eE]+),\s*([-+0-9.eE]+)\s*\},\s*(\w+),\s*(.*)$")


def _waypoint(w):
    """A PathWaypoint: {position, action, label} when the plugin encodes the type (requested
    of the plugin); until then parsed from its tostring text in the one form 0.748 was
    measured to write, and anything else kept raw and said so."""
    if isinstance(w, dict) and w.get("$t") == "PathWaypoint" and "position" in w:
        return {"position": _xyz(V.decode(w.get("position"))), "action": V.decode(w.get("action")),
                "label": w.get("label")}, True
    if isinstance(w, dict) and w.get("$t") == "PathWaypoint" and isinstance(w.get("str"), str):
        m = _WP_TEXT.match(w["str"].strip())
        if m:
            return {"position": [float(m.group(i)) for i in (1, 2, 3)], "action": m.group(4), "label": m.group(5),
                    "from": "text"}, True
    return {"raw": V.decode(w)}, False


@_tool
def rbx_path(start: list, finish: list, agent: dict = None) -> dict:
    "Roblox Studio: compute a navigation path and read its waypoints: PathfindingService:CreatePath with agent parameters (AgentRadius, AgentHeight, AgentCanJump, AgentCanClimb, WaypointSpacing, Costs {label: cost}), Path:ComputeAsync from start to finish, Path.Status, Path:GetWaypoints. A path that cannot be found answers ok true, reachable false with the status - never an error. The Path object lives outside the place and is released after. Read only for the place."
    if not _vec(start) or not _vec(finish):
        return _no("start and finish are [x, y, z]")
    args = []
    if agent is not None:
        if not isinstance(agent, dict):
            return _no("agent is an object")
        for k, v in agent.items():
            kind = _AGENT_KEYS.get(k)
            ok = (kind == "num" and _num(v)) or (kind == "bool" and isinstance(v, bool)) or (
                kind == "costs" and isinstance(v, dict) and all(isinstance(a, str) and (_num(b) or b in ("inf", math.inf))
                                                                 for a, b in v.items()))
            if not ok:
                return _no("agent %s: keys are %s (Costs an object of label to number, or \"inf\" for impassable)"
                           % (k, ", ".join(_AGENT_KEYS)))
        if isinstance(agent.get("Costs"), dict):
            # "inf" is math.huge in Studio, the only cost that makes a label impassable (RB410); plugin 0.15.0 decodes it
            agent = dict(agent, Costs={a: (math.inf if b == "inf" else b) for a, b in agent["Costs"].items()})
        args = [agent]
    # CreatePath makes a Path outside the place (a handle), as TweenService:Create makes a Tween: sent as a read, so it is
    # not recorded and the scratch gate does not judge PathfindingService's own path. Raw, after the checks above.
    wire = []
    if args:
        w, why = V.encode("Dictionary", args[0])
        if why:
            return _no(why, refusedRule="type")
        wire = [w]
    made = _rbx("call", target=["PathfindingService"], method="CreatePath", args=wire, argc=len(wire), readOnly=True,
                yields=False, callTimeout=30, timeout=45)
    ret = V.decode((made.get("returns") or [None])[0]) if made.get("ok") else None
    h = ret.get("handle") if isinstance(ret, dict) else None
    if not h:
        return dict(made, ok=False, step="CreatePath", error=made.get("error") or "CreatePath returned %r" % (ret,))
    T = _T()
    try:
        comp = _call({"handle": h}, "ComputeAsync", [list(start), list(finish)], timeout=60)
        if not comp.get("ok"):
            return dict(comp, step="ComputeAsync")
        st, err = _read({"handle": h}, ["Status"])
        if err:
            return dict(err, step="Status")
        status = st.get("Status")
        raw = _rbx("call", target={"handle": h}, method="GetWaypoints", args=[], argc=0, readOnly=True, yields=False,
                   callTimeout=30, timeout=45)
    finally:
        rel = T.rbx_release_handle(h)
    if not raw.get("ok"):
        return dict(raw, step="GetWaypoints")
    arr = (raw.get("returns") or [{}])[0]
    items = arr.get("v") if isinstance(arr, dict) else (arr or [])
    wps, decoded = [], True
    for w in items or []:
        d, good = _waypoint(w)
        wps.append(d)
        decoded = decoded and good
    out = {"ok": True, "status": status, "reachable": status == "Success",
           "partial": status in ("ClosestNoPath", "ClosestOutOfRange"), "waypoints": wps, "count": len(wps),
           "waypointsDecoded": decoded, "released": bool(rel.get("ok")), "start": list(start), "finish": list(finish)}
    if decoded and len(wps) > 1:
        out["length"] = round(sum(math.sqrt(_dot(_sub(b["position"], a["position"]), _sub(b["position"], a["position"])))
                                  for a, b in zip(wps, wps[1:])), 3)
        out["actions"] = sorted({w["action"] for w in wps if w.get("action")})
    if not decoded:
        out["note"] = ("a waypoint came back in a form this does not read: each is kept as it came; a plugin request "
                       "(filed) would encode PathWaypoint as a type")
    return out


@_tool
def rbx_path_modifier(target: Any, label: str, pass_through: bool = False, name: str = None) -> dict:
    "Roblox Studio: mark a part for pathfinding with a PathfindingModifier (Label for an agent's Costs, PassThrough true to let paths cross it), made under target. Read back: Label and PassThrough. rbx_path with Costs {label: math.huge-like big number} then avoids it."
    if not isinstance(label, str) or not label:
        return _no("label is a non-empty string")
    if not isinstance(pass_through, bool):
        return _no("pass_through is true or false")
    tparts, err = _parts(target)
    if err:
        return err
    return _T().rbx_create("PathfindingModifier", tparts, name=name or _free_name(tparts, "PathfindingModifier"),
                           props={"Label": label, "PassThrough": pass_through})


@_tool
def rbx_path_link(part0: Any, point0: list, part1: Any, point1: list, label: str = None, bidirectional: bool = True,
                  name: str = None, parent: Any = None) -> dict:
    "Roblox Studio: a PathfindingLink between two world points (a jump, a ladder, a teleport) with an Attachment made in part0 and part1 at each point, in one recording; Label names it for an agent's Costs. Read back: each attachment's WorldCFrame, Attachment0, Attachment1, Label and IsBidirectional."
    p0, err = _parts(part0, "part0")
    if err:
        return err
    p1, err = _parts(part1, "part1")
    if err:
        return err
    if not _vec(point0) or not _vec(point1):
        return _no("point0 and point1 are [x, y, z]")
    if label is not None and not isinstance(label, str):
        return _no("label is a string")
    if not isinstance(bidirectional, bool):
        return _no("bidirectional is true or false")
    par = p0 if parent is None else V.split_path(parent)
    base = name or "PathfindingLink"
    with _Group("MifBridge path link") as g:
        if not g.open:
            return g.refused()
        a0, ap0 = _attachment(p0, point0, None, _free_name(p0, base + "_A0"))
        a1, ap1 = _attachment(p1, point1, None, _free_name(p1, base + "_A1")) if a0.get("ok") else ({"ok": False}, None)
        if not (a0.get("ok") and a1.get("ok")):
            return {"ok": False, "verified": False, "cancelled": g.cancel(), "attachment0": a0, "attachment1": a1,
                    "error": a0.get("error") or a1.get("error")}
        props = {"Attachment0": _join(ap0), "Attachment1": _join(ap1), "IsBidirectional": bidirectional}
        if label is not None:
            props["Label"] = label
        mk = _T().rbx_create("PathfindingLink", par, name=_free_name(par, base), props=props)
        if mk.get("ok"):
            g.end(True)
            cancelled = False
        else:
            cancelled = g.cancel()
    return dict(mk, attachment0=a0, attachment1=a1, cancelled=cancelled)


@_tool
def rbx_character_controller(model: Any, root_part: Any = None, move_speed: float = 16, climb: bool = False,
                             swim: bool = False, name: str = "ControllerManager", prove_walk: bool = False,
                             walk_seconds: float = 2, search_distance: float = 5) -> dict:
    "Roblox Studio: a physics character controller on a model, in one recording: a ControllerManager (BaseMoveSpeed move_speed) with a GroundController and an AirController (and ClimbController / SwimController when asked) under it, a ControllerPartSensor in the root part as its GroundSensor (SensorMode Floor; a second, Ladder, as ClimbSensor with climb), each with SearchDistance search_distance (a new sensor's is 0, which senses nothing and leaves every controller inactive - measured on 0.748), and RootPart, GroundSensor and ActiveController set. root_part defaults to the model's PrimaryPart. Read back: every create and every reference. prove_walk sets MovingDirection to +X, runs a play probe sampling the root part's X for walk_seconds, puts MovingDirection back, and says whether it walked."
    cat, err = _cat()
    if err:
        return err
    mparts, err = _parts(model, "model")
    if err:
        return err
    if not _num(move_speed) or move_speed < 0:
        return _no("move_speed is a number of studs a second, 0 or more")
    if not _num(search_distance) or not 0 < search_distance <= 100:
        return _no("search_distance is above 0 and at most 100 studs")
    if root_part is None:
        vals, err = _read(mparts, ["PrimaryPart"])
        if err:
            return err
        root_part = vals.get("PrimaryPart")
        if not root_part:
            return _no("the model has no PrimaryPart: give root_part")
    rparts, err = _parts(root_part, "root_part")
    if err:
        return err
    rc, err = _class_of(rparts)
    if err:
        return err
    if not cat.is_a(rc, "BasePart"):
        return _no("%s is a %s, not a part" % (_join(rparts), rc), refusedRule="wrong-class")
    if _exists(mparts + [name]):
        return _no("%s already has a %s" % (_join(mparts), name), refusedRule="exists")
    T = _T()
    mgr = mparts + [name]
    made = []
    with _Group("MifBridge character controller") as g:
        if not g.open:
            return dict({"manager": _join(mgr), "made": made}, **g.refused())

        def mk(cls, parent, nm, props=None):
            r = T.rbx_create(cls, parent, name=nm, props=props)
            made.append({"path": r.get("path"), "className": cls, "ok": r.get("ok"), "error": r.get("error")})
            return r.get("ok")
        good = mk("ControllerManager", mparts, name, {"BaseMoveSpeed": move_speed})
        good = good and mk("GroundController", mgr, "GroundController")
        good = good and mk("AirController", mgr, "AirController")
        if climb:
            good = good and mk("ClimbController", mgr, "ClimbController")
        if swim:
            good = good and mk("SwimController", mgr, "SwimController")
        gs = _free_name(rparts, "GroundSensor")
        good = good and mk("ControllerPartSensor", rparts, gs, {"SensorMode": "Floor", "SearchDistance": search_distance})
        cs = None
        if climb:
            cs = _free_name(rparts, "ClimbSensor", taken=[gs])
            good = good and mk("ControllerPartSensor", rparts, cs, {"SensorMode": "Ladder", "SearchDistance": search_distance})
        refs = {"RootPart": _join(rparts), "GroundSensor": _join(rparts + [gs]),
                "ActiveController": _join(mgr + ["GroundController"])}
        if cs:
            refs["ClimbSensor"] = _join(rparts + [cs])
        st = T.rbx_set_props(mgr, refs) if good else {"ok": False, "error": "a create failed"}
        if good and st.get("ok"):
            g.end(True)
            cancelled = False
        else:
            cancelled = g.cancel()
    out = {"manager": _join(mgr), "made": made, "references": {k: st.get(k) for k in ("ok", "verified", "after", "mismatch", "error")}}
    if not (good and st.get("ok")):
        return dict(out, ok=False, verified=False, cancelled=cancelled,
                    error=st.get("error") or next((m["error"] for m in made if not m["ok"]), "a create failed"))
    out.update(ok=True, verified=True)
    if prove_walk:
        if not _num(walk_seconds) or not 0.5 <= walk_seconds <= 20:
            return dict(out, ok=False, error="walk_seconds is 0.5-20")
        d = T.rbx_set_props(mgr, {"MovingDirection": [1, 0, 0]})
        probe = T.rbx_play_probe("sample", target=_join(rparts), property="Position", axis="X", seconds=walk_seconds,
                                 interval=0.1, settle=0.5, timeout=walk_seconds + 60)
        back = T.rbx_set_props(mgr, {"MovingDirection": [0, 0, 0]})
        xs = [s.get("v") for s in probe.get("samples") or [] if _num(s.get("v"))]
        moved = round(xs[-1] - xs[0], 3) if len(xs) > 1 else None
        out["walk"] = {"directionSet": d.get("verified"), "directionPutBack": back.get("verified"), "samples": len(xs),
                       "movedX": moved, "walked": bool(moved is not None and moved > 1.0), "probeOk": probe.get("ok"),
                       "error": probe.get("error")}
        if not out["walk"]["walked"]:
            out.update(ok=False, error="the controller was built, and in play the root part moved %s studs along X" % moved)
    return out


_PHYSICS_ROUTES = (["@settings", "PhysicsSettings"], ["PhysicsSettings"])


@_tool
def rbx_physics_settings_get(names: list = None) -> dict:
    "Roblox Studio: Studio's physics settings, read (PhysicsSettings, PluginSecurity: AllowSleep, AreContactPointsShown, PhysicsEnvironmentalThrottle and the other display switches; default every one the catalog counts). They are Studio preferences, not the place: the bridge does not write them (studio-preference). Read only."
    cat, err = _cat()
    if err:
        return err
    allp = sorted(n for n, r in cat.classes["PhysicsSettings"]["m"].items() if r[0] == "P" and C.is_counted(r))
    if names is None:
        names = allp
    if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
        return _no("names is a list of PhysicsSettings property names")
    for n in names:
        bad = cat.read_refusal("PhysicsSettings", n, kind="P")
        if bad:
            return bad
    tried = []
    for route in _PHYSICS_ROUTES:
        r = _rbx("resolve", target=route)
        tried.append({"route": route, "className": r.get("className"), "error": r.get("error")})
        if r.get("ok") and r.get("className") == "PhysicsSettings":
            g = _rbx("get_props", target=route, props=names)
            if not g.get("ok"):
                return dict(g, route=route)
            vals = {k: V.decode(v) for k, v in T_obj(g.get("values")).items()}
            return {"ok": True, "route": route, "values": vals, "count": len(vals)}
    return _no("this plugin reaches no PhysicsSettings: game:GetService does not make it and the plugin has no settings() "
               "root yet (requested of the plugin)", refusedRule="needs-plugin",
               tried=tried)


# ================================================================ batch 12: camera and input
_SAVED = {}     # the camera as it was before this bridge first moved it, until restore


def _camera():
    vals, err = _read("Workspace", ["CurrentCamera"])
    if err:
        return None, err
    cam = vals.get("CurrentCamera")
    if not cam:
        return None, _no("Workspace.CurrentCamera is nil", refusedRule="not-found")
    return cam, None


_CAM_PROPS = ["CFrame", "Focus", "FieldOfView", "ViewportSize", "CameraType"]


def _bounds(parts):
    """(cframe, size) of a model's bounding box or a part's own box."""
    cls, err = _class_of(parts)
    if err:
        return None, None, err
    cat, _e = _cat()
    if cat.is_a(cls, "BasePart"):
        vals, err = _read(parts, ["CFrame", "Size"])
        if err:
            return None, None, err
        return vals.get("CFrame"), _xyz(vals.get("Size")), None
    if cat.is_a(cls, "Model"):
        r = _call(parts, "GetBoundingBox")
        v = r.get("values") or []
        if not r.get("ok") or len(v) < 2:
            return None, None, r
        return v[0], _xyz(v[1]), None
    return None, None, _no("%s is a %s: frame a part or a model" % (_join(parts), cls), refusedRule="wrong-class")


def _camera_set(cam, props):
    """Move the editor camera. Plugin 0.15.0's camera op sets it with no undo recording and no mode gate (it is the
    person's view, not place content: lane 4's L4-6) and reads it back; an older plugin answers unknown-op and the move
    goes through set_props (an undo step, full mode)."""
    cat, err = _cat()
    if err:
        return err
    wire = {}
    for key, prop in (("cframe", "CFrame"), ("focus", "Focus")):
        if props.get(prop) is not None:
            w, why = V.encode("CFrame", props[prop], cat.enums)
            if why:
                return _no(why, refusedRule="type")
            wire[key] = w
    if props.get("FieldOfView") is not None:
        wire["fieldOfView"] = props["FieldOfView"]
    r = _T()._rbx("camera", set=wire)
    if r.get("refusedRule") == "unknown-op":
        return dict(_T().rbx_set_props(cam, props), route="set_props")
    if not r.get("ok"):
        return r
    after = r.get("after") or {}
    problems = []
    for key, prop in (("cframe", "CFrame"), ("focus", "Focus")):
        if key in wire and not V.same(wire[key], after.get(key)):
            problems.append("%s reads %s" % (prop, V.decode(after.get(key))))
    if "fieldOfView" in wire and abs((after.get("fieldOfView") or 0) - wire["fieldOfView"]) > 1e-3:
        problems.append("FieldOfView reads %s" % after.get("fieldOfView"))
    out = {"ok": not problems, "verified": not problems, "route": "camera", "recorded": r.get("recorded"),
           "after": {k: V.decode(v) for k, v in after.items()}, "requestId": r.get("requestId"),
           "canUndo": [r.get("canUndoBefore"), r.get("canUndoAfter")]}
    if problems:
        out["error"] = "; ".join(problems)
    return out


@_tool
def rbx_camera(action: str = "get", position: list = None, look_at: list = None, fov: float = None, target: Any = None) -> dict:
    "Roblox Studio: the editor camera (Workspace.CurrentCamera). get reads CFrame, Focus, FieldOfView, ViewportSize and CameraType; set moves it to position looking at look_at (and fov); frame fits target, a part or a model, in view (Camera:ZoomToExtents on its bounding box) and says whether its center is now on screen; restore puts back the camera as it was before the first set or frame. Read back: Camera.CFrame after each move. The camera is not place content: with plugin 0.15.0 set and restore go through its camera op - no undo step, any mode - and are read back; an older plugin writes them through set_props (an undo step, full mode). frame calls Camera:ZoomToExtents, a write the gate judges."
    if action not in ("get", "set", "frame", "restore"):
        return _no("action is get, set, frame or restore")
    cam, err = _camera()
    if err:
        return err
    now, err = _read(cam, _CAM_PROPS)
    if err:
        return err
    if action == "get":
        return {"ok": True, "camera": cam, "values": now, "saved": bool(_SAVED.get(cam))}
    if action == "restore":
        saved = _SAVED.get(cam)
        if not saved:
            return _no("nothing to restore: this bridge has not moved the camera since it started", refusedRule="nothing-saved")
        st = _camera_set(cam, saved)
        if st.get("ok"):
            _SAVED.pop(cam, None)
        return dict(st, camera=cam, restored=bool(st.get("ok")))
    _SAVED.setdefault(cam, {"CFrame": now.get("CFrame"), "Focus": now.get("Focus"), "FieldOfView": now.get("FieldOfView")})
    if action == "set":
        if not _vec(position) or not _vec(look_at):
            return _no("set needs position and look_at, each [x, y, z]")
        cf = _look_at(position, look_at)
        if cf is None:
            return _no("position and look_at are the same point")
        props = {"CFrame": cf, "Focus": {"position": list(look_at)}}
        if fov is not None:
            if not _num(fov) or not 1 <= fov <= 120:
                return _no("fov is 1-120 degrees")
            props["FieldOfView"] = fov
        st = _camera_set(cam, props)
        return dict(st, camera=cam, savedForRestore=True)
    # frame
    tparts, err = _parts(target, "target")
    if err:
        return err
    cf, size, err = _bounds(tparts)
    if err:
        return err
    if not cf or not size:
        return _no("no bounding box for %s" % _join(tparts))
    z = _call(cam, "ZoomToExtents", [cf, size])
    if not z.get("ok"):
        return dict(z, step="ZoomToExtents")
    after, err = _read(cam, ["CFrame"])
    if err:
        return err
    center = cf["position"]
    proj = _call(cam, "WorldToViewportPoint", [center])
    vals = proj.get("values") or []
    on = vals[1] if len(vals) > 1 else None
    moved = after.get("CFrame") != now.get("CFrame")
    look = None
    if isinstance(after.get("CFrame"), dict):
        R = after["CFrame"]["rotation"]
        look = [-R[2], -R[5], -R[8]]
    ahead = bool(look) and _dot(look, _sub(center, after["CFrame"]["position"])) > 0
    ok = bool(on) and ahead
    out = {"ok": ok, "verified": ok, "camera": cam, "target": _join(tparts), "before": now.get("CFrame"),
           "after": after.get("CFrame"), "moved": moved, "targetCenterOnScreen": on, "targetAhead": ahead,
           "viewportPoint": _xyz(vals[0]) if vals else None, "savedForRestore": True}
    if not ok:
        out["error"] = "after ZoomToExtents the target's center is %s" % ("behind the camera" if not ahead else "off screen")
    return out


INPUT_BASES = ("InputContext", "InputAction", "InputBinding")


@_tool
def rbx_input_actions(spec: dict, parent: Any = "Workspace.MifScratch", replace: bool = False) -> dict:
    "Roblox Studio: the input action system from one description, in one recording: an InputContext {name, props (Priority, Sink, Enabled), actions: [{name, props (Type, Enabled), bindings: [{name?, props (KeyCode, Up, Down, Left, Right, Deadzone, PressedThreshold, Scale...)}]}]} under parent. Property names are the catalog's own; each is typed by it. Read back: every property sent and every instance by path. Where a context must live to act in a running game is measured by rbx_input_fire, not assumed here."
    if not isinstance(spec, dict):
        return _no("spec is {name, props, actions: [{name, props, bindings: [{name?, props}]}]}")
    extra = sorted(set(spec) - {"name", "props", "actions"})
    if extra:
        return _no("unknown keys %s in the context" % extra)

    def binding(b, i):
        if not isinstance(b, dict) or set(b) - {"name", "props"}:
            return {"class": None}
        return {"class": "InputBinding", "name": b.get("name", "Binding%d" % (i + 1)), "props": b.get("props")}

    actions = []
    for a in spec.get("actions") or []:
        if not isinstance(a, dict) or set(a) - {"name", "props", "bindings"}:
            return _no("an action is {name, props, bindings}")
        actions.append({"class": "InputAction", "name": a.get("name", "InputAction"), "props": a.get("props"),
                        "children": [binding(b, i) for i, b in enumerate(a.get("bindings") or [])]})
    node = {"class": "InputContext", "name": spec.get("name", "InputContext"), "props": spec.get("props"),
            "children": actions}
    return _build_tree(node, parent, INPUT_BASES, "input element", replace)


@_tool
def rbx_input_fire(binding: Any, value: Any = True, release: bool = True, action: Any = None, play: bool = False,
                   sample: Any = "@character", property: str = "Position", axis: str = "Y", seconds: float = 2,  # noqa: A002
                   on_client: bool = True) -> dict:
    "Roblox Studio: fire an InputBinding whose Type is Scriptable (Studio refuses Fire on any other) (InputBinding:Fire, the one input route a plugin can reach: VirtualInputManager is RobloxScriptSecurity and VirtualUser LocalUserSecurity) and measure whether it drives its InputAction: the action's Pressed and StateChanged watched while firing, InputAction:GetState before, after the fire and after the release (Fire with false when value is true). This runs in edit mode; in a running game the built-in Studio MCP server's user_keyboard_input is the route for real key presses. Answers drivesAction from what fired, never assumed. play true (plugin 0.15.0) fires it inside a play test instead: on_client true (default) has the client fire it, where input acts, and report the action's state before and at the fire and the character's rise (HumanoidRootPart height over seconds); on_client false fires it in the play server and samples sample (a path, or @character) by property and axis."
    bparts, err = _parts(binding, "binding")
    if err:
        return err
    cls, err = _class_of(bparts)
    if err:
        return err
    if cls != "InputBinding":
        return _no("%s is a %s, not an InputBinding" % (_join(bparts), cls), refusedRule="wrong-class")
    aparts = bparts[:-1] if action is None else V.split_path(action)
    acls, err = _class_of(aparts)
    if err:
        return err
    if acls != "InputAction":
        return _no("%s is a %s; give action, the InputAction this binding drives" % (_join(aparts), acls),
                   refusedRule="wrong-class")
    if not isinstance(release, bool):
        return _no("release is true or false")
    # Live 0.748: "InputBinding:Fire() can only be called when Type is Scriptable" - so the binding is read first.
    bt, err = _read(bparts, ["Type"])
    if err:
        return err
    if bt.get("Type") != "Scriptable":
        return _no("%s has Type %s, and Studio fires only a Scriptable binding: add an InputBinding with Type Scriptable "
                   "to the action (rbx_create) and fire that one" % (_join(bparts), bt.get("Type")),
                   refusedRule="not-scriptable")
    T = _T()
    if play:
        # Plugin 0.15.0's play-probe kind fire: the binding fired inside a running game, a property sampled meanwhile
        # (default the first player's HumanoidRootPart height), the action's state and events reported.
        p = T.rbx_play_probe("fire", target=bparts, value=value, target2=sample, property=property, axis=axis,
                             seconds=seconds, interval=0.05, timeout=seconds + 90, clients=bool(on_client))
        if not p.get("ok"):
            return p
        samples = [s.get("v") for s in p.get("samples") or [] if isinstance(s.get("v"), (int, float))]
        client = next((c.get("acted") for c in p.get("clients") or [] if isinstance(c.get("acted"), dict)), None)
        base = p.get("baseline")
        fired = (client or {}).get("fired") if on_client else p.get("fired")
        out = {"ok": bool(fired), "mode": "play", "firedBy": "client" if on_client else "server", "client": client,
               "binding": _join(bparts), "fired": fired,
               "fireError": p.get("fireError"), "stateBefore": p.get("stateBefore"), "stateAtFire": p.get("stateFired"),
               "stateAfter": p.get("stateAfter"), "events": p.get("events"), "player": p.get("player"),
               "sampled": {"target": sample, "property": property, "axis": axis, "baseline": base,
                           "max": max(samples) if samples else None, "min": min(samples) if samples else None,
                           "count": len(samples)},
               "errors": p.get("errors"), "testProcesses": p.get("testProcesses")}
        if samples and isinstance(base, (int, float)):
            out["sampled"]["rise"] = round(max(samples) - base, 4)
        if on_client:
            out["drivesAction"] = bool(client) and client.get("stateAtFire") != client.get("stateBefore")
        else:
            out["drivesAction"] = bool(p.get("events")) or p.get("stateFired") != p.get("stateBefore")
        if not out["ok"]:
            out["error"] = (client or {}).get("fireError") or p.get("fireError") or (
                "no client report came back" if on_client and not client else "the binding was not fired")
        return out
    s0 = _call(aparts, "GetState")
    watches = {}
    for ev in ("Pressed", "StateChanged", "Released"):
        w = T.rbx_watch("start", target=_join(aparts), event=ev, seconds=10, max_events=100)
        if w.get("ok"):
            watches[ev] = w.get("watchId")
    f1 = _call(bparts, "Fire", [value])
    time.sleep(0.2)
    s1 = _call(aparts, "GetState")
    f2 = s2 = None
    if release and value is True:
        f2 = _call(bparts, "Fire", [False])
        time.sleep(0.2)
        s2 = _call(aparts, "GetState")
    fired = {}
    for ev, wid in watches.items():
        r = T.rbx_watch("stop", watch_id=wid)
        fired[ev] = [e.get("args") for e in r.get("events") or []] if r.get("ok") else {"error": r.get("error")}
    out = {"binding": _join(bparts), "action": _join(aparts), "fire": {"ok": f1.get("ok"), "error": f1.get("error")},
           "stateBefore": s0.get("value"), "stateAfterFire": s1.get("value"), "events": fired,
           "watched": sorted(watches)}
    if f2 is not None:
        out.update(release={"ok": f2.get("ok"), "error": f2.get("error")}, stateAfterRelease=(s2 or {}).get("value"))
    if not f1.get("ok"):
        return dict(out, ok=False, error=f1.get("error"), refusedRule=f1.get("refusedRule"))
    pressed = isinstance(fired.get("Pressed"), list) and len(fired["Pressed"]) > 0
    changed = isinstance(fired.get("StateChanged"), list) and len(fired["StateChanged"]) > 0
    out.update(ok=True, drivesAction=bool(pressed or changed or out["stateAfterFire"] != out["stateBefore"]),
               mode="edit")
    if not out["drivesAction"]:
        out["note"] = ("InputBinding:Fire changed nothing the action shows in edit mode: an action may act only in a "
                       "running client; the built-in Studio MCP server's user_keyboard_input drives real keys there")
    return out


_INTERACTION = {"prompt": "ProximityPrompt", "click": "ClickDetector", "drag": "DragDetector"}


@_tool
def rbx_interactions(kind: str, parent: Any, props: dict = None, name: str = None, trigger: bool = False,
                     play: bool = False, on_client: bool = True) -> dict:
    "Roblox Studio: an interaction object on a part: a ProximityPrompt (kind prompt: ActionText, ObjectText, HoldDuration, KeyboardKeyCode, MaxActivationDistance...), ClickDetector (click: MaxActivationDistance) or DragDetector (drag: DragStyle, ResponseStyle, Min/MaxDragTranslation...), under parent with props. Read back: every property sent. trigger true on a prompt holds it (ProximityPrompt:InputHoldBegin, the hold duration, InputHoldEnd) with Triggered and TriggerEnded watched and says how often each fired - measured in edit mode, where no player is present. play true (plugin 0.15.0) holds it inside a play test instead: on_client true (default) has the client hold it once its character has landed, and the server counts PromptButtonHoldBegan, PromptButtonHoldEnded, Triggered and TriggerEnded with the player each passed."
    cls = _INTERACTION.get(kind)
    if cls is None:
        return _no("kind is prompt, click or drag")
    pparts, err = _parts(parent, "parent")
    if err:
        return err
    if not isinstance(trigger, bool) or (trigger and kind != "prompt"):
        return _no("trigger is true only for a prompt")
    want_hold = (props or {}).get("HoldDuration") if isinstance(props, dict) else None
    if trigger and not play and _num(want_hold) and want_hold > MAX_EDIT_HOLD:
        return _no("HoldDuration %g is above the %g s an edit-mode trigger waits out (its event watches last 30 s): "
                   "nothing was made. Trigger it with play true, or hold it for at most %g s" % (
                       want_hold, MAX_EDIT_HOLD, MAX_EDIT_HOLD), refusedRule="hold-too-long")
    T = _T()
    nm = name or _free_name(pparts, cls)
    mk = T.rbx_create(cls, pparts, name=nm, props=props)
    if not mk.get("ok") or not trigger:
        return mk
    path = mk["path"]
    if play:
        # Plugin 0.15.0's play-probe kind prompt: held in a running game, its events counted with the player passed.
        p = T.rbx_play_probe("prompt", target=path, seconds=1, timeout=120, clients=bool(on_client))
        pr = p.get("prompt") or {}
        client = next((c.get("acted") for c in p.get("clients") or [] if isinstance(c.get("acted"), dict)), None)
        counts = pr.get("counts") or {}
        tr = {"mode": "play", "ok": p.get("ok"), "heldBy": pr.get("heldBy"), "client": client,
              "holdBegin": (client or pr).get("holdBegin"), "holdEnd": (client or pr).get("holdEnd"), "holdSeconds": pr.get("holdDuration"), "fired": counts,
              "triggered": counts.get("Triggered"), "events": pr.get("events"),
              "error": p.get("error"), "testProcesses": p.get("testProcesses")}
        return _triggered(mk, tr, p.get("ok"))
    vals, _e = _read(path, ["HoldDuration", "Enabled"])
    hold = (vals or {}).get("HoldDuration")
    hold = hold if _num(hold) and hold > 0 else 0
    if hold > MAX_EDIT_HOLD:
        return dict(mk, ok=False, verified=False, refusedRule="hold-too-long",
                    trigger={"mode": "edit", "ran": False, "holdSeconds": hold},
                    error="the prompt was made at %s, but its HoldDuration %g is above the %g s an edit-mode trigger waits "
                          "out, so it was not held" % (_show(path), hold, MAX_EDIT_HOLD))
    watches = {}
    for ev in ("Triggered", "TriggerEnded", "PromptButtonHoldBegan"):
        w = T.rbx_watch("start", target=path, event=ev, seconds=min(30, hold + 5), max_events=100)
        if w.get("ok"):
            watches[ev] = w.get("watchId")
    b = _call(path, "InputHoldBegin")
    time.sleep(hold + 0.2)        # the whole hold: at most MAX_EDIT_HOLD, inside every watch's window
    e = _call(path, "InputHoldEnd")
    time.sleep(0.2)
    fired = {}
    for ev, wid in watches.items():
        r = T.rbx_watch("stop", watch_id=wid)
        fired[ev] = len(r.get("events") or []) if r.get("ok") else None
    tr = {"holdBegin": b.get("ok"), "holdEnd": e.get("ok"), "holdSeconds": hold, "fired": fired,
          "triggered": fired.get("Triggered"), "mode": "edit", "error": b.get("error") or e.get("error")}
    return _triggered(mk, tr, b.get("ok") and e.get("ok"))


MAX_EDIT_HOLD = 25      # an edit-mode trigger's watches last min(30, hold + 5) s: a longer hold would outlast them


def _triggered(mk, tr, ran):
    """The create's answer with the trigger's: ok needs the hold to have run and Triggered to have fired at least once."""
    n = tr.get("triggered")
    if ran and isinstance(n, int) and not isinstance(n, bool) and n >= 1:
        return dict(mk, trigger=tr)
    why = (tr.get("error") or "the trigger did not run") if not ran else (
        "Triggered was not watched" if n is None else "Triggered fired %s time(s)" % n)
    return dict(mk, ok=False, verified=False, trigger=tr,
                error="the prompt was made at %s, but holding it did not trigger it: %s" % (_show(mk.get("path")), why))


@_tool
def rbx_screen_space(points: list = None, pixels: list = None, targets: list = None, screenshot: bool = False) -> dict:
    "Roblox Studio: world and screen through the editor camera. points [[x,y,z], ...] project to the viewport (Camera:WorldToViewportPoint: pixel x, y, depth and whether on screen); pixels [[x, y], ...] cast back to rays (Camera:ViewportPointToRay); targets (parts or models) give each one's box on screen from its eight bounding-box corners and its center. screenshot true captures the viewport (rbx_screenshot) and scales each box to the image's pixels, saying whether the center falls inside its box. Read only."
    if not any((points, pixels, targets)):
        return _no("give points, pixels or targets")
    if points is not None and (not isinstance(points, list) or not all(_vec(p) for p in points) or len(points) > 200):
        return _no("points is a list of up to 200 [x, y, z]")
    if pixels is not None and (not isinstance(pixels, list) or not all(_vec(p, 2) for p in pixels) or len(pixels) > 200):
        return _no("pixels is a list of up to 200 [x, y]")
    if targets is not None and (not isinstance(targets, list) or len(targets) > 50):
        return _no("targets is a list of up to 50 paths")
    cam, err = _camera()
    if err:
        return err
    vp, err = _read(cam, ["ViewportSize", "CFrame"])
    if err:
        return err
    size = _xy(vp.get("ViewportSize"))
    out = {"ok": True, "camera": cam, "viewportSize": size, "cameraCFrame": vp.get("CFrame")}

    def project(p):
        r = _call(cam, "WorldToViewportPoint", [list(p)])
        v = r.get("values") or []
        if not r.get("ok") or len(v) < 2:
            return {"point": list(p), "error": r.get("error") or "no projection"}
        x = _xyz(v[0])
        return {"point": list(p), "viewport": x[:2] if x else None, "depth": x[2] if x else None, "onScreen": v[1]}

    if points:
        out["points"] = [project(p) for p in points]
    if pixels:
        rays = []
        for px in pixels:
            r = _call(cam, "ViewportPointToRay", [px[0], px[1]])
            rays.append({"pixel": list(px), "ray": r.get("value"), "error": r.get("error")})
        out["rays"] = rays
    if targets:
        boxes = []
        for t in targets:
            tp, err = _parts(t, "target")
            if err:
                return err
            cf, sz, err = _bounds(tp)
            if err:
                boxes.append({"target": _join(tp), "error": err.get("error")})
                continue
            hs = [s / 2 for s in sz]
            corners = [_cf_point(cf, [sx * hs[0], sy * hs[1], sz_ * hs[2]]) for sx in (-1, 1) for sy in (-1, 1) for sz_ in (-1, 1)]
            pr = [project(c) for c in corners]
            c = project(cf["position"])
            vs = [p["viewport"] for p in pr if p.get("viewport") and (p.get("depth") or 0) > 0]
            box = None
            if vs:
                box = [min(v[0] for v in vs), min(v[1] for v in vs), max(v[0] for v in vs), max(v[1] for v in vs)]
            boxes.append({"target": _join(tp), "center": c, "box": box,
                          "allCornersInFront": len(vs) == 8, "centerInsideBox": bool(box and c.get("viewport") and
                          box[0] <= c["viewport"][0] <= box[2] and box[1] <= c["viewport"][1] <= box[3])})
        out["targets"] = boxes
    if screenshot:
        shot = _T().rbx_screenshot(name="screen_space")
        out["screenshot"] = {k: shot.get(k) for k in ("ok", "file", "width", "height", "originalSize", "error")}
        if shot.get("ok") and size and size[0]:
            sc = shot["width"] / float(size[0])
            out["screenshot"]["scale"] = round(sc, 6)
            for b in out.get("targets") or []:
                if b.get("box"):
                    b["imageBox"] = [round(v * sc, 1) for v in b["box"]]
                    cv = (b.get("center") or {}).get("viewport")
                    b["imageCenter"] = [round(v * sc, 1) for v in cv] if cv else None
                    b["centerInImage"] = bool(cv) and 0 <= cv[0] * sc < shot["width"] and 0 <= cv[1] * sc < shot["height"]
    if targets and not any(b.get("box") for b in out["targets"]):
        out.update(ok=False, error="no target is in front of the camera: nothing projected")
    return out
