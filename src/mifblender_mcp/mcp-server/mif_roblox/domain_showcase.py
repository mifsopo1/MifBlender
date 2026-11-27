"""Showing models: a model cut out on a key colour, and a gallery of a whole set.

    rbx_model_cutout   a copy of a model staged alone in front of a key-colour wall far from the map, the menus hidden,
                       captured by rbx_screenshot and returned as a PNG with alpha (the key keyed out, the spill taken off
                       the edge, one pixel of blur on the alpha). A second capture of the wall alone checks the key: every
                       wall pixel must come out clear.
    rbx_model_gallery  copies of every model of a set laid out in rows on a floor in a scratch area, each labelled, the
                       camera stepped along the rows for numbered edit-mode shots, the gallery removed after.

WHY. Store art is made from the game's own models on a key colour, cut out by hand (L52); a game reviewed a few hundred
primitive-built objects from gallery shots, and rows with no floor under them read as floating and were misjudged (L63).

THE WINDOW. rbx_screenshot draws nothing while Studio's window is minimized (RB1037). When the Studio is the one
rbx_launch started for this daemon's port, its window is restored without activation (user32 ShowWindow
SW_SHOWNOACTIVATE) for the captures and minimized again after; any other Studio's window is never touched, and a blank
frame there is refused with the reason.

Everything is made under Workspace.MifScratch and removed after (rbx_destroy, read back as gone); StarterGui's
ShowDevelopmentGui is put back to what it read (RB1195: false hides the ScreenGuis from the edit viewport).

NEVER RAISE: every failure is {"ok": False, ...}.
"""
import functools
import math
import os
import time
import uuid
from typing import Any

from mif_roblox import values as V

__all__ = []

# Above any map rather than far out along it: Studio keeps positions in 32-bit floats, and at 24,576 studs a pivot reads
# back about 0.002 off (measured live 2026-11-20; rbx_pivot's read-back now allows 8 float steps there, RB1750); under
# 1,024 a float holds 6e-5, so a shot is framed from numbers that are what they say.
STAGE = [600.0, 900.0, 600.0]
KEYS = {"green": (0.0, 1.0, 0.0), "blue": (0.0, 0.0, 1.0)}
KEY_LO, KEY_HI = 0.12, 0.40            # spill (key channel minus the larger other) where alpha starts to fall / reaches 0
MAX_GALLERY = 200
EDGE = 3                                # pixels in from the model's edge where the verify wants full opacity
SHADOW_FIND_LIMIT = 5000                # rbx_find's most; the find has no page, so a larger copy says it was cut


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


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


# ---------------------------------------------------------------- the window
def _own_studio_pid():
    from mif_roblox import launch as L
    try:
        mine = L.ours(_T().PORT)
    except Exception:  # noqa: BLE001 - no record is no window to touch
        return None
    return mine[0]["pid"] if len(mine) == 1 else None


def _windows_of(pid):
    if os.name != "nt":
        return []
    import ctypes
    from ctypes import wintypes
    u = ctypes.windll.user32
    found = []
    proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def cb(hwnd, _l):
        p = wintypes.DWORD()
        u.GetWindowThreadProcessId(hwnd, ctypes.byref(p))
        if p.value == pid and u.IsWindowVisible(hwnd) and u.GetWindowTextLengthW(hwnd) > 0 and not u.GetWindow(hwnd, 4):
            found.append(hwnd)
        return True
    u.EnumWindows(proc(cb), 0)
    return found


class _Window:
    """Restore our own Studio's minimized window for the capture (no activation), minimize it again after."""

    def __init__(self):
        self.pid, self.hwnd, self.restored = _own_studio_pid(), None, False

    def __enter__(self):
        if self.pid is None or os.name != "nt":
            return self
        import ctypes
        u = ctypes.windll.user32
        for h in _windows_of(self.pid):
            if u.IsIconic(h):
                u.ShowWindow(h, 4)            # SW_SHOWNOACTIVATE
                self.hwnd, self.restored = h, True
                time.sleep(1.0)
                break
        return self

    def __exit__(self, *exc):
        if self.restored:
            import ctypes
            ctypes.windll.user32.ShowWindow(self.hwnd, 7)   # SW_SHOWMINNOACTIVE
        return False

    def info(self):
        return {"ownStudioPid": self.pid, "restoredForCapture": self.restored}


def _shot(camera, max_width, name, folder):
    s = _T().rbx_screenshot(camera=camera, max_width=max_width, name=name, folder=folder)
    if not s.get("ok") and "blank" in str(s.get("error")):
        s["error"] = ("%s - Studio draws nothing while its window is minimized (RB1037); restore the Studio window (only "
                      "the Studio rbx_launch started for this port is restored by the bridge)" % s.get("error"))
    return s


# ---------------------------------------------------------------- shared steps
def _box(path):
    """(cframe {position, rotation}, size [x,y,z]) of a Model (GetBoundingBox) or a part (ExtentsCFrame, ExtentsSize)."""
    T = _T()
    cls = T._resolve(V.split_path(path) if isinstance(path, str) else list(path))[0]
    cat, err = T._catalog()
    if err:
        return None, None, err
    if cls == "Model":
        r = T.rbx_call(path, "GetBoundingBox")
        vals = r.get("values") or []
        if not r.get("ok") or len(vals) < 2:
            return None, None, dict(r, ok=False, error=r.get("error") or "GetBoundingBox answered %r" % (vals,))
        cf, sz = vals[0], vals[1]
    elif cls and cat.is_a(cls, "BasePart"):
        g = T.TOOLS["rbx_get_props"](path, ["ExtentsCFrame", "ExtentsSize"])
        vals = g.get("values") or {}
        cf, sz = V.decode(vals.get("ExtentsCFrame")), V.decode(vals.get("ExtentsSize"))
    else:
        return None, None, _no("%s is a %s: give a Model or a part" % (path, cls), refusedRule="wrong-class")
    if isinstance(cf, dict) and "position" in cf and isinstance(sz, dict):
        return cf, [sz["x"], sz["y"], sz["z"]], None
    return None, None, _no("the box of %s read as %r, %r" % (path, cf, sz))


def _scratch():
    T = _T()
    if T.TOOLS["rbx_get_props"]("Workspace.MifScratch", ["Name"]).get("ok"):
        return None
    mk = T.TOOLS["rbx_create"]("Folder", "Workspace", name="MifScratch")
    return None if mk.get("ok") else mk


def _dev_gui(value):
    """Set StarterGui.ShowDevelopmentGui; answers (what it was, failure)."""
    T = _T()
    g = T.TOOLS["rbx_get_props"]("StarterGui", ["ShowDevelopmentGui"])
    was = (g.get("values") or {}).get("ShowDevelopmentGui") if g.get("ok") else None
    if was is None:
        return None, dict(g, ok=False, error="StarterGui.ShowDevelopmentGui could not be read: %s" % g.get("error"))
    if was != value:
        r = T.TOOLS["rbx_set_props"]("StarterGui", {"ShowDevelopmentGui": value})
        if not r.get("ok"):
            return was, r
    return was, None


def _remove(path):
    T = _T()
    r = T.TOOLS["rbx_destroy"](path)
    gone = not T.TOOLS["rbx_get_props"](path, ["Name"]).get("ok")
    return {"path": path, "removed": bool(r.get("ok")) and gone, "error": None if gone else r.get("error")}


PLACE_TOL = 0.01                          # studs: a staged copy only has to stand where the camera looks


def _move(path, pivot):
    """rbx_pivot one target, accepted within PLACE_TOL: a model whose pivot comes through its PrimaryPart's PivotOffset
    reads back a few float steps off (1.8e-4 at y 883, measured live); rbx_pivot allows 8 float steps since RB1750, and
    a staged copy only has to stand where the camera looks."""
    r = _T().TOOLS["rbx_pivot"]([path], pivots=[pivot])
    if not r.get("ok") and isinstance(r.get("maxError"), (int, float)) and r["maxError"] <= PLACE_TOL:
        r = dict(r, ok=True, note="read back %.6f studs off, inside %g" % (r["maxError"], PLACE_TOL))
        r.pop("error", None)
    return r


def _look_of(cf):
    r = cf["rotation"]
    return [-r[2], -r[5], -r[8]]          # LookVector: the -Z column, a model's front


# ---------------------------------------------------------------- keying
def key_out(w, h, rgba, key="green", lo=KEY_LO, hi=KEY_HI):
    """RGBA bytes with the key colour keyed to alpha: spill = key channel minus the larger other channel (0-1); alpha 1
    below lo, 0 above hi, linear between; the spill taken off kept pixels (the key channel clamped to the larger other);
    then a 3x3 box blur on alpha (one pixel)."""
    kc = 1 if key == "green" else 2
    o1, o2 = (0, 2) if kc == 1 else (0, 1)
    n = w * h
    alpha = [0.0] * n
    out = bytearray(rgba)
    for i in range(n):
        p = i * 4
        c = [rgba[p] / 255.0, rgba[p + 1] / 255.0, rgba[p + 2] / 255.0]
        other = max(c[o1], c[o2])
        s = c[kc] - other
        a = 1.0 if s <= lo else 0.0 if s >= hi else (hi - s) / (hi - lo)
        alpha[i] = a
        if c[kc] > other:
            out[p + kc] = int(round(other * 255))
    blurred = [0] * n
    for y in range(h):
        for x in range(w):
            t, k = 0.0, 0
            for yy in (y - 1, y, y + 1):
                if 0 <= yy < h:
                    row = yy * w
                    for xx in (x - 1, x, x + 1):
                        if 0 <= xx < w:
                            t += alpha[row + xx]
                            k += 1
            blurred[y * w + x] = int(round(255 * t / k))
    for i in range(n):
        out[i * 4 + 3] = blurred[i]
    return bytes(out), alpha


@_tool
def rbx_model_cutout(target: Any, key: str = "green", distance: float = None, fov: float = 40, max_width: int = 1280,
                     folder: str = None, name: str = None, verify: bool = True) -> dict:
    "Roblox Studio: a model (or part) shot alone and returned cut out. A copy is staged under Workspace.MifScratch far from the map (the original is not moved), a Neon wall of the key colour (green or blue) set behind it filling the view (the copy's parts cast no shadow on it), StarterGui.ShowDevelopmentGui set false so no ScreenGui shows, and rbx_screenshot taken from in front of the model (its pivot's LookVector, Roblox's front) at distance (default fits the model in fov). The key is keyed out: alpha from the key channel minus the larger other channel, the spill taken off the edge, one pixel of blur; written as a PNG with alpha. verify true also shoots the wall with the copy hidden and checks every wall pixel keys to alpha 0, and that the model's own pixels (where the two shots differ, three pixels in from every edge, less any that still show the key, the wall seen through a gap, and the pixel round it the blur reaches) are opaque. Everything made is removed and ShowDevelopmentGui put back; the Studio rbx_launch started for this port has its minimized window restored without activation for the shots (RB1037), any other Studio's window is never touched. Edit mode."
    T = _T()
    if key not in KEYS:
        return _no("key is green or blue")
    if distance is not None and not (_num(distance) and 1 <= distance <= 5000):
        return _no("distance is 1-5000 studs")
    if not (_num(fov) and 5 <= fov <= 100) or not isinstance(verify, bool):
        return _no("fov is 5-100 degrees; verify true or false")
    if name is not None and not (isinstance(name, str) and name and all(c.isalnum() or c in "-_" for c in name)):
        return _no("name goes in a file name: letters, digits, - and _")
    src = V.split_path(target) if isinstance(target, str) else target
    if not src:
        return _no("target is a path")
    cf0, size0, err = _box(src)
    if err:
        return err
    bad = _scratch()
    if bad:
        return dict(bad, step="Workspace.MifScratch")
    tag = "MifCutout_" + uuid.uuid4().hex[:6]
    made, out, was_gui = [], {"target": V.join_path(src)}, None
    def body():
        nonlocal was_gui
        stage = T.TOOLS["rbx_create"]("Model", "Workspace.MifScratch", name=tag)
        if not stage.get("ok"):
            return dict(stage, step="stage")
        made.append("Workspace.MifScratch." + tag)
        cl = T.TOOLS["rbx_clone"](src, parent="Workspace.MifScratch." + tag, name="Subject")
        if not cl.get("ok"):
            return dict(cl, step="clone")
        subj = "Workspace.MifScratch.%s.Subject" % tag
        # The copy's box centre goes to the stage, its rotation kept: the front stays the model's own.
        pv = T.rbx_call(subj, "GetPivot")
        piv = pv.get("value") if pv.get("ok") else None
        if not isinstance(piv, dict):
            return dict(pv, ok=False, step="GetPivot")
        off = [piv["position"][i] - cf0["position"][i] for i in range(3)]
        moved = _move(subj, {"position": [STAGE[i] + off[i] for i in range(3)], "rotation": piv["rotation"]})
        if not moved.get("ok"):
            return dict(moved, step="stage the copy")
        # No shadow from the copy on the wall: a shadowed key is a darker green the verify would count as the model.
        # Every part, in batches of rbx_batch's 100: a part left casting shadows darkens the key the verify reads as model.
        fd = T.TOOLS["rbx_find"](class_name="BasePart", root=subj, limit=SHADOW_FIND_LIMIT)
        if not fd.get("ok"):
            return dict(fd, ok=False, step="find the copy's parts")
        kids = [k for k in fd.get("matches") or [] if isinstance(k, dict) and k.get("path")]
        if T._resolve(V.split_path(subj))[0] not in (None, "Model"):
            kids = [{"path": subj}] + kids
        ops = [{"tool": "rbx_set_props", "args": {"target": k["path"], "props": {"CastShadow": False}}} for k in kids]
        for i in range(0, len(ops), 100):
            sh = T.TOOLS["rbx_batch"](ops[i:i + 100])
            if not sh.get("ok"):
                return dict(sh, step="CastShadow off on the copy", shadowsOff=i)
        out["shadowsOff"] = len(ops)
        out["shadowsTruncated"] = bool(fd.get("truncated"))
        if fd.get("truncated"):
            out["shadowsNote"] = ("the copy has more than %d parts; one find reaches %d, so the rest may still cast shadows "
                                  "on the wall" % (SHADOW_FIND_LIMIT, SHADOW_FIND_LIMIT))
        centre = list(STAGE)
        look = _look_of(cf0)
        half = max(size0) / 2
        dist = distance or max(1.0, half / math.tan(math.radians(fov) / 2) * 1.15 + half)
        cam_pos = [centre[i] + look[i] * dist for i in range(3)]
        behind = half + max(4.0, half)
        wall_d = dist + behind
        span = 2 * wall_d * math.tan(math.radians(min(100.0, fov * 2.2)) / 2) * 2.2
        rot = cf0["rotation"]
        wall_c = [centre[i] - look[i] * behind for i in range(3)]
        wall = T.TOOLS["rbx_create"]("Part", "Workspace.MifScratch." + tag, name="KeyWall", props={
            "Anchored": True, "CanCollide": False, "CastShadow": False, "Material": "Neon",
            "Color": list(KEYS[key]), "Size": [min(2048, span), min(2048, span), 1],
            "CFrame": {"position": wall_c, "rotation": rot}})
        if not wall.get("ok"):
            return dict(wall, step="key wall")
        was_gui, err = _dev_gui(False)
        if err:
            return dict(err, step="ShowDevelopmentGui")
        camera = {"position": cam_pos, "look_at": centre, "fov": fov}
        fname = name or tag
        with _Window() as win:
            shot = _shot(camera, max_width, fname + "_raw", folder)
            out["window"] = win.info()
            if not shot.get("ok"):
                return dict(shot, ok=False, step="capture")
            wall_shot = None
            if verify:
                now = T.rbx_call(subj, "GetPivot").get("value") or {}
                aside = _move(subj, {"position": [now["position"][0], now["position"][1] + 512, now["position"][2]],
                                     "rotation": now["rotation"]}) if "position" in now else {"ok": False,
                                                                                               "error": "GetPivot read %r" % (now,)}
                if not aside.get("ok"):
                    return dict(aside, step="move the copy out of view")
                wall_shot = _shot(camera, max_width, fname + "_wall", folder)
                if not wall_shot.get("ok") and "blank" not in str(wall_shot.get("error")):
                    return dict(wall_shot, ok=False, step="wall capture")
        from mif_roblox import png as P
        w, h, rgba = P.decode_rgba(open(shot["file"], "rb").read())
        cut, alpha = key_out(w, h, rgba, key)
        folder_ = os.path.dirname(shot["file"])
        path = os.path.join(folder_, "%s_cutout_%s.png" % (fname, time.strftime("%Y%m%d_%H%M%S")))
        with open(path, "wb") as fh:
            fh.write(P.encode_rgba(w, h, cut))
        a_bytes = cut[3::4]
        opaque = [i for i in range(w * h) if a_bytes[i] == 255]
        clear = sum(1 for b in a_bytes if b == 0)
        xs = [i % w for i in range(w * h) if a_bytes[i] > 0]
        ys = [i // w for i in range(w * h) if a_bytes[i] > 0]
        out.update(file=path, raw=shot["file"], width=w, height=h, key=key, camera=camera,
                   pixels={"opaque": len(opaque), "clear": clear, "partial": w * h - len(opaque) - clear},
                   modelRect=[min(xs), min(ys), max(xs) - min(xs) + 1, max(ys) - min(ys) + 1] if xs else None)
        problems = []
        if not opaque:
            problems.append("nothing was left after keying: the model did not show in front of the wall")
        if clear < w * h * 0.05:
            problems.append("under 5% of the frame keyed clear: the wall did not fill the view")
        if verify and wall_shot is not None and wall_shot.get("file"):
            ww, wh, wrgba = P.decode_rgba(open(wall_shot["file"], "rb").read())
            if (ww, wh) != (w, h):
                problems.append("the wall shot is %sx%s, the model shot %sx%s" % (ww, wh, w, h))
            else:
                wcut, _wa = key_out(w, h, wrgba, key)
                wa = wcut[3::4]
                wall_max = max(wa)
                # The model's pixels: where the two shots differ, EDGE pixels in from every edge (a full square
                # erosion), less any pixel that still shows the key colour - the wall seen through a gap in the model.
                diff = [max(abs(rgba[i * 4 + c] - wrgba[i * 4 + c]) for c in range(3)) > 40 for i in range(w * h)]
                rows_ok = [False] * (w * h)
                for y in range(h):
                    run = 0
                    for x in range(w):
                        run = run + 1 if diff[y * w + x] else 0
                        if run >= 2 * EDGE + 1:
                            rows_ok[y * w + x - EDGE] = True
                inner, seen_through = [], 0
                for x in range(w):
                    run = 0
                    for y in range(h):
                        run = run + 1 if rows_ok[y * w + x] else 0
                        if run >= 2 * EDGE + 1:
                            i = (y - EDGE) * w + x
                            p = i * 4
                            kc = 1 if key == "green" else 2
                            other = max(rgba[p + (0 if kc == 1 else 0)], rgba[p + (2 if kc == 1 else 1)])
                            xx, yy = i % w, i // w
                            if (rgba[p + kc] - other) / 255.0 > KEY_LO or any(
                                    alpha[(yy + dy) * w + xx + dx] < 1.0 for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                                    if 0 <= xx + dx < w and 0 <= yy + dy < h):
                                seen_through += 1           # the key itself, or one pixel from it (the alpha blur)
                            else:
                                inner.append(i)
                inner_opaque = sum(1 for i in inner if a_bytes[i] == 255)
                out["check"] = {"wallShot": wall_shot["file"], "wallAlphaMax": wall_max,
                                "modelInnerPixels": len(inner), "modelInnerOpaque": inner_opaque,
                                "keySeenThroughModel": seen_through}
                if wall_max != 0:
                    problems.append("the wall alone keys to alpha up to %d, not 0 everywhere" % wall_max)
                if not inner:
                    problems.append("no pixel told the model from the wall: nothing to check inside")
                elif inner_opaque != len(inner):
                    problems.append("%d of %d pixels inside the model are not opaque" % (len(inner) - inner_opaque, len(inner)))
        out.update(ok=not problems, verified=bool(verify) and not problems)
        if problems:
            out["error"] = "; ".join(problems)
        return out
    res = {}
    try:
        res = body()
    finally:
        if was_gui is not None:
            r = _dev_gui(was_gui)[1]
            out["showDevelopmentGuiRestored"] = r is None
        out["removed"] = [_remove(p) for p in reversed(made)]
    res.update({k: v for k, v in out.items() if k not in res or k in ("removed", "showDevelopmentGuiRestored")})
    if any(not r["removed"] for r in res.get("removed") or []):
        res.update(ok=False, verified=False, error="%s; and the staging was not removed: %s" % (res.get("error"), res["removed"]))
    return res

@_tool
def rbx_model_gallery(models: Any, per_row: int = 6, gap: float = 4, per_shot: int = 3, label: bool = True,
                      fov: float = 50, max_width: int = 1280, folder: str = None, keep: bool = False) -> dict:
    "Roblox Studio: every model of a set laid out for review. models is a list of paths (Models or parts), or one path whose children are the set. Copies (the originals are not moved) go in rows of per_row under Workspace.MifScratch, each seated on one floor part that runs under every row (models with nothing under them read as floating), gap studs apart, facing the camera (their own front, -Z, toward it), each labelled with its name by a BillboardGui. The camera steps along each row per_shot models at a time for numbered edit-mode shots (rbx_screenshot; a blank frame, Studio's window minimized, RB1037, is refused, and the Studio rbx_launch started for this port has its window restored without activation). The gallery is removed after unless keep, its absence read back. Answers each model's slot, each shot's file and which models it frames."
    T = _T()
    if not (isinstance(per_row, int) and 1 <= per_row <= 50 and isinstance(per_shot, int) and 1 <= per_shot <= per_row):
        return _no("per_row is 1-50, per_shot 1-per_row")
    if not (_num(gap) and 0 <= gap <= 200 and _num(fov) and 5 <= fov <= 100):
        return _no("gap is 0-200 studs, fov 5-100 degrees")
    if not isinstance(label, bool) or not isinstance(keep, bool):
        return _no("label and keep are true or false")
    if isinstance(models, str):
        tr = T._rbx("tree", root=V.split_path(models), depth=1, limit=MAX_GALLERY + 1)
        if not tr.get("ok"):
            return tr
        names = [c.get("name") for c in (tr.get("root") or {}).get("children") or []]
        paths = [V.split_path(models) + [n] for n in names]
    elif isinstance(models, list) and all(isinstance(m, (str, list)) for m in models):
        paths = [V.split_path(m) if isinstance(m, str) else list(m) for m in models]
    else:
        return _no("models is a list of paths, or one path whose children are the set")
    if not paths:
        return _no("the set is empty: nothing to lay out")
    if len(paths) > MAX_GALLERY:
        return _no("%d models; at most %d a gallery" % (len(paths), MAX_GALLERY), refusedRule="budget")
    boxes = []
    for p in paths:
        cf, sz, err = _box(p)
        if err:
            return dict(err, model=V.join_path(p))
        boxes.append((cf, sz))
    bad = _scratch()
    if bad:
        return dict(bad, step="Workspace.MifScratch")
    tag = "MifGallery_" + uuid.uuid4().hex[:6]
    root = "Workspace.MifScratch." + tag
    out, made = {"gallery": root, "count": len(paths)}, []
    def body():
        mk = T.TOOLS["rbx_create"]("Model", "Workspace.MifScratch", name=tag)
        if not mk.get("ok"):
            return dict(mk, step="gallery")
        made.append(root)
        cell = max(max(sz[0], sz[2]) for _cf, sz in boxes) + gap
        rows = [list(range(i, min(i + per_row, len(paths)))) for i in range(0, len(paths), per_row)]
        floor_top = STAGE[1]
        width = per_row * cell
        # The camera stands in front of each row, so the rows are pitched farther apart than it stands back: a row
        # never sits between the camera and the row it shoots (measured: at two cells apart the first row hid the second).
        tall = max(sz[1] for _cf, sz in boxes)
        d_cam = max(per_shot * cell, tall * 1.6) / 2 / math.tan(math.radians(fov) / 2) * 1.15
        pitch = d_cam + cell * 1.5
        depth = len(rows) * pitch
        fl = T.TOOLS["rbx_create"]("Part", root, name="Floor", props={
            "Anchored": True, "Material": "SmoothPlastic", "Color": [0.62, 0.64, 0.66],
            "Size": [min(2048, width + 2 * cell), 1, min(2048, depth + 2 * cell)],
            "CFrame": {"position": [STAGE[0] + width / 2 - cell / 2, floor_top - 0.5, STAGE[2] + depth / 2 - pitch / 2],
                       "rotation": [1, 0, 0, 0, 1, 0, 0, 0, 1]}})
        if not fl.get("ok"):
            return dict(fl, step="floor")
        slots = []
        for r, row in enumerate(rows):
            for c, i in enumerate(row):
                cf, sz = boxes[i]
                nm = "M%03d" % (i + 1)
                cl = T.TOOLS["rbx_clone"](paths[i], parent=root, name=nm)
                if not cl.get("ok"):
                    return dict(cl, step="copy %s" % V.join_path(paths[i]))
                mp = root + "." + nm
                pv = T.rbx_call(mp, "GetPivot")
                piv = pv.get("value") if pv.get("ok") else None
                if not isinstance(piv, dict):
                    return dict(pv, ok=False, step="GetPivot %s" % mp)
                # Seat the box's bottom on the floor at the slot, the front (-Z) toward the camera rows at -Z.
                want = [STAGE[0] + c * cell, floor_top + sz[1] / 2, STAGE[2] + r * pitch]
                off = [piv["position"][k] - cf["position"][k] for k in range(3)]
                mv = _move(mp, {"position": [want[k] + off[k] for k in range(3)], "rotation": piv["rotation"]})
                if not mv.get("ok"):
                    return dict(out, ok=False, step="place %s" % mp, error=mv.get("error"), pivot=mv)
                if label:
                    fp = _first_part(mp)
                    lb = T.TOOLS["rbx_world_gui"]("billboard", mp + "." + fp if fp else mp, name="Label", parent=mp,
                                                  props={"Size": [0, 200, 0, 40], "StudsOffsetWorldSpace": [0, sz[1] / 2 + 1.5, 0],
                                                         "AlwaysOnTop": False},
                                                  children=[{"class": "TextLabel", "name": "Text", "props": {
                                                      "Size": [1, 0, 1, 0], "BackgroundTransparency": 1, "TextScaled": True,
                                                      "Text": "%d %s" % (i + 1, paths[i][-1]), "TextColor3": [1, 1, 1],
                                                      "TextStrokeTransparency": 0}}])
                    if not lb.get("ok"):
                        return dict(lb, step="label %s" % mp)
                slots.append({"n": i + 1, "model": V.join_path(paths[i]), "copy": mp, "row": r + 1, "column": c + 1,
                              "at": want, "size": [round(x, 3) for x in sz]})
        out["slots"] = slots
        shots = []
        with _Window() as win:
            out["window"] = win.info()
            for r, row in enumerate(rows):
                for s in range(0, len(row), per_shot):
                    group = row[s:s + per_shot]
                    span = (len(group) - 1) * cell + cell
                    cx = STAGE[0] + ((s + s + len(group) - 1) / 2) * cell
                    cz = STAGE[2] + r * pitch
                    d = max(span, tall * 1.6) / 2 / math.tan(math.radians(fov) / 2) * 1.15
                    cam = {"position": [cx, floor_top + tall * 0.6 + d * 0.25, cz - d], "look_at": [cx, floor_top + tall / 2, cz],
                           "fov": fov}
                    sh = _shot(cam, max_width, "gallery_%02d" % (len(shots) + 1), folder)
                    if not sh.get("ok"):
                        out["shots"] = shots
                        return dict(out, ok=False, step="shot %d" % (len(shots) + 1), error=sh.get("error"))
                    shots.append({"n": len(shots) + 1, "file": sh["file"], "models": [g + 1 for g in group],
                                  "frame": sh.get("frame")})
        out.update(shots=shots, ok=True, verified=True)
        return out
    res = {}
    try:
        res = body()
    finally:
        if not keep:
            out["removed"] = [_remove(p) for p in reversed(made)]
    res.update({k: v for k, v in out.items() if k not in res or k == "removed"})
    if any(not r["removed"] for r in res.get("removed") or []):
        res.update(ok=False, verified=False, error="the gallery was not removed: %s" % res["removed"])
    return res

def _first_part(model_path):
    """The name of a model's first part (the BillboardGui's Adornee), or the model itself for a part."""
    T = _T()
    cat, err = T._catalog()
    cls = T._resolve(V.split_path(model_path))[0]
    if cls and not err and cat.is_a(cls, "BasePart"):
        return None
    tr = T._rbx("tree", root=V.split_path(model_path), depth=4, limit=500)

    def walk(n, at):
        for c in n.get("children") or []:
            if c.get("className") and cat.is_a(c["className"], "BasePart"):
                return at + [c["name"]]
            got = walk(c, at + [c["name"]])
            if got:
                return got
        return None
    got = walk(tr.get("root") or {}, [])
    return ".".join(got) if got else None
