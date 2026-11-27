"""Modeled kits over grey-box parts: the manifest, its placement rules, the palette atlas and the part-kit check, as plain
arithmetic shared by every tool that uses them (standard library only, no Studio and no Blender).

A grey-box structure built by code is dressed by meshes modeled in the local frame of the part each replaces (the internal notes L71,
L72, L86). The manifest says which mesh goes on which part and how:

    {"meshes": {"<key>": {"template": "<path of a BasePart in the place>" and/or "file": "<an .fbx>",
                          "size": [x, y, z], "offset": [x, y, z], "nominal": [x, y, z]}},
     "rules": [{"pattern": "<Lua pattern on the part's Name>", "mesh": "<key>",
                "yaw": <degrees> | "yawBySide": {"<first capture>": <degrees>, ...},
                "offset": [...], "size": [...],                     (override the mesh's)
                "tile": {"axes": ["X", "Z"], "tileSize": <studs>},  (parts of varying size)
                "grey": "hide" | "keep" | "delete"}],
     "atlas": {"grid": 16, "pixels": 1024, "marginPx": 20, "emissive": [[col, row], ...] | "emissiveRows": [15]}}

PLACEMENT (L71): mesh.CFrame = part.CFrame * CFrame.Angles(0, yaw, 0) * CFrame.new(offset); mesh.Size = size.
TILES (L72): along each tiled part axis n = max(1, floor(len / tileSize + 0.5)) cells of len / n; each cell's mesh is
scaled by cell / nominal on the mesh axis that the part axis turns into under the yaw (yaw must then be a multiple of 90),
at part.CFrame * CFrame.new(cellCenter) * yaw * CFrame.new(offset * scale). The first rule whose pattern matches a part's
Name takes it.

CFrames here are (R, p): R a 3x3 row-major rotation, p a position; Roblox's 12 components are p then R row by row.
NEVER RAISE from a public function: each answers (value, error) or a dict with ok.
"""
import json
import math
import os
import re

AXES = ("X", "Y", "Z")
GREY_MODES = ("hide", "keep", "delete")
MANIFEST_KEYS = {"meshes", "rules", "atlas", "parts", "slots", "assembly", "budgets", "assemblyCap", "combinations",
                 "slotTolerance", "sizeTolerance"}
MESH_KEYS = {"template", "file", "size", "offset", "nominal", "center", "maps"}
MAP_KEYS = ("color", "normal", "roughness", "metalness")   # a mesh's PNG maps for rbx_kit_apply's preview
RULE_KEYS = {"pattern", "mesh", "yaw", "yawBySide", "offset", "size", "tile", "grey", "name"}
TILE_KEYS = {"axes", "tileSize"}
ATLAS_KEYS = {"grid", "pixels", "marginPx", "emissive", "emissiveRows", "colors", "emissiveColors"}


# --------------------------------------------------------------------------------------------------- Lua patterns
_LUA_CLASS = {"a": "A-Za-z", "d": "0-9", "l": "a-z", "u": "A-Z", "s": r"\s", "w": "A-Za-z0-9", "x": "0-9A-Fa-f",
              "p": re.escape("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"), "c": r"\x00-\x1f\x7f"}


def lua_to_regex(pat):
    """A Lua pattern as a Python regex, or (None, error) for the parts Python has no equal for (%b, %f, position
    captures, back-references). Lua's '-' quantifier is the lazy '*?'."""
    if not isinstance(pat, str) or not pat:
        return None, "a pattern is a non-empty Lua pattern string"
    out, i, n = [], 0, len(pat)

    def cls(ch):
        if ch.lower() in _LUA_CLASS:
            inner = _LUA_CLASS[ch.lower()]
            return ("[^%s]" if ch.isupper() else "[%s]") % inner, inner, ch.isupper()
        if ch.isalnum():
            return None, None, None
        return re.escape(ch), re.escape(ch), False

    while i < n:
        c = pat[i]
        if c == "^" and i == 0:
            out.append("^")
            i += 1
            continue
        if c == "$" and i == n - 1:
            out.append("$")
            i += 1
            continue
        if c == "%":
            if i + 1 >= n:
                return None, "pattern ends with a lone %"
            d = pat[i + 1]
            if d in "bf" or d.isdigit():
                return None, "%%%s has no Python equal; use a plain pattern" % d
            item, _inner, _neg = cls(d)
            if item is None:
                return None, "%%%s is not a Lua class" % d
            out.append(item)
            i += 2
        elif c == "[":
            j, body, first = i + 1, [], True
            if j < n and pat[j] == "^":
                body.append("^")
                j += 1
            while j < n and (first or pat[j] != "]"):          # Lua reads a ] first in a set as itself
                first = False
                if pat[j] == "%" and j + 1 < n:
                    item, inner, neg = cls(pat[j + 1])
                    if item is None or neg:
                        return None, "%%%s inside a set is not supported" % pat[j + 1]
                    body.append(inner)
                    j += 2
                    continue
                body.append("\\" + pat[j] if pat[j] in "\\[]" else pat[j])
                j += 1
            if j >= n:
                return None, "a [ set is not closed"
            out.append("[" + "".join(body) + "]")
            i = j + 1
        elif c == "(":
            if i + 1 < n and pat[i + 1] == ")":
                return None, "a position capture () has no Python equal"
            out.append("(")
            i += 1
        elif c == ")":
            out.append(")")
            i += 1
        elif c == ".":
            out.append(".")
            i += 1
        elif c in "*+?":
            out.append(c)
            i += 1
        elif c == "-":
            prev = out[-1] if out else ""
            out.append("*?" if prev and prev not in ("^", "(") else re.escape("-"))
            i += 1
        else:
            out.append(re.escape(c))
            i += 1
    try:
        return re.compile("".join(out)), None
    except re.error as e:
        return None, "pattern %r did not translate (%s)" % (pat, e)


# --------------------------------------------------------------------------------------------------------- CFrames
def cf_from_components(c):
    """Roblox's 12 CFrame components (x, y, z, R00..R22) -> (R, p)."""
    c = [float(v) for v in c]
    return [c[3:6], c[6:9], c[9:12]], c[0:3]


def cf_components(cf):
    R, p = cf
    return [p[0], p[1], p[2]] + [R[i][j] for i in range(3) for j in range(3)]


def cf_mul(a, b):
    (Ra, pa), (Rb, pb) = a, b
    R = [[sum(Ra[i][k] * Rb[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    p = [sum(Ra[i][k] * pb[k] for k in range(3)) + pa[i] for i in range(3)]
    return R, p


def cf_new(p):
    return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [float(v) for v in p]


def cf_yaw(deg):
    """CFrame.Angles(0, rad(deg), 0)."""
    a = math.radians(float(deg))
    c, s = math.cos(a), math.sin(a)
    return [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], [0.0, 0.0, 0.0]


def cf_roll(deg):
    """CFrame.Angles(0, 0, rad(deg)): a roll about the part's own Z."""
    a = math.radians(float(deg))
    c, s = math.cos(a), math.sin(a)
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], [0.0, 0.0, 0.0]


def cf_inverse(cf):
    R, p = cf
    Rt = [[R[j][i] for j in range(3)] for i in range(3)]
    return Rt, [-sum(Rt[i][k] * p[k] for k in range(3)) for i in range(3)]


def cf_point(cf, v):
    R, p = cf
    return [sum(R[i][k] * v[k] for k in range(3)) + p[i] for i in range(3)]


def _vec3(v):
    return isinstance(v, (list, tuple)) and len(v) == 3 and all(
        isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in v)


def _round(v, d=6):
    return [round(float(x), d) + 0.0 for x in v]


# -------------------------------------------------------------------------------------------------------- manifest
def read_manifest(manifest):
    """A manifest dict, or a path to its JSON -> (dict, base dir for relative files, error)."""
    base = None
    if isinstance(manifest, str):
        try:
            with open(manifest, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError) as e:
            return None, None, "manifest %s did not read: %s" % (manifest, e)
        base = os.path.dirname(os.path.abspath(manifest))
        manifest = data
    if not isinstance(manifest, dict):
        return None, None, "manifest is a dict or a path to a JSON file"
    unknown = sorted(set(manifest) - MANIFEST_KEYS)
    if unknown:
        return None, None, "manifest has unknown key(s) %s; it takes %s" % (", ".join(unknown), ", ".join(sorted(MANIFEST_KEYS)))
    return manifest, base, None


def check_placement_manifest(m):
    """Every problem in a placement manifest's meshes and rules, as a list of strings; empty when usable."""
    probs = []
    meshes = m.get("meshes")
    rules = m.get("rules")
    if not isinstance(meshes, dict) or not meshes:
        probs.append("meshes is a non-empty {key: {template | file, size, offset}} map")
        meshes = {}
    if not isinstance(rules, list) or not rules:
        probs.append("rules is a non-empty list of {pattern, mesh, yaw | yawBySide, ...}")
        rules = []
    for k, mesh in meshes.items():
        if not isinstance(mesh, dict):
            probs.append("meshes.%s is a dict" % k)
            continue
        bad = sorted(set(mesh) - MESH_KEYS)
        if bad:
            probs.append("meshes.%s has unknown key(s) %s; it takes %s" % (k, ", ".join(bad), ", ".join(sorted(MESH_KEYS))))
        if not mesh.get("template") and not mesh.get("file"):
            probs.append("meshes.%s names neither a template (a part in the place) nor a file" % k)
        for key in ("size", "offset", "nominal", "center"):
            if key in mesh and not _vec3(mesh[key]):
                probs.append("meshes.%s.%s is three finite numbers" % (k, key))
        if "size" in mesh and _vec3(mesh["size"]) and not all(v > 0 for v in mesh["size"]):
            probs.append("meshes.%s.size is positive on every axis" % k)
        maps = mesh.get("maps")
        if maps is not None and (not isinstance(maps, dict) or set(maps) - set(MAP_KEYS)
                                 or not all(isinstance(v, str) and v for v in maps.values())):
            probs.append("meshes.%s.maps is {%s} each a PNG path (relative to the manifest's folder)" % (k, ", ".join(MAP_KEYS)))
    for i, r in enumerate(rules):
        if not isinstance(r, dict):
            probs.append("rules[%d] is a dict" % i)
            continue
        bad = sorted(set(r) - RULE_KEYS)
        if bad:
            probs.append("rules[%d] has unknown key(s) %s; it takes %s" % (i, ", ".join(bad), ", ".join(sorted(RULE_KEYS))))
        rx, err = lua_to_regex(r.get("pattern"))
        if err:
            probs.append("rules[%d].pattern: %s" % (i, err))
        if r.get("mesh") not in meshes:
            probs.append("rules[%d].mesh %r is not a key of meshes" % (i, r.get("mesh")))
        if "yaw" in r and "yawBySide" in r:
            probs.append("rules[%d] gives both yaw and yawBySide" % i)
        if "yaw" in r and not (isinstance(r["yaw"], (int, float)) and not isinstance(r["yaw"], bool)
                               and math.isfinite(r["yaw"])):
            probs.append("rules[%d].yaw is degrees" % i)
        if "yawBySide" in r:
            ys = r["yawBySide"]
            if not isinstance(ys, dict) or not ys or not all(isinstance(v, (int, float)) for v in ys.values()):
                probs.append("rules[%d].yawBySide is {side: degrees}" % i)
            elif rx is not None and rx.groups < 1:
                probs.append("rules[%d].yawBySide reads the side from the pattern's first capture, and it has none" % i)
        for key in ("offset", "size"):
            if key in r and not _vec3(r[key]):
                probs.append("rules[%d].%s is three finite numbers" % (i, key))
        if r.get("grey", "hide") not in GREY_MODES:
            probs.append("rules[%d].grey is one of %s" % (i, ", ".join(GREY_MODES)))
        if "tile" in r:
            t = r["tile"]
            if not isinstance(t, dict) or set(t) - TILE_KEYS:
                probs.append("rules[%d].tile is {axes, tileSize}" % i)
            else:
                ax = t.get("axes")
                if not isinstance(ax, list) or not ax or not all(a in AXES for a in ax) or len(set(ax)) != len(ax):
                    probs.append("rules[%d].tile.axes is a list of X, Y, Z" % i)
                ts = t.get("tileSize")
                if not (isinstance(ts, (int, float)) and not isinstance(ts, bool) and ts > 0):
                    probs.append("rules[%d].tile.tileSize is studs, above 0" % i)
    return probs


def _part_axis_of(R_yaw, mesh_axis):
    """Which part axis (0-2) the mesh axis lies along under the yaw, or None when the yaw is not a multiple of 90."""
    col = [R_yaw[0][mesh_axis], R_yaw[1][mesh_axis], R_yaw[2][mesh_axis]]
    for k in range(3):
        if abs(abs(col[k]) - 1.0) < 1e-6:
            return k
    return None


def plan(parts, manifest):
    """Apply a placement manifest's rules to a part list. parts: [{path, name, cframe (12 numbers), size [3]}].
    -> {ok, placements [{part, partName, rule, mesh, name, cframe, size, tile?, grey}], matched, unmatched [part],
    rulesUnmatched [rule index and pattern], refused [{part, error}], error?}"""
    m, _base, err = read_manifest(manifest)
    if err:
        return {"ok": False, "error": err}
    probs = check_placement_manifest(m)
    if probs:
        return {"ok": False, "error": "the manifest is not usable: " + "; ".join(probs), "problems": probs}
    if not isinstance(parts, list) or not parts:
        return {"ok": False, "error": "no parts to dress: an empty part list places nothing"}
    rules = []
    for i, r in enumerate(m["rules"]):
        rx, _e = lua_to_regex(r["pattern"])
        rules.append((i, r, rx))
    hits = {i: 0 for i, _r, _x in rules}
    placements, unmatched, refused, matched = [], [], [], 0
    for part in parts:
        name = part.get("name")
        cfc, size = part.get("cframe"), part.get("size")
        if not isinstance(name, str) or not (isinstance(cfc, list) and len(cfc) == 12) or not _vec3(size):
            refused.append({"part": part.get("path"), "error": "a part is {path, name, cframe (12 numbers), size [3]}"})
            continue
        rule = None
        for i, r, rx in rules:
            mt = rx.search(name)
            if mt:
                rule = (i, r, mt)
                break
        if rule is None:
            unmatched.append(part.get("path") or name)
            continue
        i, r, mt = rule
        mesh = m["meshes"][r["mesh"]]
        if "yawBySide" in r:
            side = mt.group(1) if mt.groups() else None
            if side not in r["yawBySide"]:
                refused.append({"part": part.get("path") or name, "rule": i,
                                "error": "side %r from the name has no yaw in rules[%d].yawBySide (%s)" % (
                                    side, i, ", ".join(sorted(r["yawBySide"])))})
                continue
            yaw = float(r["yawBySide"][side])
        else:
            yaw = float(r.get("yaw", 0.0))
        offset = r.get("offset", mesh.get("offset", [0.0, 0.0, 0.0]))
        msize = r.get("size", mesh.get("size"))
        if not _vec3(msize):
            refused.append({"part": part.get("path") or name, "rule": i,
                            "error": "meshes.%s has no size and rules[%d] gives none" % (r["mesh"], i)})
            continue
        hits[i] += 1
        matched += 1
        pcf = cf_from_components(cfc)
        ycf = cf_yaw(yaw)
        base = {"part": part.get("path") or name, "partName": name, "rule": i, "mesh": r["mesh"],
                "grey": r.get("grey", "hide"), "yaw": yaw}
        if "tile" not in r:
            cf = cf_mul(cf_mul(pcf, ycf), cf_new(offset))
            tmpl = r.get("name", "%s_Kit")
            placements.append(dict(base, name=tmpl % name if "%s" in tmpl else tmpl, cframe=_round(cf_components(cf)),
                                   size=_round(msize)))
            continue
        t = r["tile"]
        S = float(t["tileSize"])
        nominal = mesh.get("nominal", msize)
        tiled = {AXES.index(a) for a in t["axes"]}
        counts, cells = [1, 1, 1], [float(size[0]), float(size[1]), float(size[2])]
        for k in tiled:
            counts[k] = max(1, int(math.floor(float(size[k]) / S + 0.5)))
            cells[k] = float(size[k]) / counts[k]
        scale = [1.0, 1.0, 1.0]
        bad_axis = None
        for a in range(3):
            k = _part_axis_of(ycf[0], a)
            if k is None:
                bad_axis = a
                break
            if k in tiled:
                scale[a] = cells[k] / float(nominal[a])
        if bad_axis is not None:
            hits[i] -= 1
            matched -= 1
            refused.append({"part": part.get("path") or name, "rule": i,
                            "error": "a tiled rule's yaw must be a multiple of 90 degrees (got %g)" % yaw})
            continue
        k_off = [offset[a] * scale[a] for a in range(3)]
        tsize = [msize[a] * scale[a] for a in range(3)]
        idx = 0
        for ix in range(counts[0]):
            for iy in range(counts[1]):
                for iz in range(counts[2]):
                    c = [(-size[k] / 2.0 + ((ix, iy, iz)[k] + 0.5) * cells[k]) if k in tiled else 0.0 for k in range(3)]
                    cf = cf_mul(cf_mul(cf_mul(pcf, cf_new(c)), ycf), cf_new(k_off))
                    idx += 1
                    placements.append(dict(base, name="%s_Kit%d" % (name, idx), cframe=_round(cf_components(cf)),
                                           size=_round(tsize), tile={"cell": [ix, iy, iz], "counts": counts,
                                                                     "scale": _round(scale)}))
    out = {"ok": True, "placements": placements, "matched": matched, "unmatched": unmatched,
           "rulesUnmatched": [{"rule": i, "pattern": r["pattern"]} for i, r, _x in rules if not hits[i]],
           "refused": refused}
    if refused:
        out.update(ok=False, error="%d part(s) refused: %s" % (len(refused), "; ".join(
            "%s: %s" % (x["part"], x["error"]) for x in refused[:5])))
    elif not placements:
        out.update(ok=False, error="no rule matched any of the %d part(s): nothing would be placed" % len(parts))
    return out


# ----------------------------------------------------------------------------------------------- palette atlas
DEFAULT_ATLAS = {"grid": 16, "pixels": 1024, "marginPx": 20, "emissiveRows": [15]}


def atlas_spec(atlas):
    """(spec, error): grid, pixels, marginPx and the emissive cells as a set of (col, row) with row 0 at the image top."""
    a = dict(DEFAULT_ATLAS)
    if isinstance(atlas, str):
        try:
            with open(atlas, encoding="utf-8") as fh:
                atlas = json.load(fh)
        except (OSError, ValueError) as e:
            return None, "atlas %s did not read: %s" % (atlas, e)
    if isinstance(atlas, dict) and isinstance(atlas.get("cells"), list):     # the cell table rbx_palette_atlas wrote
        atlas = {k: atlas[k] for k in ("grid", "pixels", "marginPx", "emissive") if k in atlas}
    if atlas is not None:
        if not isinstance(atlas, dict) or set(atlas) - ATLAS_KEYS:
            return None, "atlas is {%s}" % ", ".join(sorted(ATLAS_KEYS))
        if "emissive" in atlas:
            a.pop("emissiveRows", None)
        a.update(atlas)
    g, px, mg = a.get("grid"), a.get("pixels"), a.get("marginPx")
    if not (isinstance(g, int) and 1 <= g <= 256 and isinstance(px, int) and px >= g and isinstance(mg, (int, float))
            and 0 <= mg < px / g / 2.0):
        return None, "atlas grid is 1-256 cells a side, pixels at least that, marginPx under half a cell (%s)" % (a,)
    em = set()
    for c in a.get("emissive") or []:
        if not (isinstance(c, (list, tuple)) and len(c) == 2 and all(isinstance(v, int) and 0 <= v < g for v in c)):
            return None, "atlas.emissive is a list of [col, row] cells inside the grid"
        em.add((c[0], c[1]))
    for row in a.get("emissiveRows") or []:
        if not (isinstance(row, int) and 0 <= row < g):
            return None, "atlas.emissiveRows are row numbers inside the grid (0 is the top)"
        em.update((col, row) for col in range(g))
    a["emissiveSet"] = em
    return a, None


def uv_cell(spec, u, v):
    """(col, row, inner): the cell a UV falls in (row 0 at the image top, v up in UV space) and whether it is inside
    the cell's inner area, marginPx from every edge."""
    g, px, mg = spec["grid"], spec["pixels"], spec["marginPx"]
    x, y = u * px, (1.0 - v) * px
    cw = px / float(g)
    col, row = int(math.floor(x / cw)), int(math.floor(y / cw))
    fx, fy = x - col * cw, y - row * cw
    inner = mg - 1e-6 <= fx <= cw - mg + 1e-6 and mg - 1e-6 <= fy <= cw - mg + 1e-6 and 0 <= col < g and 0 <= row < g
    return col, row, inner


def face_cells(spec, polys, uv_layer):
    """For each face, the one cell its UVs sit inside, or why not -> (cells used, bad faces [{face, cells, why}])."""
    used, bad, pv = set(), [], 0
    for fi, poly in enumerate(polys):
        cells, inner_all = set(), True
        for _k in poly:
            t = uv_layer[pv] if pv < len(uv_layer) else None
            pv += 1
            if t is None:
                inner_all = False
                continue
            col, row, inner = uv_cell(spec, t[0], t[1])
            cells.add((col, row))
            inner_all = inner_all and inner
        if len(cells) == 1 and inner_all:
            used |= cells
        else:
            bad.append({"face": fi, "cells": sorted([list(c) for c in cells]),
                        "why": "spans %d cells" % len(cells) if len(cells) > 1 else "inside the cell's margin"})
            used |= {c for c in cells if 0 <= c[0] < spec["grid"] and 0 <= c[1] < spec["grid"]}
    return used, bad


def _hsv(h, s, v):
    i = int(h * 6) % 6
    f = h * 6 - int(h * 6)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    return [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i]


def default_colors(grid):
    """A grid x grid palette: hue across, value and saturation down; the last row bright for emissive."""
    out = {}
    for row in range(grid):
        for col in range(grid):
            if row == grid - 1:
                r, g, b = _hsv(col / float(grid), 0.85, 1.0)
            elif col == 0:
                r = g = b = row / float(max(1, grid - 2))
            else:
                r, g, b = _hsv((col - 1) / float(max(1, grid - 1)), 0.25 + 0.6 * (row % 4) / 3.0,
                               0.35 + 0.6 * (row // 4) / max(1.0, (grid - 2) / 4.0))
            out[(col, row)] = [int(round(min(1, max(0, x)) * 255)) for x in (r, g, b)]
    return out


def make_atlas(spec, colors=None, grime=0.0, seed=1):
    """(color RGBA bytes, emissive-mask RGBA bytes, cell table) for the atlas. colors {(col,row): [r,g,b]} default to
    default_colors; grime (0-0.3) darkens each pixel by a seeded noise so faces whose UVs differ show it differently."""
    import random
    g, px = spec["grid"], spec["pixels"]
    cw = px // g
    cols = default_colors(g)
    cols.update(colors or {})
    rnd = random.Random(seed)
    color, mask = bytearray(px * px * 4), bytearray(px * px * 4)
    for y in range(px):
        row = min(g - 1, y // cw)
        for x in range(px):
            col = min(g - 1, x // cw)
            r, gg, b = cols[(col, row)]
            k = 1.0 - (rnd.random() * grime if grime else 0.0)
            i = (y * px + x) * 4
            color[i:i + 4] = bytes((int(r * k), int(gg * k), int(b * k), 255))
            e = 255 if (col, row) in spec["emissiveSet"] else 0
            mask[i:i + 4] = bytes((e, e, e, 255))
    table = {"grid": g, "pixels": px, "marginPx": spec["marginPx"],
             "cells": [{"col": c, "row": r, "rgb": cols[(c, r)], "emissive": (c, r) in spec["emissiveSet"],
                        "uvCenter": [round((c + 0.5) / g, 6), round(1 - (r + 0.5) / g, 6)]}
                       for r in range(g) for c in range(g)],
             "emissive": sorted([list(c) for c in spec["emissiveSet"]])}
    return bytes(color), bytes(mask), table


# --------------------------------------------------------------------------- Roblox placements as Blender matrices
# Roblox (X, Y, Z) = Blender (-x, z, y) (RB1156); B is its inverse, Blender = (-X, Z, Y). Both are proper turns (det +1).
R_BL_TO_RBX = [[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
R_RBX_TO_BL = [[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]


def _m4(R, p=(0.0, 0.0, 0.0)):
    return [[R[0][0], R[0][1], R[0][2], p[0]], [R[1][0], R[1][1], R[1][2], p[1]], [R[2][0], R[2][1], R[2][2], p[2]],
            [0.0, 0.0, 0.0, 1.0]]


def _mm4(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def blender_matrix(cframe12, size=None, mesh_size=None, mesh_center=None, studs_per_unit=1.0):
    """The Blender matrix_world (4x4 row-major, Blender units = studs) for:
    - a mesh imported from an FBX with its transforms applied (Blender metres), placed as a Roblox MeshPart at cframe12
      with Size `size`, when mesh_size and mesh_center (its box on Roblox axes, studs) are given:
      B * M * diag(size / mesh_size) * T(-mesh_center) * (studs_per_unit * R);
    - a unit cube (size 1) drawn as a part of Size `size` at cframe12, when mesh_size is None: B * M * diag(size);
    - a camera at cframe12 (looking down its -Z, Y up in both programs) when size is None: B * M."""
    R, p = cf_from_components(cframe12)
    m = _mm4(_m4(R_RBX_TO_BL), _m4(R, p))
    if size is None:
        return m
    if mesh_size is None:
        return _mm4(m, _m4([[size[0], 0, 0], [0, size[1], 0], [0, 0, size[2]]]))
    s = [float(size[k]) / float(mesh_size[k]) if mesh_size[k] else 1.0 for k in range(3)]
    k = float(studs_per_unit)
    rest = _m4([[k * R_BL_TO_RBX[i][j] for j in range(3)] for i in range(3)], [-c for c in mesh_center])
    return _mm4(_mm4(m, _m4([[s[0], 0, 0], [0, s[1], 0], [0, 0, s[2]]])), rest)


def look_at(eye, target):
    """CFrame.lookAt(eye, target) as 12 components (Y up)."""
    f = [target[k] - eye[k] for k in range(3)]
    n = math.sqrt(sum(v * v for v in f)) or 1.0
    f = [v / n for v in f]
    up = [0.0, 1.0, 0.0] if abs(f[1]) < 0.999 else [0.0, 0.0, -1.0]
    r = [f[1] * up[2] - f[2] * up[1], f[2] * up[0] - f[0] * up[2], f[0] * up[1] - f[1] * up[0]]
    rn = math.sqrt(sum(v * v for v in r)) or 1.0
    r = [v / rn for v in r]
    u = [r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]]
    back = [-v for v in f]
    R = [[r[0], u[0], back[0]], [r[1], u[1], back[1]], [r[2], u[2], back[2]]]
    return cf_components((R, list(eye)))


def default_cameras(boxes, fov_deg=50.0):
    """An aerial and a three-quarter view framing every box [{cframe, size}] -> [{name, cframe}]: each camera far
    enough back that the boxes' bounding sphere fits the field of view (fov_deg, the preview's default)."""
    pts = []
    for b in boxes:
        c = b["cframe"][:3]
        h = math.sqrt(sum(float(v) * float(v) for v in b["size"])) / 2.0
        pts += [[c[k] - h for k in range(3)], [c[k] + h for k in range(3)]]
    if not pts:
        return []
    lo = [min(p[k] for p in pts) for k in range(3)]
    hi = [max(p[k] for p in pts) for k in range(3)]
    mid = [(lo[k] + hi[k]) / 2.0 for k in range(3)]
    radius = max(1e-3, math.sqrt(sum((hi[k] - lo[k]) ** 2 for k in range(3))) / 2.0)
    dist = radius / math.sin(math.radians(fov_deg) / 2.0) * 1.05
    out = []
    for name, d in (("aerial", [0.0, 1.0, 0.02]), ("threeQuarter", [0.6, 0.5, 0.6])):
        n = math.sqrt(sum(v * v for v in d))
        out.append({"name": name, "cframe": look_at([mid[k] + d[k] / n * dist for k in range(3)], mid)})
    return out


def key_color(i):
    """A distinct preview color for the i-th mesh file (golden-angle hues), [r, g, b] 0-1."""
    return [round(v, 4) for v in _hsv((i * 0.618034) % 1.0, 0.55, 0.9)]
