"""Modeled kits over a grey box, and the mesh pipeline before an upload.

    rbx_mesh_preflight     an FBX checked against the art contract, from the file alone: triangles per mesh, one
                           material, one UV set inside 0-1 and inside the palette atlas's cells, the emissive cells it
                           uses, no n-gons, no closed part wound inward (negative signed volume, L122), its size and box center on Roblox axes under the measured import mapping
                           (RB1156), against a manifest entry when given. Offline.
    rbx_palette_atlas      the shared palette atlas: a grid of flat colors as a PNG, its emissive mask, and the cell table
                           (each cell's color, UV center and emissive flag). Offline; read back from the written files.
    rbx_kit_apply          a placement manifest applied to a built model in Studio: each matching part gets a copy of its
                           mesh's template at part.CFrame * yaw * offset (tiles scaled per cell), non-colliding, in the
                           part's parent; the grey part hidden, kept or deleted - and never deleted while a script names
                           it (it is hidden instead, and the references listed). Reports matched, unmatched and rules
                           that matched nothing; every write read back.
    rbx_kit_rules_module   the same manifest as a Luau ModuleScript (rules + an apply function) for parts a game builds
                           at run time, written into the place and read back, or returned as source.
    rbx_kit_check          a part kit's manifest checked: every assembly socket present on its parent slot's parts,
                           each slot's parts within one box (15%), per-slot and whole-assembly triangle budgets from the
                           exported files, each file's box against the manifest's size and center (2%); and the named
                           combinations assembled into placements bl_kit_preview renders. Offline.
    rbx_asset_upload_batch a manifest of files uploaded one at a time under a lock: an entry already uploaded or rejected
                           (by file sha256 or mesh-data hash) is skipped, the run stops at a 401 or 403, the manifest is
                           written after each file. Gated: dry_run (the default) says exactly what would be sent.
                           ids_module writes the manifest's asset ids as a generated Luau module (.luau, .lua or
                           an .rbxmx ModuleScript), keyed by each entry's file.

The placement arithmetic, the Lua pattern reader and the atlas are mif_roblox/kit.py; the FBX reader is fbx_mesh.py.
NEVER RAISE: every failure is {"ok": False, ...}.
"""
import json
import os
import re
import time
from typing import Any

from mif_roblox import fbx_mesh as F
from mif_roblox import kit as K

__all__ = ["rbx_mesh_preflight", "rbx_palette_atlas", "rbx_kit_apply", "rbx_kit_rules_module", "rbx_kit_check",
           "rbx_asset_upload_batch", "rbx_rig_check"]

TRIANGLE_LIMIT = 20000            # Roblox's mesh specifications page (mif_roblox/preflight.py RULES["triangles"])
PREVIEW_MAX_PX = 1024             # a preview map's longest side
VOLUME_EPS = 1e-9                 # cubic studs: below this a closed part's signed volume is noise, not a winding
SIZE_TOL = 0.02                   # a file's box against the manifest's size, 2%
SLOT_TOL = 0.15                   # parts filling one slot fit one box within 15% on every axis (L79)
POS_TOL = 1e-3                    # studs, a placed mesh read back
MAX_PARTS = 5000
BATCH_OPS = 100                   # rbx_batch's limit


def _T():
    from mif_roblox import tools
    return tools


def _G():
    from mif_roblox import domain_geo
    return domain_geo


def _no(error, **kw):
    return dict({"ok": False, "ran": False, "error": error}, **kw)


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


# ------------------------------------------------------------------------------------------------ the preflight
def _preflight_one(path, spec, expect, triangle_limit, size_tolerance):
    try:
        sc = F.load(path)
    except (OSError, ValueError) as e:
        return {"file": path, "ok": False, "pass": False, "error": "did not read: %s" % e}
    out = {"file": path, "ok": True, "notes": sc.notes}
    fails = []
    if not sc.meshes:
        return dict(out, ok=False, **{"pass": False, "error": "no meshes in the file: nothing was checked"})
    if not sc.rbx_axes():
        fails.append("the file's axis settings %s are not the ones Studio's import was measured on (%s, RB1156), so its "
                     "size and front on Roblox axes are unknown" % (sc.axis_settings, F.MEASURED_AXES))
    meshes = []
    for m in sc.meshes:
        row = {"name": m["name"], "triangles": m["triangles"], "materials": m["materials"], "uvSets": m["uvSets"],
               "ngons": m["ngons"], "largestPolygon": m["largestPolygon"], "skinned": m["skinned"]}
        if m["triangles"] > triangle_limit:
            fails.append("%s: %d triangles, over %d" % (m["name"], m["triangles"], triangle_limit))
        if len(m["materials"]) > 1:
            fails.append("%s: %d materials (%s); a mesh takes one" % (m["name"], len(m["materials"]), ", ".join(m["materials"])))
        if m["uvSets"] != 1:
            fails.append("%s: %d UV sets; Studio takes one" % (m["name"], m["uvSets"]))
        if m["ngons"]:
            fails.append("%s: %d n-gon(s), the largest %d-sided" % (m["name"], m["ngons"], m["largestPolygon"]))
        uv = m["uvLayers"][0] if m["uvLayers"] else []
        outside = sum(1 for t in uv if t is None or not (-1e-6 <= t[0] <= 1 + 1e-6 and -1e-6 <= t[1] <= 1 + 1e-6))
        row["uvsOutside01"] = outside
        if outside:
            fails.append("%s: %d UV(s) outside 0-1" % (m["name"], outside))
        if spec and uv:
            used, bad = K.face_cells(spec, m["polyList"], uv)
            row["cellsUsed"] = sorted([list(c) for c in used])
            row["emissiveCells"] = sorted([list(c) for c in used if c in spec["emissiveSet"]])
            row["facesOffCell"] = len(bad)
            if bad:
                row["facesOffCellSample"] = bad[:10]
                fails.append("%s: %d face(s) not inside one atlas cell's inner area (%s)" % (
                    m["name"], len(bad), "; ".join("face %d %s" % (b["face"], b["why"]) for b in bad[:3])))
        # INSIDE OUT (L122): a closed part whose signed volume is negative has every face wound inward, and Studio culls
        # its outside. Each part is welded by position and judged alone, so one inverted part among good ones is named.
        parts = m.get("shells") or []
        inverted = [i for i, s in enumerate(parts) if s["closed"] and s["signedVolume"] is not None
                    and s["signedVolume"] < -VOLUME_EPS]
        row["parts"] = [{"part": i + 1, "faces": s["faces"], "closed": s["closed"], "signedVolume": s["signedVolume"],
                         "insideOut": i in inverted} for i, s in enumerate(parts)]
        row["closedParts"] = sum(1 for s in parts if s["closed"])
        row["mirroredNode"] = m["mirroredNode"]
        for i in inverted:
            s = parts[i]
            a, b = F.rbx_point(s["boxMin"]), F.rbx_point(s["boxMax"])
            mid = [round((a[k] + b[k]) / 2, 4) for k in range(3)]
            fails.append("%s: closed part %d of %d (%d faces, centered %s on Roblox axes) has signed volume %.6g: its "
                         "faces wind inward, so Studio culls its outside (a mirrored placement matrix, L122)%s" % (
                             m["name"], i + 1, len(parts), s["faces"], mid, s["signedVolume"],
                             "; the node's own transform is mirrored" if m["mirroredNode"] else ""))
        size, center = F.rbx_box(m)
        row["sizeStuds"], row["centerStuds"] = size, center
        meshes.append(row)
    size, center = F.scene_rbx_box(sc)
    out.update(meshes=meshes, sizeStuds=size, centerStuds=center,
               front="the file's +Z (Blender's -Y under forward -Z) is Roblox's front, -Z (LookVector) (RB1156)",
               mapping="Roblox (X, Y, Z) = file (-X, +Y, -Z) at 1 stud per file unit (RB1156, RB1700)")
    if spec:
        allc = sorted({tuple(c) for r in meshes for c in r.get("cellsUsed", [])})
        out["cellsUsed"] = [list(c) for c in allc]
        out["emissiveCells"] = [list(c) for c in allc if c in spec["emissiveSet"]]
    if expect:
        cmp = {}
        for key, got in (("size", size), ("center", center)):
            want = expect.get(key)
            if want is None:
                continue
            if not (isinstance(want, list) and len(want) == 3 and all(_num(v) for v in want)):
                fails.append("expect.%s is three numbers" % key)
                continue
            scale = max(max(abs(v) for v in want), max(abs(v) for v in size), 1e-6)
            err = max(abs(got[k] - want[k]) for k in range(3))
            ok = err <= size_tolerance * scale
            cmp[key] = {"manifest": want, "file": got, "maxError": round(err, 6), "pass": ok}
            if not ok:
                fails.append("%s on Roblox axes is %s, the manifest says %s (off by %.4g, tolerance %g%% of %.4g)" % (
                    key, got, want, err, size_tolerance * 100, scale))
        out["againstManifest"] = cmp
    out["pass"] = not fails
    out["failed"] = fails
    return out


def rbx_mesh_preflight(file: Any, atlas: Any = None, expect: dict = None, triangle_limit: int = TRIANGLE_LIMIT,
                       size_tolerance: float = SIZE_TOL) -> dict:
    "Roblox: check FBX files against a kit's art contract before upload, from the files alone (nothing reaches Studio): per mesh the triangles against triangle_limit (20,000, Roblox's page), one material, one UV set, UVs inside 0-1, no n-gons, and every closed part (welded by position) with a positive signed volume - a part wound inward fails by mesh name and part number, since Studio culls its outside; with atlas ({grid, pixels, marginPx, emissive [[col,row]] | emissiveRows}, or the cell table rbx_palette_atlas wrote) every face's UVs inside one cell's inner area and the emissive cells used; the box size and center on Roblox axes under the measured import mapping (file (-X, +Y, -Z), 1 stud per file unit, RB1156) and the front (the file's +Z, Blender -Y, becomes Roblox -Z); with expect ({size, center}, a manifest entry) both compared within size_tolerance. file is a path or a list of paths. pass is per file; ok false when a file did not read or any failed."
    files = file if isinstance(file, list) else [file]
    if not files or not all(isinstance(f, str) and f for f in files):
        return _no("file is an .fbx path or a list of them")
    if not (isinstance(triangle_limit, int) and not isinstance(triangle_limit, bool) and triangle_limit > 0):
        return _no("triangle_limit is a positive whole number")
    if not (_num(size_tolerance) and 0 < size_tolerance < 1):
        return _no("size_tolerance is a fraction between 0 and 1")
    if expect is not None and (not isinstance(expect, dict) or set(expect) - {"size", "center"}):
        return _no("expect is {size, center} on Roblox axes, studs")
    spec = None
    if atlas is not None:
        spec, err = K.atlas_spec(atlas)
        if err:
            return _no(err)
    missing = [f for f in files if not os.path.isfile(f)]
    if missing:
        return _no("not a file: %s" % ", ".join(missing))
    results = [_preflight_one(f, spec, expect, triangle_limit, size_tolerance) for f in files]
    failed = [r["file"] for r in results if not r.get("pass")]
    return {"ok": not failed, "files": results, "checked": len(results), "failedFiles": failed,
            "atlas": {k: spec[k] for k in ("grid", "pixels", "marginPx")} if spec else None,
            "error": "%d of %d file(s) failed the contract" % (len(failed), len(results)) if failed else None}


# --------------------------------------------------------------------------------------------------- the atlas
def rbx_palette_atlas(out_dir: str, name: str = "palette", atlas: dict = None, colors: dict = None, grime: float = 0.0,
                      seed: int = 1) -> dict:
    "Roblox: write a kit's shared palette atlas into out_dir: <name>.png (a grid of flat colors, default 16 x 16 at 1024 px, hue across and value down, the last row bright), <name>_emissive.png (white where a cell is emissive, default the last row) and <name>_cells.json (each cell's rgb, uvCenter and emissive flag, plus grid, pixels and marginPx - the table rbx_mesh_preflight and bl_uv_to_cell read). colors overrides cells as {\"col,row\": [r, g, b]} (0-255); grime 0-0.3 darkens pixels by seeded noise so faces with different UVs vary. Every file is read back: its size and each cell's center pixel against the table. Offline; one upload of the PNG then serves every mesh."
    from mif_roblox import png as P
    if not isinstance(out_dir, str) or not out_dir:
        return _no("out_dir is a folder path")
    if not isinstance(name, str) or not re.match(r"^[A-Za-z0-9_\-]{1,64}$", name):
        return _no("name is 1-64 letters, digits, _ or -")
    if not (_num(grime) and 0 <= grime <= 0.3):
        return _no("grime is 0-0.3")
    spec, err = K.atlas_spec(atlas)
    if err:
        return _no(err)
    if spec["pixels"] % spec["grid"]:
        return _no("pixels (%d) must divide into grid (%d) whole cells" % (spec["pixels"], spec["grid"]))
    cmap = {}
    for key, rgb in (colors or {}).items():
        mt = re.match(r"^(\d+),(\d+)$", str(key))
        if not mt or not (isinstance(rgb, list) and len(rgb) == 3 and all(isinstance(v, int) and 0 <= v <= 255 for v in rgb)):
            return _no("colors is {\"col,row\": [r, g, b]} with 0-255 whole numbers, got %r: %r" % (key, rgb))
        c = (int(mt.group(1)), int(mt.group(2)))
        if not (0 <= c[0] < spec["grid"] and 0 <= c[1] < spec["grid"]):
            return _no("colors cell %s is outside the %d-cell grid" % (key, spec["grid"]))
        cmap[c] = rgb
    # WRITE ROOTS (S13, 2026-11-22): out_dir must be the project, scratch (the temp folder) or a folder the person lists
    import tempfile
    import mif_write_roots as W
    refused = W.check(os.path.join(out_dir, name + ".png"), project=None, scratch=[tempfile.gettempdir()])
    if refused:
        return refused
    color, mask, table = K.make_atlas(spec, cmap, grime=grime, seed=seed)
    os.makedirs(out_dir, exist_ok=True)
    paths = {"color": os.path.join(out_dir, name + ".png"), "emissiveMask": os.path.join(out_dir, name + "_emissive.png"),
             "cells": os.path.join(out_dir, name + "_cells.json")}
    px = spec["pixels"]
    with open(paths["color"], "wb") as fh:
        fh.write(P.encode_rgba(px, px, color))
    with open(paths["emissiveMask"], "wb") as fh:
        fh.write(P.encode_rgba(px, px, mask))
    with open(paths["cells"], "w", encoding="utf-8") as fh:
        json.dump(table, fh, indent=1)
    # read back what is on disk
    problems = []
    w, h, rgba = P.decode_rgba(open(paths["color"], "rb").read())
    mw, mh, mrgba = P.decode_rgba(open(paths["emissiveMask"], "rb").read())
    back = json.load(open(paths["cells"], encoding="utf-8"))
    if (w, h) != (px, px) or (mw, mh) != (px, px):
        problems.append("the PNGs read back %dx%d and %dx%d, not %d" % (w, h, mw, mh, px))
    cw = px // spec["grid"]
    for cell in back["cells"]:
        x, y = cell["col"] * cw + cw // 2, cell["row"] * cw + cw // 2
        i = (y * px + x) * 4
        got = list(rgba[i:i + 3])
        if not grime and got != cell["rgb"]:
            problems.append("cell %d,%d reads %s, the table says %s" % (cell["col"], cell["row"], got, cell["rgb"]))
        if (mrgba[i] == 255) != cell["emissive"]:
            problems.append("cell %d,%d's emissive mask reads %d" % (cell["col"], cell["row"], mrgba[i]))
    out = {"ok": not problems, "files": paths, "grid": spec["grid"], "pixels": px, "marginPx": spec["marginPx"],
           "cells": len(back["cells"]), "emissiveCells": back["emissive"], "verified": not problems}
    if problems:
        out["error"] = "; ".join(problems[:5])
    return out


# -------------------------------------------------------------------------------------- the kit in a live place
def _ref_regex(name):
    return re.compile(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])")


def script_references(sources, names):
    """Where each name appears in a script as a whole token (a string literal, a .Name index, FindFirstChild("..."))
    and where its stem (the name without trailing digits) opens a string that is concatenated - a name built at run time.
    sources [{path, source}] -> {name: [{script, line, text, kind}]}."""
    out = {}
    lines = [(s.get("path"), i + 1, ln) for s in sources if isinstance(s.get("source"), str)
             for i, ln in enumerate(s["source"].split("\n"))]
    for name in names:
        rx = _ref_regex(name)
        stem = re.sub(r"[\d_\-]+$", "", name)
        srx = re.compile(r"[\"']" + re.escape(stem) + r"[^\"']*[\"']\s*\.\.") if stem and stem != name else None
        hits = []
        for path, no, text in lines:
            if rx.search(text):
                hits.append({"script": path, "line": no, "text": text.strip()[:200], "kind": "name"})
            elif srx and srx.search(text):
                hits.append({"script": path, "line": no, "text": text.strip()[:200], "kind": "built-name"})
        if hits:
            out[name] = hits
    return out


def _read_parts(root_parts, limit):
    """[{path, name, cframe, size}] for every BasePart under root, or (None, failure)."""
    T, G, V = _T(), _G(), _T().V
    cat, err = G._cat()
    if err:
        return None, err
    paths, page = [], 0
    while True:
        q = T._rbx("query", root=root_parts, page=page, pageSize=500)
        if not q.get("ok"):
            return None, q
        for m in q.get("matches") or []:
            c = m.get("className")
            if c and c != "Terrain" and G._is_a(cat, c, "BasePart"):
                paths.append(m["path"])
        if not q.get("more") or len(paths) >= limit:
            break
        page += 1
    if len(paths) > limit:
        return None, _no("more than %d parts under %s; pass a narrower root or a higher limit" % (limit, V.join_path(root_parts)))
    if not paths:
        return [], None
    rows, route = G._many([{"target": p} for p in paths], props=["Name", "CFrame", "Size"])
    if rows is None:
        return None, route
    parts = []
    for p, row in zip(paths, rows):
        if row.get("error"):
            return None, _no("%s did not read: %s" % (V.join_path(p), row["error"]))
        v = row.get("values") or {}
        cf = G._cf(v.get("CFrame"))
        size = G._v3(v.get("Size"))
        if not cf or not size:
            return None, _no("%s's CFrame or Size did not read" % V.join_path(p))
        parts.append({"path": V.join_path(p), "pathParts": p, "name": v.get("Name"),
                      "cframe": cf["position"] + cf["rotation"], "size": size})
    return parts, None


def _fit_png(path, max_px):
    """(path, [w, h], error): a PNG no larger than max_px on its longest side - the file itself, or a copy taken every
    k-th pixel (k a power of two) written to the temp folder. An EditableImage map costs memory per pixel, and a whole
    kit's maps at 4096 is what a preview cannot afford."""
    import tempfile
    from mif_roblox import png as P
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError as e:
        return None, None, "%s did not read: %s" % (path, e)
    hd = P.header(data)
    if not hd:
        return None, None, "%s is not a PNG" % path
    w, h = hd[:2]
    if max(w, h) <= max_px:
        return path, [w, h], None
    try:
        w, h, rgba = P.decode_rgba(data)
    except ValueError as e:
        return None, None, "%s did not decode: %s" % (path, e)
    k = 1
    while max(w, h) // k > max_px:
        k *= 2
    nw, nh, stride = w // k, h // k, w * 4
    rows = []
    for y in range(nh):
        row = memoryview(rgba[y * k * stride:y * k * stride + stride]).cast("I")
        rows.append(row[::k].tobytes()[:nw * 4])
    out_dir = os.path.join(tempfile.gettempdir(), "mif_kit_preview")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "%s_%d.png" % (os.path.splitext(os.path.basename(path))[0], nw))
    with open(out, "wb") as fh:
        fh.write(P.encode_rgba(nw, nh, b"".join(rows)))
    return out, [nw, nh], None


def _preview_templates(m, base, folder, max_px, dry_run, replace):
    """((templates {key: path parts}, report {key: ...}), error): rbx_kit_apply's preview - one EditableMesh template per
    manifest mesh, built from its exported FBX through fbx_mesh.preview_payload, its maps on a SurfaceAppearance as
    EditableImages. dry_run reads every file and builds
    nothing."""
    T = _T()
    V = T.V
    fparts = V.split_path(folder) if isinstance(folder, str) else None
    if not fparts or len(fparts) < 2:
        return None, _no("preview is the path of a folder for the templates, e.g. Workspace.MifScratch.KitPreview")
    if not (isinstance(max_px, int) and not isinstance(max_px, bool) and 64 <= max_px <= 4096):
        return None, _no("preview_max_px is 64-4096")
    payloads, rows = {}, {}
    for key, mesh in sorted(m["meshes"].items()):
        f = mesh.get("file")
        if not f:
            return None, _no("meshes.%s has no file: a preview builds each template from its exported FBX" % key)
        path = f if os.path.isabs(f) or not base else os.path.join(base, f)
        try:
            sc = F.load(path)
        except (OSError, ValueError) as e:
            return None, _no("meshes.%s.file %s did not read: %s" % (key, path, e))
        if not sc.meshes:
            return None, _no("meshes.%s.file %s holds no mesh" % (key, path))
        pl = F.preview_payload(sc)
        if pl is None:
            return None, _no("meshes.%s.file %s has axis settings %s, not the ones Studio's import was measured on "
                             "(RB1156): its preview would not match its upload" % (key, path, sc.axis_settings))
        if pl["triangleCount"] > TRIANGLE_LIMIT:
            return None, _no("meshes.%s.file %s has %d triangles, over Roblox's %d per mesh" % (
                key, path, pl["triangleCount"], TRIANGLE_LIMIT))
        maps = {}
        for mk, mp in sorted((mesh.get("maps") or {}).items()):
            mpath = mp if os.path.isabs(mp) or not base else os.path.join(base, mp)
            if not os.path.isfile(mpath):
                return None, _no("meshes.%s.maps.%s %s is not a file" % (key, mk, mpath))
            maps[mk] = mpath
        payloads[key] = (path, pl, maps)
        rows[key] = {"file": path, "triangles": pl["triangleCount"], "fileMeshes": pl["meshes"],
                     "path": V.join_path(fparts + [key]), "maps": sorted(maps)}
    templates = {key: fparts + [key] for key in payloads}
    if dry_run:
        return (templates, rows), None
    if not T._rbx("resolve", target=fparts).get("ok"):
        mk = T.rbx_create("Folder", V.join_path(fparts[:-1]), name=fparts[-1])
        if not mk.get("ok"):
            return None, dict(mk, ok=False, ran=False, error="the preview folder %s: %s" % (folder, mk.get("error")))
    clash = [V.join_path(t) for t in templates.values() if T._rbx("resolve", target=t).get("ok")]
    if clash and not replace:
        return None, _no("%d preview template(s) already there (%s): pass replace true to build them again" % (
            len(clash), ", ".join(clash[:3])), refusedRule="exists", existing=clash)
    for c in clash:
        d = T.rbx_destroy(c)
        if not d.get("ok"):
            return None, dict(d, ok=False, error="removing the old template %s: %s" % (c, d.get("error")))
    for i, (key, (path, pl, maps)) in enumerate(sorted(payloads.items())):
        # out of the way of the model being dressed: a row 2,000 studs up, one mesh per 64 studs
        r = T.rbx_mesh_preview(pl["vertices"], pl["triangles"], name=key, parent=V.join_path(fparts),
                               position=[64.0 * i, 2000.0, 0.0], corner_uvs=pl["cornerUVs"],
                               corner_normals=pl["cornerNormals"])
        rows[key]["mesh"] = {k: r.get(k) for k in ("ok", "faces", "meshSize", "expectedSize", "error")}
        if not r.get("ok"):
            return None, _no("the template for %s did not build: %s" % (key, r.get("error")), preview=rows)
        if maps:
            fitted = {}
            for mk, mp in maps.items():
                fp, size, ferr = _fit_png(mp, max_px)
                if ferr:
                    return None, _no("meshes.%s.maps.%s: %s" % (key, mk, ferr), preview=rows)
                fitted[mk] = fp
                rows[key].setdefault("mapSizes", {})[mk] = size
            sa = T.rbx_surface_appearance(V.join_path(templates[key]), ready_wait=0, **fitted)
            rows[key]["surfaceAppearance"] = {k: sa.get(k) for k in ("ok", "verified", "path", "error")}
            if not sa.get("ok"):
                return None, _no("the maps for %s did not load: %s" % (key, sa.get("error")), preview=rows)
    return (templates, rows), None


def _template_maps(T, templates, clear=False):
    """domain_assets.broken_maps over every template, summed: {ok, checked, templates, surfaceAppearances, broken,
    cleared?, warning?, error?}."""
    from mif_roblox import domain_assets as A  # noqa: PLC0415 - the map check lives with the asset reads
    out = {"ok": True, "checked": True, "templates": len(templates), "surfaceAppearances": 0, "broken": []}
    errors = []
    for tp in templates.values():
        r = A.broken_maps(T, tp, clear=clear)
        out["surfaceAppearances"] += r.get("surfaceAppearances") or 0
        out["broken"] += r.get("broken") or []
        if "cleared" in r:
            out.setdefault("cleared", []).extend(r["cleared"])
        if not r.get("ok"):
            errors.append(r.get("error"))
        if not r.get("checked"):
            out["checked"] = False
    if errors:
        out.update(ok=False, error="; ".join(str(e) for e in errors))
    if out["broken"]:
        out["warning"] = ("%d map(s) on the templates name rbxassetid://6, the broken emissive mask an FBX upload with an "
                          "emissive texture leaves; every copy would carry it and play clients draw those meshes grey "
                          "(RB2766). %s" % (len(out["broken"]), "Cleared on the templates before copying, read back "
                                            "empty: %d." % len(out.get("cleared") or []) if clear else
                                            "clear_broken_maps true clears them on the templates before copying"))
    return out


def rbx_kit_apply(root: Any, manifest: Any, dry_run: bool = False, replace: bool = False, grey: str = None,
                  limit: int = MAX_PARTS, preview: str = None, preview_max_px: int = PREVIEW_MAX_PX,
                  clear_broken_maps: bool = False) -> dict:
    "Roblox Studio: dress a built model with a kit: for every BasePart under root whose Name matches a manifest rule (Lua patterns, first rule wins), a copy of the rule's mesh template (a BasePart already in the place, usually a MeshPart) is put in the part's parent at part.CFrame * CFrame.Angles(0, yaw, 0) * CFrame.new(offset) with the mesh's size - yaw fixed or per side from the pattern's first capture; tile rules split the part into max(1, round(len / tileSize)) cells per tiled axis and scale each copy by cell / nominal - Anchored, CanCollide, CanQuery and CanTouch false, named <part>_Kit (tiles <part>_Kit1..n). The grey part is then hidden (Transparency 1, its collision kept), kept, or deleted (grey per rule, or this call's grey for all); a part any script names (rbx's script_scan: the name as a whole token, or its stem opening a concatenated string) is never deleted - it is hidden and its references listed. Reports parts matched, parts no rule matched and rules that matched nothing. dry_run answers the plan and writes nothing. An existing <part>_Kit is refused unless replace. Every write is read back (positions to 1e-3 studs). Full mode, or inside Workspace.MifScratch. preview (a folder path, made if missing): before any upload, each mesh's template is built there from its exported FBX (meshes.<key>.file) as an EditableMesh MeshPart, with meshes.<key>.maps PNGs (color, normal, roughness, metalness; downscaled to preview_max_px) on a SurfaceAppearance as EditableImages, and the same manifest dresses the model with them; nothing is uploaded or saved. Without preview, every template's SurfaceAppearance maps are read before anything is written: one naming rbxassetid://6 (the broken emissive mask an FBX upload with an emissive texture leaves, which every copy would carry into play clients, drawn grey: RB2766) is answered under brokenMaps with a warning, and clear_broken_maps true sets it to an empty Content on the template, read back empty, before the first copy (the dry run only reports)."
    T, G = _T(), _G()
    V = T.V
    if grey is not None and grey not in K.GREY_MODES:
        return _no("grey is one of %s, or omitted to use each rule's" % ", ".join(K.GREY_MODES))
    if not isinstance(dry_run, bool) or not isinstance(replace, bool):
        return _no("dry_run and replace are true or false")
    if not (isinstance(limit, int) and not isinstance(limit, bool) and 1 <= limit <= MAX_PARTS):
        return _no("limit is 1-%d" % MAX_PARTS)
    m, _base, err = K.read_manifest(manifest)
    if err:
        return _no(err)
    probs = K.check_placement_manifest(m)
    if probs:
        return _no("the manifest is not usable: " + "; ".join(probs), problems=probs)
    rparts = V.split_path(root) if isinstance(root, str) else None
    if not rparts:
        return _no("root is the path of the built model")
    templates, previewed = {}, None
    if preview is not None:
        built, perr = _preview_templates(m, _base, preview, preview_max_px, dry_run, replace)
        if perr:
            return perr
        templates, previewed = built
    for key, mesh in m["meshes"].items():
        if preview is not None:
            break
        if not mesh.get("template"):
            return _no("meshes.%s has no template: rbx_kit_apply copies a part already in the place (upload the mesh "
                       "and place one with rbx_asset_place first)" % key)
        tp = V.split_path(mesh["template"])
        r = T._rbx("resolve", target=tp)
        if not r.get("ok"):
            return _no("meshes.%s.template %s is not in the place: %s" % (key, mesh["template"], r.get("error")))
        cat, cerr = G._cat()
        if cerr:
            return cerr
        if not G._is_a(cat, r.get("className") or "", "BasePart"):
            return _no("meshes.%s.template %s is a %s; a template is a BasePart" % (key, mesh["template"], r.get("className")))
        templates[key] = tp
    if not isinstance(clear_broken_maps, bool):
        return _no("clear_broken_maps is true or false")
    maps = None
    if preview is None:
        # a template placed from an FBX upload with an emissive texture carries the broken rbxassetid://6 mask, and every
        # copy would carry it into play clients, drawn grey (RB2766): read before anything is written
        maps = _template_maps(T, templates)
        if not maps["ok"]:
            return _no("the templates' SurfaceAppearance maps did not read: %s" % maps.get("error"), brokenMaps=maps)
    parts, err = _read_parts(rparts, limit)
    if err:
        return err
    tpl_paths = {V.join_path(p) for p in templates.values()}
    parts = [p for p in parts if p["path"] not in tpl_paths and not re.search(r"_Kit\d*$", p["name"] or "")]
    if not parts:
        return _no("no parts under %s: nothing to dress" % V.join_path(rparts), scanned=0)
    pl = K.plan(parts, m)
    report = {"root": V.join_path(rparts), "scanned": len(parts), "matched": pl.get("matched", 0),
              "preview": previewed,
              "unmatched": pl.get("unmatched", []), "rulesUnmatched": pl.get("rulesUnmatched", []),
              "refused": pl.get("refused", []), "placements": len(pl.get("placements") or [])}
    if maps is not None:
        report["brokenMaps"] = maps
        if maps.get("warning"):
            report["warning"] = maps["warning"]
    if not pl.get("ok"):
        return dict(report, ok=False, ran=False, error=pl.get("error"))
    by_path = {p["path"]: p for p in parts}
    greys = {}
    for pm in pl["placements"]:
        greys.setdefault(pm["part"], grey or pm["grey"])
    # who names the parts that would be deleted
    refs = {}
    to_delete = [p for p, g in greys.items() if g == "delete"]
    if to_delete:
        scan = T._rbx("script_scan", root=[])
        if not scan.get("ok"):
            return dict(report, ok=False, ran=False, error="the script scan the delete needs failed: %s" % scan.get("error"))
        srcs = [{"path": V.join_path(s.get("path")), "source": s.get("source")} for s in scan.get("sources") or []]
        refs = script_references(srcs, sorted({by_path[p]["name"] for p in to_delete}))
        report["scriptsScanned"] = scan.get("scanned", len(srcs))
    actions = {}
    for p, g in greys.items():
        if g == "delete" and by_path[p]["name"] in refs:
            actions[p] = "hide"
        else:
            actions[p] = g
    report["referenced"] = [{"part": p, "references": refs[by_path[p]["name"]]} for p in to_delete
                            if by_path[p]["name"] in refs]
    report["grey"] = {a: sum(1 for v in actions.values() if v == a) for a in K.GREY_MODES}
    if dry_run:
        return dict(report, ok=True, ran=False, dryRun=True, plan=pl["placements"][:200],
                    planTruncated=len(pl["placements"]) > 200, greyActions=actions)
    # an existing copy of a planned name
    parent_of = {p["path"]: p["pathParts"][:-1] for p in parts}
    clash = []
    for pm in pl["placements"]:
        target = parent_of[pm["part"]] + [pm["name"]]
        if T._rbx("resolve", target=target).get("ok"):
            clash.append(V.join_path(target))
    if clash and not replace:
        return dict(report, ok=False, ran=False, refusedRule="exists", existing=clash[:20],
                    error="%d kit mesh(es) already in the place (%s...): pass replace true to put new ones in their "
                          "place" % (len(clash), ", ".join(clash[:3])))
    # rbx_batch checks every target before the first op runs, so a copy's props cannot ride in the batch that makes it
    # (measured live: "has no child named Floor_1_Kit1"). Copies are made a batch at a time, then their props; a props
    # batch that fails removes the copies its clone batch made, so nothing is left at the template's place.
    def run(ops):
        for i in range(0, len(ops), BATCH_OPS):
            r = T.rbx_batch(ops[i:i + BATCH_OPS])
            if not r.get("ok"):
                return r, i
        return None, len(ops)

    if clear_broken_maps and maps and maps["broken"]:
        maps = _template_maps(T, templates, clear=True)
        report["brokenMaps"], report["warning"] = maps, maps.get("warning")
        if not maps["ok"]:
            return dict(report, ok=False, ran=True, opsDone=0, error="clearing the templates' broken maps failed, nothing "
                        "was copied: %s" % maps.get("error"))
    done = 0
    if clash:
        bad, n = run([{"tool": "rbx_destroy", "args": {"target": c}} for c in clash])
        if bad:
            return dict(report, ok=False, ran=True, opsDone=n, error="removing the old copies failed: %s" % bad.get("error"))
        done += n
    per = BATCH_OPS
    for i in range(0, len(pl["placements"]), per):
        chunk = pl["placements"][i:i + per]
        clones, props = [], []
        for pm in chunk:
            parent = V.join_path(parent_of[pm["part"]])
            tgt = V.join_path(parent_of[pm["part"]] + [pm["name"]])
            clones.append({"tool": "rbx_clone", "args": {"target": V.join_path(templates[pm["mesh"]]), "parent": parent,
                                                         "name": pm["name"]}})
            props.append({"tool": "rbx_set_props", "args": {"target": tgt, "props": {
                "CFrame": {"position": pm["cframe"][:3], "rotation": pm["cframe"][3:]}, "Size": pm["size"],
                "Anchored": True, "CanCollide": False, "CanQuery": False, "CanTouch": False}}})
        bad, n = run(clones)
        if bad:
            return dict(report, ok=False, ran=True, opsDone=done, error="a clone batch failed: %s" % bad.get("error"),
                        batch=bad)
        done += n
        bad, n = run(props)
        if bad:
            run([{"tool": "rbx_destroy", "args": {"target": c["args"]["target"]}} for c in props])
            return dict(report, ok=False, ran=True, opsDone=done, error="a props batch failed, its copies were removed: %s"
                        % bad.get("error"), batch=bad)
        done += n
    greyops = []
    for p, a in sorted(actions.items()):
        if a == "hide":
            greyops.append({"tool": "rbx_set_props", "args": {"target": p, "props": {"Transparency": 1}}})
        elif a == "delete":
            greyops.append({"tool": "rbx_destroy", "args": {"target": p}})
    bad, n = run(greyops) if greyops else (None, 0)
    if bad:
        return dict(report, ok=False, ran=True, opsDone=done, error="the grey parts' batch failed: %s" % bad.get("error"))
    done += n
    # read back every mesh and every grey part
    targets = [parent_of[pm["part"]] + [pm["name"]] for pm in pl["placements"]]
    rows, route = G._many([{"target": t} for t in targets], props=["CFrame", "Size", "CanCollide", "Anchored"])
    if rows is None:
        return dict(report, ok=False, ran=True, error="the read-back failed: %s" % route)
    bad = []
    for pm, row in zip(pl["placements"], rows):
        v = row.get("values") or {}
        cf, size = G._cf(v.get("CFrame")), G._v3(v.get("Size"))
        if row.get("error") or not cf or not size:
            bad.append("%s did not read back: %s" % (pm["name"], row.get("error")))
            continue
        got = cf["position"] + cf["rotation"]
        err = max(abs(a - b) for a, b in zip(got, pm["cframe"]))
        serr = max(abs(a - b) for a, b in zip(size, pm["size"]))
        if err > POS_TOL or serr > POS_TOL or v.get("CanCollide") is not False:
            bad.append("%s reads CFrame off by %.4g, Size off by %.4g, CanCollide %s" % (pm["name"], err, serr,
                                                                                       v.get("CanCollide")))
    hidden = [p for p, a in actions.items() if a == "hide"]
    deleted = [p for p, a in actions.items() if a == "delete"]
    if hidden:
        hrows, _r = G._many([{"target": by_path[p]["pathParts"]} for p in hidden], props=["Transparency"])
        for p, row in zip(hidden, hrows or []):
            if (row.get("values") or {}).get("Transparency") != 1:
                bad.append("%s was to be hidden and reads Transparency %s" % (p, (row.get("values") or {}).get("Transparency")))
    for p in deleted:
        if T._rbx("resolve", target=by_path[p]["pathParts"]).get("ok"):
            bad.append("%s was to be deleted and is still there" % p)
    out = dict(report, ok=not bad, ran=True, verified=not bad, placed=len(targets), replaced=len(clash),
               hidden=len(hidden), deleted=len(deleted), readRoute=route, opsDone=done)
    if bad:
        out["error"] = "%d read-back problem(s): %s" % (len(bad), "; ".join(bad[:5]))
    return out


# ------------------------------------------------------------------------------------- the rules as a Luau module
LUAU_TEMPLATE = r'''--!strict
-- Kit placement rules, generated by MifBridge rbx_kit_rules_module from a kit manifest. Do not edit: regenerate.
-- Each rule: the first whose Lua pattern matches a part's Name dresses it. mesh.CFrame = part.CFrame * yaw * offset.
local Rules = {}

Rules.meshes = %(meshes)s
Rules.rules = %(rules)s

local function yawFor(rule, name: string): number?
	if rule.yawBySide then
		local side = string.match(name, rule.pattern)
		return side and rule.yawBySide[side] or nil
	end
	return rule.yaw or 0
end

-- Placements for one part: {name, cframe, size, mesh} each, or nil and why.
function Rules.placements(part: BasePart): ({any}?, string?)
	for _, rule in ipairs(Rules.rules) do
		if string.match(part.Name, rule.pattern) then
			local mesh = Rules.meshes[rule.mesh]
			local yaw = yawFor(rule, part.Name)
			if yaw == nil then
				return nil, "no yaw for this side"
			end
			local offset = rule.offset or mesh.offset or {0, 0, 0}
			local size = rule.size or mesh.size
			local ycf = CFrame.Angles(0, math.rad(yaw), 0)
			if not rule.tile then
				return {{name = part.Name .. "_Kit", mesh = rule.mesh, grey = rule.grey,
					cframe = part.CFrame * ycf * CFrame.new(offset[1], offset[2], offset[3]),
					size = Vector3.new(size[1], size[2], size[3])}}, nil
			end
			local nominal = mesh.nominal or size
			local psize = {part.Size.X, part.Size.Y, part.Size.Z}
			local counts, cells, tiled = {1, 1, 1}, {psize[1], psize[2], psize[3]}, {}
			for _, a in ipairs(rule.tile.axes) do
				local k = (a == "X" and 1) or (a == "Y" and 2) or 3
				tiled[k] = true
				counts[k] = math.max(1, math.floor(psize[k] / rule.tile.tileSize + 0.5))
				cells[k] = psize[k] / counts[k]
			end
			local scale = {1, 1, 1}
			local axes = {ycf.RightVector, ycf.UpVector, -ycf.LookVector}
			for a = 1, 3 do
				local v = axes[a]
				local comps = {math.abs(v.X), math.abs(v.Y), math.abs(v.Z)}
				for k = 1, 3 do
					if comps[k] > 0.999 and tiled[k] then
						scale[a] = cells[k] / nominal[a]
					end
				end
			end
			local out, i = {}, 0
			for ix = 0, counts[1] - 1 do
				for iy = 0, counts[2] - 1 do
					for iz = 0, counts[3] - 1 do
						local idx = {ix, iy, iz}
						local c = {0, 0, 0}
						for k = 1, 3 do
							if tiled[k] then
								c[k] = -psize[k] / 2 + (idx[k] + 0.5) * cells[k]
							end
						end
						i += 1
						table.insert(out, {name = part.Name .. "_Kit" .. i, mesh = rule.mesh, grey = rule.grey,
							cframe = part.CFrame * CFrame.new(c[1], c[2], c[3]) * ycf
								* CFrame.new(offset[1] * scale[1], offset[2] * scale[2], offset[3] * scale[3]),
							size = Vector3.new(size[1] * scale[1], size[2] * scale[2], size[3] * scale[3])})
					end
				end
			end
			return out, nil
		end
	end
	return nil, "no rule matches"
end

-- Dress one part from templates (a folder of meshes named by the manifest's mesh keys). Returns the copies made.
function Rules.apply(part: BasePart, templates: Instance): {BasePart}
	local made = {}
	local list = Rules.placements(part)
	if not list then
		return made
	end
	for _, p in ipairs(list) do
		local template = templates:FindFirstChild(p.mesh)
		if template and template:IsA("BasePart") then
			local copy = template:Clone()
			copy.Name = p.name
			copy.Anchored = true
			copy.CanCollide = false
			copy.CanQuery = false
			copy.CanTouch = false
			copy.Size = p.size
			copy.CFrame = p.cframe
			copy.Parent = part.Parent
			table.insert(made, copy)
		end
	end
	if #made > 0 and (list[1].grey or "hide") ~= "keep" then
		part.Transparency = 1
	end
	return made
end

return Rules
'''


def _lua(v, indent=1):
    """A JSON-able value as a Luau literal."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "nil"
    if isinstance(v, (int, float)):
        return repr(float(v)) if isinstance(v, float) else str(v)
    if isinstance(v, str):
        return json.dumps(v)
    tab = "\t" * indent
    if isinstance(v, list):
        return "{" + ", ".join(_lua(x, indent + 1) for x in v) + "}"
    items = []
    for k in sorted(v):
        key = k if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", k) else "[%s]" % json.dumps(k)
        items.append("%s%s = %s," % (tab, key, _lua(v[k], indent + 1)))
    return "{\n" + "\n".join(items) + "\n" + "\t" * (indent - 1) + "}"


def rbx_kit_rules_module(manifest: Any, target: str = None, confirm_overwrite: bool = False) -> dict:
    "Roblox: a kit manifest's placement rules as a Luau ModuleScript for parts a game builds at run time (dress build-time parts as they are built): Rules.placements(part) answers each mesh's name, CFrame and Size by the same arithmetic as rbx_kit_apply (first matching Lua pattern, yaw or yaw per side from the first capture, offset, tiles), and Rules.apply(part, templatesFolder) clones the templates (named by mesh key) into the part's parent, non-colliding, and hides the part. Without target the source is returned; with target (a ModuleScript path, created when absent) it is written with rbx_script_write and read back byte for byte. An existing script is overwritten only with confirm_overwrite."
    T = _T()
    m, _base, err = K.read_manifest(manifest)
    if err:
        return _no(err)
    probs = K.check_placement_manifest(m)
    if probs:
        return _no("the manifest is not usable: " + "; ".join(probs), problems=probs)
    meshes = {k: {kk: v[kk] for kk in ("size", "offset", "nominal") if kk in v} for k, v in m["meshes"].items()}
    rules = []
    for r in m["rules"]:
        row = {k: r[k] for k in ("pattern", "mesh", "yaw", "yawBySide", "offset", "size", "tile", "grey") if k in r}
        rules.append(row)
    src = LUAU_TEMPLATE % {"meshes": _lua(meshes), "rules": _lua(rules)}
    out = {"ok": True, "rules": len(rules), "meshes": len(meshes), "bytes": len(src)}
    if target is None:
        return dict(out, source=src, written=False)
    if not isinstance(target, str) or not T.V.split_path(target):
        return _no("target is the path of a ModuleScript")
    parts = T.V.split_path(target)
    exists = T._rbx("resolve", target=parts)
    if exists.get("ok"):
        if exists.get("className") != "ModuleScript":
            return _no("%s is a %s, not a ModuleScript" % (target, exists.get("className")))
        if not confirm_overwrite:
            return _no("%s already exists: pass confirm_overwrite true to replace its source" % target,
                       refusedRule="exists")
    else:
        made = T.rbx_create("ModuleScript", T.V.join_path(parts[:-1]), name=parts[-1])
        if not made.get("ok"):
            return dict(made, ok=False, step="create")
    w = T.rbx_script_write(target, src)
    if not w.get("ok"):
        return dict(w, ok=False, step="write")
    back = T.rbx_script_read(target)
    same = back.get("ok") and back.get("source") == src
    return dict(out, ok=bool(same), written=True, target=target, verified=bool(same),
                error=None if same else "the module read back differently: %s" % (back.get("error") or "source differs"))


# ------------------------------------------------------------------------------------------- the part-kit check
PART_KEYS = {"slot", "file", "size", "center", "sockets"}
ASSEMBLY_KEYS = {"socket", "roll", "root"}


def _socket_cf(s):
    """A socket: [x, y, z] or {position, yaw?, roll?} -> (R, p), or None."""
    if isinstance(s, list) and len(s) == 3 and all(_num(v) for v in s):
        return K.cf_new(s)
    if isinstance(s, dict) and isinstance(s.get("position"), list) and len(s["position"]) == 3:
        cf = K.cf_new(s["position"])
        if _num(s.get("yaw")):
            cf = K.cf_mul(cf, K.cf_yaw(s["yaw"]))
        if _num(s.get("roll")):
            cf = K.cf_mul(cf, K.cf_roll(s["roll"]))
        return cf
    return None


def kit_check(m, base, scenes=None):
    """The part-kit check over a loaded manifest. scenes {file: fbx_mesh.Scene} (read here when None). -> dict."""
    parts, assembly = m.get("parts"), m.get("assembly")
    probs = []
    if not isinstance(parts, dict) or not parts:
        return {"ok": False, "error": "parts is a non-empty {key: {slot, file, size, center, sockets}} map"}
    if not isinstance(assembly, dict) or not assembly:
        return {"ok": False, "error": "assembly is a non-empty {slot: {root: true} | {socket: \"<slot>.<socket>\", roll?}} map"}
    for k, p in parts.items():
        if not isinstance(p, dict) or set(p) - PART_KEYS:
            probs.append("parts.%s is {%s}" % (k, ", ".join(sorted(PART_KEYS))))
            continue
        if not isinstance(p.get("slot"), str):
            probs.append("parts.%s.slot is a slot name" % k)
        for key in ("size", "center"):
            if not K._vec3(p.get(key)):
                probs.append("parts.%s.%s is three numbers (Roblox axes, studs)" % (k, key))
        for sn, s in (p.get("sockets") or {}).items():
            if _socket_cf(s) is None:
                probs.append("parts.%s.sockets.%s is [x, y, z] or {position, yaw?, roll?}" % (k, sn))
    slots = {}
    for k, p in parts.items():
        if isinstance(p, dict) and isinstance(p.get("slot"), str):
            slots.setdefault(p["slot"], []).append(k)
    roots = [s for s, a in assembly.items() if isinstance(a, dict) and a.get("root")]
    if len(roots) != 1:
        probs.append("assembly names exactly one root slot ({root: true}); it names %d" % len(roots))
    for s, a in assembly.items():
        if not isinstance(a, dict) or set(a) - ASSEMBLY_KEYS:
            probs.append("assembly.%s is {root: true} or {socket: \"<slot>.<socket>\", roll?}" % s)
            continue
        if s not in slots:
            probs.append("assembly.%s names a slot no part fills" % s)
        if not a.get("root"):
            sock = a.get("socket")
            if not (isinstance(sock, str) and "." in sock):
                probs.append("assembly.%s.socket is \"<slot>.<socket>\"" % s)
            elif sock.split(".", 1)[0] not in assembly:
                probs.append("assembly.%s mounts on slot %s, which is not in the assembly" % (s, sock.split(".", 1)[0]))
            if "roll" in a and not _num(a["roll"]):
                probs.append("assembly.%s.roll is degrees" % s)
    if probs:
        return {"ok": False, "error": "the kit manifest is not usable: " + "; ".join(probs), "problems": probs}
    fails = []
    # sockets: every part of the parent slot carries the socket each mounted slot names
    missing = []
    for s, a in sorted(assembly.items()):
        if a.get("root"):
            continue
        pslot, sname = a["socket"].split(".", 1)
        for pk in slots.get(pslot, []):
            if sname not in (parts[pk].get("sockets") or {}):
                missing.append({"part": pk, "socket": sname, "for": s})
    if missing:
        fails.append("%d socket(s) missing: %s" % (len(missing), ", ".join("%s.%s" % (x["part"], x["socket"]) for x in missing[:8])))
    # slot boxes within the tolerance
    tol = float(m.get("slotTolerance", SLOT_TOL))
    spread = {}
    for s, keys in sorted(slots.items()):
        sizes = [parts[k]["size"] for k in keys]
        ax = []
        for i in range(3):
            lo, hi = min(v[i] for v in sizes), max(v[i] for v in sizes)
            ax.append(round(hi / lo - 1.0, 4) if lo > 0 else None)
        ok = all(a is not None and a <= tol for a in ax)
        spread[s] = {"parts": keys, "spread": ax, "pass": ok}
        if not ok:
            fails.append("slot %s's parts differ by %s per axis, over %g%%" % (s, ax, tol * 100))
    # files: triangles, and each file's box against the manifest's size and center
    stol = float(m.get("sizeTolerance", SIZE_TOL))
    files, tris = {}, {}
    scenes = scenes if scenes is not None else {}
    for k, p in sorted(parts.items()):
        f = p.get("file")
        if not f:
            fails.append("parts.%s names no file: its triangles and box were not read" % k)
            continue
        path = f if os.path.isabs(f) or not base else os.path.join(base, f)
        try:
            sc = scenes.get(path) or F.load(path)
            scenes[path] = sc
        except (OSError, ValueError) as e:
            fails.append("parts.%s: %s did not read (%s)" % (k, f, e))
            continue
        size, center = F.scene_rbx_box(sc)
        tris[k] = sum(mm["triangles"] for mm in sc.meshes)
        scale = max(max(abs(v) for v in p["size"]), 1e-6)
        serr = max(abs(size[i] - p["size"][i]) for i in range(3))
        cerr = max(abs(center[i] - p["center"][i]) for i in range(3))
        row = {"file": path, "triangles": tris[k], "sizeStuds": size, "centerStuds": center,
               "sizeError": round(serr, 6), "centerError": round(cerr, 6), "pass": serr <= stol * scale and cerr <= stol * scale}
        files[k] = row
        if not row["pass"]:
            fails.append("parts.%s: the file's box is %s centred %s, the manifest says %s centred %s" % (
                k, size, center, p["size"], p["center"]))
    # budgets: each slot's largest part, and the assembly total
    budgets = m.get("budgets") or {}
    per_slot, total = {}, 0
    for s in sorted(assembly):
        got = [tris[k] for k in slots.get(s, []) if k in tris]
        worst = max(got) if got else None
        lim = budgets.get(s)
        ok = worst is not None and (lim is None or worst <= lim)
        per_slot[s] = {"largest": worst, "budget": lim, "pass": ok}
        if worst is not None:
            total += worst
        if lim is not None and worst is not None and worst > lim:
            fails.append("slot %s: its largest part has %d triangles, budget %d" % (s, worst, lim))
    cap = m.get("assemblyCap")
    if cap is not None and total > cap:
        fails.append("the assembly's largest combination has %d triangles, cap %d" % (total, cap))
    # combinations: named, or the i-th part of every slot
    combos = m.get("combinations")
    if combos is None:
        n = max(len(v) for v in slots.values())
        combos = {"variant%d" % (i + 1): {s: slots[s][i % len(slots[s])] for s in assembly} for i in range(n)}
    built = {}
    for cname, pick in sorted(combos.items()):
        placed, frames, cfails = [], {}, []
        order = [roots[0]] + [s for s in assembly if s != roots[0]]
        pending = list(order)
        while pending:
            progressed = False
            for s in list(pending):
                a = assembly[s]
                pk = pick.get(s)
                if pk not in parts:
                    cfails.append("%s: slot %s picks %r, not a part" % (cname, s, pk))
                    pending.remove(s)
                    progressed = True
                    continue
                if a.get("root"):
                    origin = K.cf_new([0, 0, 0])
                else:
                    pslot, sname = a["socket"].split(".", 1)
                    if pslot not in frames:
                        continue
                    sock = _socket_cf((parts[pick[pslot]].get("sockets") or {}).get(sname))
                    if sock is None:
                        cfails.append("%s: %s has no socket %s" % (cname, pick[pslot], sname))
                        pending.remove(s)
                        progressed = True
                        continue
                    origin = K.cf_mul(frames[pslot], sock)
                    if _num(a.get("roll")):
                        origin = K.cf_mul(origin, K.cf_roll(a["roll"]))
                frames[s] = origin
                cf = K.cf_mul(origin, K.cf_new(parts[pk]["center"]))   # a MeshPart is placed by its box center
                placed.append({"slot": s, "part": pk, "file": (files.get(pk) or {}).get("file"),
                               "cframe": K._round(K.cf_components(cf)), "size": parts[pk]["size"]})
                pending.remove(s)
                progressed = True
            if not progressed:
                cfails.append("%s: slots %s never reach the root (a socket chain loops or breaks)" % (cname, pending))
                break
        built[cname] = {"placements": placed, "triangles": sum(tris.get(p["part"], 0) for p in placed)}
        fails += cfails
    return {"ok": not fails, "pass": not fails, "failed": fails, "missingSockets": missing, "slots": spread,
            "files": files, "budgets": per_slot, "assemblyTriangles": total, "assemblyCap": cap,
            "combinations": built}


def rbx_kit_check(manifest: Any) -> dict:
    "Roblox: check a part kit's manifest and its exported files offline: every socket an assembly slot mounts on present on each part of its parent slot; each slot's parts within one box (slotTolerance, default 15% per axis); each part's .fbx box on Roblox axes against its manifest size and center (sizeTolerance, default 2%); each slot's largest part against budgets[slot] and the assembly total (the largest part of every slot) against assemblyCap; and every combination (combinations {name: {slot: part}}, default the i-th part of every slot) assembled from the manifest alone - part.CFrame = socket * roll * CFrame.new(center), sockets chained from the root slot - into placements bl_roblox_kit_check renders. manifest: {parts {key: {slot, file, size, center, sockets {name: [x,y,z] | {position, yaw?, roll?}}}}, assembly {slot: {root: true} | {socket: \"<slot>.<socket>\", roll?}}, budgets?, assemblyCap?, combinations?}, or a path to its JSON (files relative to it). Per-mesh checks (materials, UVs, n-gons) are rbx_mesh_preflight's."
    m, base, err = K.read_manifest(manifest)
    if err:
        return _no(err)
    r = kit_check(m, base)
    if not r.get("ok") and "error" not in r:
        r["error"] = "%d check(s) failed: %s" % (len(r["failed"]), "; ".join(r["failed"][:4]))
    return r


# -------------------------------------------------------------------------------------------- the batch upload
LOCK_SUFFIX = ".lock"
ENTRY_KEYS = {"file", "displayName", "assetType", "description", "assetId", "sha256", "meshHash", "status", "at",
              "error", "operation"}


def _pid_alive(pid):
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if os.name == "nt":
        import ctypes
        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(h)
        return code.value == 259                                      # STILL_ACTIVE
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


LOCK_TAKING_SECONDS = 10.0   # an empty or unreadable lock this young is a run between creating it and writing it


def _proc_started(pid):
    """The process's creation time as an opaque number (Windows FILETIME, Linux start ticks), or None when it is gone
    or this platform cannot say. Compared with the lock's, it tells a reused pid from the run that wrote the lock."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return None
    try:
        if os.name == "nt":
            import ctypes  # noqa: PLC0415
            k32 = ctypes.windll.kernel32
            h = k32.OpenProcess(0x1000, False, pid)                       # PROCESS_QUERY_LIMITED_INFORMATION
            if not h:
                return None
            try:
                t = [ctypes.c_ulonglong() for _ in range(4)]
                if not k32.GetProcessTimes(h, *[ctypes.byref(x) for x in t]):
                    return None
                return int(t[0].value)
            finally:
                k32.CloseHandle(h)
        with open("/proc/%d/stat" % pid, encoding="ascii", errors="replace") as fh:
            return int(fh.read().rsplit(")", 1)[1].split()[19])          # field 22, starttime
    except (OSError, ValueError, IndexError, AttributeError):
        return None


def lock_holder(lock):
    """(held dict, state) of an existing lock: state "running", "being-taken" (empty or unreadable and seconds old: the
    holder is mid-write), "pid-reused" (the pid runs, but a process started at another time) or "dead"."""
    try:
        held = json.load(open(lock, encoding="utf-8"))
    except (OSError, ValueError):
        held = None
    if not isinstance(held, dict) or held.get("pid") is None:
        try:
            age = time.time() - os.path.getmtime(lock)
        except OSError:
            age = 0.0
        return (held if isinstance(held, dict) else {}), ("being-taken" if age < LOCK_TAKING_SECONDS else "dead")
    if not _pid_alive(held.get("pid")):
        return held, "dead"
    want = held.get("started")
    if want is not None:
        now = _proc_started(held.get("pid"))
        if now is not None and now != want:
            return held, "pid-reused"
    return held, "running"


def take_lock(path):
    """Take <manifest>.lock exclusively -> (lock path, None) or (None, failure). The lock is written whole (pid, process
    creation time, time) to a file of its own and linked into place, so a second run never reads a half-written one;
    a lock whose process is gone, or whose pid now belongs to a process started later, is reported, never silently
    taken."""
    lock = path + LOCK_SUFFIX
    me = {"pid": os.getpid(), "started": _proc_started(os.getpid()), "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    tmp = "%s.%d.%d.tmp" % (lock, os.getpid(), int(time.time() * 1e6))
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(me, fh)
        try:
            os.link(tmp, lock)                                            # fails, atomically, when the lock exists
        except FileExistsError:
            raise
        except OSError:                                                   # no hard links here: create, then write
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(me, fh)
    except FileExistsError:
        held, state = lock_holder(lock)
        words = {"running": "running",
                 "being-taken": "being taken by a run starting now (the lock is not written yet)",
                 "pid-reused": "NOT running - that pid now belongs to a process started after the lock was written, so "
                               "the run that took it died; delete the lock file once you have checked no upload is in "
                               "flight",
                 "dead": "NOT running - a run that died; delete the lock file once you have checked no upload is in "
                         "flight"}[state]
        return None, _no("the manifest is locked by %s (pid %s, %s, since %s): another upload run is writing it, and a "
                         "second writer loses the first one's entries" % (lock, held.get("pid"), words, held.get("at")),
                         refusedRule="locked", lock=lock, holder=held, holderState=state,
                         holderAlive=state in ("running", "being-taken"))
    except OSError as exc:
        return None, _no("the lock %s could not be written: %s" % (lock, exc), lock=lock)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
    return lock, None


def write_manifest(path, data):
    tmp = path + ".tmp"
    from mif_roblox import durable
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
        durable.sync(fh)
    os.replace(tmp, path)


IDS_MARK = "-- Generated by rbx_asset_upload_batch from "
IDS_EXTS = (".luau", ".lua", ".rbxmx")


def _luau_str(s):
    """A string as a Luau literal on one line: UTF-8 kept, quotes, backslashes and control characters escaped (json.dumps
    writes \\uXXXX, which Luau does not read)."""
    out = []
    for ch in s:
        o = ord(ch)
        out.append("\\" + ch if ch in '"\\' else "\\n" if ch == "\n" else "\\x%02X" % o if o < 32 or o == 127 else ch)
    return '"' + "".join(out) + '"'


def ids_key(entry):
    """An entry's key in the ids module: its file as the manifest names it, with forward slashes."""
    return entry["file"].replace("\\", "/")


def _asset_id(v):
    """A manifest's assetId as a whole number (Open Cloud answers a string of digits), or None."""
    if isinstance(v, int) and not isinstance(v, bool) and v > 0:
        return v
    if isinstance(v, str) and v.isascii() and v.isdigit() and int(v) > 0:
        return int(v)
    return None


def ids_module_source(assets, manifest_name, manifest_sha256):
    """The manifest's uploaded entries as Luau: a table from ids_key to "rbxassetid://<id>", sorted by key, each line's
    asset type beside it; entries with no asset id listed in the header and left out, each with the reason the manifest
    itself holds (never this run's, which would change the bytes from one run to the next). -> (source, keys, left_out).
    Pure: the same manifest gives the same bytes."""
    from mif_roblox import opencloud as OC
    made, left = [], []
    for e in assets:
        aid = _asset_id(e.get("assetId"))
        if e.get("status") == "uploaded" and aid:
            kind = e.get("assetType") or OC.CONTENT.get(os.path.splitext(e["file"])[1].lower(), (None, "asset", None))[1]
            made.append((ids_key(e), aid, kind))
            continue
        why = "uploaded, but the manifest holds no asset id" if e.get("status") == "uploaded" else             "%s: %s" % (e["status"], e.get("error")) if e.get("status") in ("rejected", "failed") else "not uploaded"
        left.append({"key": ids_key(e), "why": why})
    made.sort()
    lines = [IDS_MARK + _luau_str(manifest_name) + ". Never edit it: the next run writes it again.",
             "-- manifest sha256: %s" % manifest_sha256,
             "-- Each key is an entry's file as the manifest names it; each value is its content id."]
    for x in left:
        lines.append("-- left out: %s (%s)" % (_luau_str(x["key"]), _luau_str(x["why"])[1:-1]))
    lines.append("return {")
    for key, aid, kind in made:
        lines.append("\t[%s] = \"rbxassetid://%d\", -- %s" % (_luau_str(key), aid, _luau_str(str(kind))[1:-1]))
    lines.append("}")
    return "\n".join(lines) + "\n", [m[0] for m in made], left


def ids_module_text(path, source):
    """The text of the file at path: the Luau itself, or for .rbxmx a ModuleScript named after the file holding it."""
    if not path.lower().endswith(".rbxmx"):
        return source
    import xml.etree.ElementTree as ET
    root = ET.Element("roblox", {"version": "4"})
    props = ET.SubElement(ET.SubElement(root, "Item", {"class": "ModuleScript", "referent": "RBX0"}), "Properties")
    ET.SubElement(props, "string", {"name": "Name"}).text = os.path.splitext(os.path.basename(path))[0]
    ET.SubElement(props, "ProtectedString", {"name": "Source"}).text = source
    return ET.tostring(root, encoding="unicode") + "\n"


def _ids_module_refusal(ids_module, assets):
    """Why this ids_module cannot be written for these entries, before anything is sent. -> a refusal or None."""
    if not isinstance(ids_module, str) or not ids_module.lower().endswith(IDS_EXTS):
        return _no("ids_module is the path of the file to write, ending in %s" % ", ".join(IDS_EXTS))
    if not os.path.isdir(os.path.dirname(os.path.abspath(ids_module))):
        return _no("ids_module's folder is not there: %s" % os.path.dirname(os.path.abspath(ids_module)))
    if os.path.exists(ids_module):
        try:
            with open(ids_module, "rb") as fh:
                old = fh.read()
        except OSError as e:
            return _no("ids_module did not read: %s" % e)
        if IDS_MARK.encode("utf-8") not in old[:400]:
            return _no("%s exists and was not generated by this tool: it is not overwritten; name another path or move "
                       "the file away" % ids_module, refusedRule="not-generated")
    return _ids_dup_refusal(assets)


def _ids_dup_refusal(assets):
    """The keys' refusal shared by ids_module and ids_target: a key that is not text, or a file listed twice."""
    seen, dup = {}, []
    for i, e in enumerate(assets):
        k = ids_key(e)
        try:
            k.encode("utf-8")
        except UnicodeEncodeError:
            return _no("entry %d's file is not text a module can hold" % i)
        if k in seen:
            dup.append({"key": k, "entries": [seen[k], i]})
        seen.setdefault(k, i)
    if dup:
        return _no("ids_module keys each entry by its file, and %d file(s) are listed twice (%s): one key cannot hold "
                   "two ids; remove the repeat" % (len(dup), ", ".join(d["key"] for d in dup[:5])),
                   refusedRule="duplicate-key", duplicates=dup)
    return None


def _write_ids_module(ids_module, manifest, assets, dry_run):
    """Write (or, in a dry run, describe) the ids module for the manifest as it now stands on disk. -> dict."""
    import hashlib
    from mif_roblox import opencloud as OC
    src, keys, left = ids_module_source(assets, os.path.basename(manifest), OC.file_sha256(manifest))
    raw = ids_module_text(ids_module, src).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    old = None
    if os.path.isfile(ids_module):
        with open(ids_module, "rb") as fh:
            old = fh.read()
    out = {"ok": True, "path": ids_module, "keys": keys, "leftOut": left, "sha256": digest, "bytes": len(raw),
           "unchanged": old == raw, "written": False}
    if not keys:
        return dict(out, ok=dry_run, why="no entry has an asset id yet: no module is written, for an empty table would "
                    "answer nil for every file")
    if dry_run:
        return dict(out, wouldWrite=old != raw)
    if old == raw:
        return out
    tmp = ids_module + ".tmp"
    from mif_roblox import durable
    with open(tmp, "wb") as fh:
        fh.write(raw)
        durable.sync(fh)
    os.replace(tmp, ids_module)
    back = OC.file_sha256(ids_module)
    if back != digest:
        return dict(out, ok=False, written=True, verified=False, why="the module read back as sha256 %s" % back)
    return dict(out, written=True, verified=True)


def _ids_target_state(target):
    """The open place's side of ids_target, read before anything is sent: (state, refusal). state {parts, exists,
    source}; a path that is not a ModuleScript, or one this tool did not generate, is refused."""
    T = _T()
    parts = T.V.split_path(target) if isinstance(target, str) else None
    if not parts or len(parts) < 2:
        return None, _no("ids_target is the path of a ModuleScript in the open place, such as "
                         "ReplicatedStorage.AssetIds")
    got = T._rbx("resolve", target=parts)
    if not got.get("ok"):
        if got.get("editorState") in ("daemonDown", "noPlugin", "portBusy") or got.get("refusedRule") == "not-ours":
            return None, dict(got, ok=False, ran=False, error="ids_target needs a connected Studio: %s" % got.get("error"))
        return {"parts": parts, "exists": False, "source": None}, None
    if got.get("className") != "ModuleScript":
        return None, _no("%s is a %s, not a ModuleScript" % (target, got.get("className")))
    back = T.rbx_script_read(target)
    if not back.get("ok"):
        return None, _no("%s did not read: %s" % (target, back.get("error")), ran=True)
    if IDS_MARK not in (back.get("source") or "")[:400]:
        return None, _no("%s exists and was not generated by this tool: it is not overwritten; name another path or "
                         "move the script away" % target, refusedRule="not-generated")
    return {"parts": parts, "exists": True, "source": back.get("source")}, None


def _write_ids_target(target, state, manifest, assets, dry_run):
    """The same module written into the open place at target: created when absent, one undo step, read back by sha256.
    An unchanged script is not written again. -> dict."""
    import hashlib
    from mif_roblox import opencloud as OC
    T = _T()
    src, keys, left = ids_module_source(assets, os.path.basename(manifest), OC.file_sha256(manifest))
    digest = hashlib.sha256(src.encode("utf-8")).hexdigest()
    out = {"ok": True, "target": target, "keys": keys, "leftOut": left, "sha256": digest,
           "unchanged": state["source"] == src, "written": False, "created": False}
    if not keys:
        return dict(out, ok=dry_run, why="no entry has an asset id yet: no module is written, for an empty table would "
                    "answer nil for every file")
    if dry_run:
        return dict(out, wouldWrite=state["source"] != src, wouldCreate=not state["exists"])
    if state["source"] == src:
        return out
    g = T.begin_group("MifBridge asset ids module")
    if not g.get("ok"):
        return dict(out, ok=False, why="no undo group could be opened, so nothing was written: %s" % g.get("error"))
    if not state["exists"]:
        made = T.rbx_create("ModuleScript", T.V.join_path(state["parts"][:-1]), name=state["parts"][-1])
        if not made.get("ok"):
            T.rbx_undo_group("cancel")
            return dict(out, ok=False, why="the ModuleScript could not be made: %s" % made.get("error"))
        out["created"] = True
    w = T.rbx_script_write(target, src)
    if not w.get("ok"):
        T.rbx_undo_group("cancel")
        return dict(out, ok=False, created=False, why="the source was not written (the group was cancelled): %s"
                    % w.get("error"))
    c = T.rbx_undo_group("commit")
    back = T.rbx_script_read(target)
    same = bool(back.get("ok")) and back.get("sha256") == digest
    return dict(out, ok=same and bool(c.get("ok")), written=True, verified=same, undoSteps=1 if c.get("ok") else 0,
                why=None if same else "the module read back as sha256 %s" % back.get("sha256"))


def rbx_asset_upload_batch(manifest: str, confirm: bool = False, dry_run: bool = True, stop_after: int = None,
                           accept_shrink: bool = False, ids_module: str = None, ids_target: str = None) -> dict:
    "Roblox: upload a manifest's files one at a time through rbx_asset_upload, with one writer at a time. manifest is a JSON file {\"assets\": [{file (relative to the manifest), displayName, assetType?, description?}]}; the run takes <manifest>.lock exclusively (a second run is refused, naming the holder's pid and whether it still runs), skips an entry whose file sha256 or (for an .fbx) mesh-data hash matches an entry already uploaded or rejected by moderation - a re-export of an unchanged mesh is not uploaded again (Blender rewrites the bytes on every export, RB1475) - stops at a 401 or 403 and at a refusal that sent nothing (not full mode, no signed-in user), writes the manifest after each file (assetId, sha256, meshHash, status uploaded, rejected when moderation answered Rejected, or failed - a 400 such as a bad name may go again, at), and releases the lock. A run that sends nothing while no entry is uploaded (every file missing or unreadable) answers ok false. dry_run (the default) changes nothing: it lists for each entry what would be sent (type, name, bytes, hashes) or why it is skipped, and the skinned-size warning (RB1472). An upload is permanent: dry_run false needs confirm true and the panel in full mode. stop_after limits how many files are sent this run. ids_module names a file on this machine (.luau, .lua, or .rbxmx for a ModuleScript to insert) written when the run ends from the manifest as it then stands: a Luau module returning a table from each uploaded entry's file (as the manifest names it, forward slashes) to its content id \"rbxassetid://<id>\", sorted by key, the asset type beside each line and the manifest's sha256 in its header, so game code requires the ids and nobody types them. Entries with no asset id (not uploaded, rejected, failed, or skipped as a copy of another entry) are left out and listed, in the answer and in the header. It is generated, never edited: the same manifest gives the same bytes, an unchanged module is not written again (unchanged true), a file there that this tool did not generate is refused, a file listed twice is refused before anything is sent, and the write is read back by sha256. A dry run writes nothing and answers the keys and whether it would write; with no asset id in the manifest no module is written. ids_target names a ModuleScript in the open place (ReplicatedStorage.AssetIds) to hold the same source: created when absent and written in one undo step, read back by sha256, left alone when unchanged; one there that this tool did not generate, or anything but a ModuleScript, is refused before anything is sent; it needs a connected Studio."
    if not isinstance(manifest, str) or not os.path.isfile(manifest):
        return _no("manifest is the path of an existing JSON file")
    if not isinstance(dry_run, bool) or not isinstance(confirm, bool) or not isinstance(accept_shrink, bool):
        return _no("dry_run, confirm and accept_shrink are true or false")
    if stop_after is not None and not (isinstance(stop_after, int) and not isinstance(stop_after, bool) and stop_after >= 1):
        return _no("stop_after is a whole number, 1 or more")
    if not dry_run and confirm is not True:
        return _no("an upload makes permanent, moderated assets on the person's account: pass confirm true once they "
                   "agreed, or dry_run true to see what would be sent", refusedRule="needs-confirm")
    try:
        data = json.load(open(manifest, encoding="utf-8"))
    except (OSError, ValueError) as e:
        return _no("manifest did not read: %s" % e)
    assets = data.get("assets") if isinstance(data, dict) else None
    if not isinstance(assets, list) or not assets:
        return _no("manifest is {\"assets\": [{file, displayName, ...}]} with at least one entry: nothing to upload")
    bad = []
    for i, e in enumerate(assets):
        if not isinstance(e, dict) or set(e) - ENTRY_KEYS or not isinstance(e.get("file"), str) \
                or not isinstance(e.get("displayName"), str) or not 1 <= len(e.get("displayName") or "") <= 50:
            bad.append(i)
    if bad:
        return _no("entries %s are not {file, displayName (1-50 characters), assetType?, description?} (keys: %s)" % (
            bad[:10], ", ".join(sorted(ENTRY_KEYS))))
    if ids_module is not None:
        refusal = _ids_module_refusal(ids_module, assets)
        if refusal:
            return refusal
    ids_state = None
    if ids_target is not None:
        refusal = _ids_dup_refusal(assets)      # the same keys as the file: a file listed twice is refused here too
        if refusal:
            return refusal
        ids_state, refusal = _ids_target_state(ids_target)
        if refusal:
            return refusal
    base = os.path.dirname(os.path.abspath(manifest))
    lock, err = (None, None) if dry_run else take_lock(manifest)
    if err:
        return err
    from mif_roblox import opencloud as OC
    rows, sent, stopped = [], 0, None
    try:
        done_hashes = {}
        for e in assets:
            if e.get("status") in ("uploaded", "rejected"):
                for key in ("sha256", "meshHash"):
                    if e.get(key):
                        done_hashes[(key, e[key])] = e
        for i, e in enumerate(assets):
            path = e["file"] if os.path.isabs(e["file"]) else os.path.join(base, e["file"])
            row = {"index": i, "file": e["file"], "displayName": e["displayName"]}
            if not os.path.isfile(path):
                rows.append(dict(row, action="skip", why="the file is not there"))
                continue
            digest = OC.file_sha256(path)
            mh, warn = None, None
            if path.lower().endswith(".fbx"):
                try:
                    sc = F.load(path)
                    mh = F.mesh_hash(sc)
                    warn = F.skinned_size_check(sc)
                except (OSError, ValueError) as ex:
                    rows.append(dict(row, action="skip", why="the FBX did not read: %s" % ex))
                    continue
            row.update(sha256=digest, meshHash=mh, bytes=os.path.getsize(path))
            prior = done_hashes.get(("sha256", digest)) or (done_hashes.get(("meshHash", mh)) if mh else None)
            if prior is not None:
                rows.append(dict(row, action="skip", why="unchanged since %s %s (%s)" % (
                    "asset" if prior.get("assetId") else "the rejected entry", prior.get("assetId") or prior.get("file"),
                    prior.get("status")), assetId=prior.get("assetId")))
                continue
            if warn and warn["skinned"] and warn["overLimit"] and not accept_shrink:
                rows.append(dict(row, action="skip", why="skinned, largest mesh %s is %.1f studs: the import would "
                                 "shrink it to 2048 and leave its bones at full size (RB1023, RB1472); re-export it "
                                 "smaller, or accept_shrink" % (warn["largestMesh"], warn["largestStuds"]), skinnedSize=warn))
                continue
            if stop_after is not None and sent >= stop_after:
                rows.append(dict(row, action="later", why="stop_after %d reached" % stop_after))
                continue
            if dry_run:
                ext = os.path.splitext(path)[1].lower()
                ctype, default_type, _allowed = OC.CONTENT.get(ext, (None, None, None))
                rows.append(dict(row, action="would-upload", request={
                    "assetType": e.get("assetType") or default_type, "displayName": e["displayName"],
                    "description": e.get("description") or "", "contentType": ctype,
                    "creationContext": "the signed-in Studio user (StudioService:GetUserId)"}, skinnedSize=warn))
                sent += 1
                continue
            T = _T()
            up = T.rbx_asset_upload(path, e["displayName"], confirm=True, description=e.get("description") or "",
                                    asset_type=e.get("assetType"), accept_shrink=accept_shrink)
            status = up.get("createStatus")
            if up.get("ran") is False and not up.get("unchanged"):
                # refused before anything was sent (not full mode, no signed-in user, ...): every other entry would be
                # refused the same way, so the run stops here and this entry is left as it was
                rows.append(dict(row, action="refused", refusedRule=up.get("refusedRule"), error=up.get("error")))
                stopped = "nothing was sent for %s: %s; the run stopped" % (e["file"], up.get("error"))
                break
            if not up.get("unchanged"):
                sent += 1
            if up.get("moderationState") == "Rejected":
                # moderation's verdict is the only sticky one: the same content is never sent again
                e.update(assetId=up.get("assetId"), status="rejected", error=str(up.get("error") or "moderation "
                                                                                 "rejected the asset")[:300])
            elif up.get("assetId"):
                e.update(assetId=up["assetId"], status="uploaded", error=None)
            else:
                # a 400 (a bad name, the costs-robux refusal) is not moderation: fixed, the file may go again
                e.update(status="failed", error=str(up.get("error"))[:300])
            e.update(sha256=digest, meshHash=mh, at=time.strftime("%Y-%m-%dT%H:%M:%S"), operation=up.get("operation"))
            if e["status"] in ("uploaded", "rejected"):
                done_hashes[("sha256", digest)] = e
                if mh:
                    done_hashes[("meshHash", mh)] = e
            write_manifest(manifest, data)
            rows.append(dict(row, action=e["status"], assetId=e.get("assetId"), createStatus=status,
                             error=e.get("error")))
            if status in (401, 403):
                stopped = "Roblox answered %s: the key cannot upload (%s); the run stopped" % (status, e.get("error"))
                break
    finally:
        if lock:
            try:
                os.remove(lock)
            except OSError:
                pass
    # an empty result fails: a run that sent nothing (or, dry, would send nothing) while no entry is uploaded or
    # rejected - every file missing or unreadable - did nothing, and says so
    nothing = sent == 0 and not any(e.get("status") in ("uploaded", "rejected") for e in assets if isinstance(e, dict))
    # every row skipped for a file that is missing or did not read fails the run, even when older entries were uploaded
    # (decided 2026-11-23: never report success for a batch that verified nothing)
    unread = [r for r in rows if r.get("action") == "skip" and str(r.get("why") or "").startswith(
        ("the FBX did not read", "the file is not there"))]
    if rows and len(unread) == len(rows):
        nothing = True
    out = {"ok": stopped is None and not nothing and not any(r["action"] == "failed" for r in rows), "dryRun": dry_run,
           "manifest": manifest, "entries": len(assets), "rows": rows,
           "wouldUpload" if dry_run else "sent": sent,
           "skipped": sum(1 for r in rows if r["action"] == "skip"), "lockReleased": bool(lock) and not os.path.exists(lock)}
    if stopped:
        out["error"] = stopped
    elif nothing:
        out["error"] = "nothing %s and no entry is uploaded yet: %s" % (
            "would be sent" if dry_run else "was sent", "; ".join(sorted({"%s (%s)" % (r["file"], r.get("why"))
                                                                         for r in rows})[:5]) or "no rows")
    elif not out["ok"]:
        out["error"] = "%d upload(s) failed" % sum(1 for r in rows if r["action"] == "failed")
    if ids_module is not None:
        try:
            out["idsModule"] = _write_ids_module(ids_module, manifest, assets, dry_run)
        except OSError as e:
            out["idsModule"] = {"ok": False, "path": ids_module, "written": False, "why": "the module was not written: %s" % e}
        if not out["idsModule"]["ok"]:
            out["ok"] = False
            out.setdefault("error", "ids_module: %s" % out["idsModule"].get("why"))
    if ids_target is not None:
        out["idsTarget"] = _write_ids_target(ids_target, ids_state, manifest, assets, dry_run)
        if not out["idsTarget"]["ok"]:
            out["ok"] = False
            out.setdefault("error", "ids_target: %s" % out["idsTarget"].get("why"))
    return out


# ------------------------------------------------------------------------------------------- the rig after import
def rig_compare(fbx_bones, fbx_meshes, got_bones, got_meshes, tolerance):
    """The bone frames and mesh sizes Studio gave, against the file's. fbx_bones [{name, parent, rbx [3]}], fbx_meshes
    {name: rbx size}, got_bones {name: position}, got_meshes {name: Size}. -> dict (pure; the tests run it offline)."""
    roots = [b for b in fbx_bones if not b["parent"]]
    root = next((b for b in roots if b["name"] in got_bones), None)
    if root is None:
        return {"ok": False, "error": "no root bone of the file (%s) is among the %d bones in Studio" % (
            ", ".join(b["name"] for b in roots) or "none", len(got_bones))}
    pairs, missing = [], []
    for b in fbx_bones:
        if b["name"] not in got_bones:
            missing.append(b["name"])
            continue
        e = [b["rbx"][k] - root["rbx"][k] for k in range(3)]
        a = [got_bones[b["name"]][k] - got_bones[root["name"]][k] for k in range(3)]
        pairs.append((b["name"], e, a))
    ee = sum(v * v for _n, e, _a in pairs for v in e)
    bone_scale = sum(e[k] * a[k] for _n, e, a in pairs for k in range(3)) / ee if ee > 0 else None
    worst, worst_name, rows = 0.0, None, []
    for n, e, a in pairs:
        r = max(abs(a[k] - (bone_scale or 1.0) * e[k]) for k in range(3))
        rows.append({"bone": n, "expected": [round(v, 4) for v in e], "studio": [round(v, 4) for v in a],
                     "residual": round(r, 4)})
        if r > worst:
            worst, worst_name = r, n
    ratios = []
    mesh_rows = []
    for n, size in sorted(fbx_meshes.items()):
        g = got_meshes.get(n)
        if not g:
            continue
        k = max(range(3), key=lambda i: size[i])
        if size[k] > 0:
            ratios.append(g[k] / size[k])
            mesh_rows.append({"mesh": n, "file": size, "studio": [round(v, 4) for v in g], "scale": round(g[k] / size[k], 6)})
    ratios.sort()
    mesh_scale = ratios[len(ratios) // 2] if ratios else None
    extent = max((max(abs(v) for v in e) for _n, e, _a in pairs), default=0.0)
    fails = []
    if bone_scale is None:
        fails.append("no bone away from the root to measure a scale with")
    if mesh_scale is None:
        fails.append("no MeshPart in Studio shares a name with a mesh of the file")
    if bone_scale is not None and mesh_scale is not None and abs(bone_scale - mesh_scale) > tolerance * max(bone_scale, mesh_scale):
        fails.append("the bones are at x%.4f of the file and the meshes at x%.4f: the import shrank the meshes and not "
                     "the bones, so the skin is out of the bones' space (RB1023, RB1472)" % (bone_scale, mesh_scale))
    if bone_scale is not None and extent and worst > tolerance * extent * abs(bone_scale):
        fails.append("bone %s is %.4g studs from where the file puts it (tolerance %.4g)" % (
            worst_name, worst, tolerance * extent * abs(bone_scale)))
    return {"ok": not fails, "pass": not fails, "failed": fails, "root": root["name"],
            "boneScale": round(bone_scale, 6) if bone_scale is not None else None,
            "meshScale": round(mesh_scale, 6) if mesh_scale is not None else None,
            "worstBone": worst_name, "worstResidual": round(worst, 4), "bonesCompared": len(pairs),
            "bonesMissing": missing, "bones": rows, "meshes": mesh_rows}


def rbx_rig_check(file: str, target: Any = None, asset_id: int = None, tolerance: float = 0.01, keep: bool = False) -> dict:
    "Roblox Studio: after a skinned FBX's import, read its Bones' WorldCFrame and its MeshParts' Size and hold them against the file: each bone's position relative to the root against the file's bind pose on Roblox axes (file (-X, +Y, -Z), 1 stud per file unit, RB1156, RB1700), the scale fitted from the bones against the scale of the meshes. A clean import has both at x1 with every bone in place; an import that shrank the meshes (largest over 2048 studs, RB1023) leaves the bones at full size and is reported with both scales (RB1472). target is the imported model in the place, or asset_id loads one of the person's assets into Workspace.MifScratch (a read) and removes it afterwards unless keep. tolerance is the fraction allowed (default 1%)."
    T, G = _T(), _G()
    V = T.V
    if not isinstance(file, str) or not os.path.isfile(file):
        return _no("file is the .fbx the asset was made from")
    if (target is None) == (asset_id is None):
        return _no("give target (a model in the place) or asset_id (one of the person's assets), one of them")
    if not (_num(tolerance) and 0 < tolerance < 0.5):
        return _no("tolerance is a fraction between 0 and 0.5")
    try:
        sc = F.load(file)
    except (OSError, ValueError) as e:
        return _no("the FBX did not read: %s" % e)
    if not sc.bones:
        return _no("the file has no bones: nothing to check")
    if not sc.rbx_axes():
        return _no("the file's axis settings %s are not the ones the import was measured on (RB1156)" % sc.axis_settings)
    placed = None
    if asset_id is not None:
        ins = T.rbx_insert_asset(asset_id)
        if not ins.get("ok"):
            return dict(ins, ok=False, step="load")
        placed = target = ins["path"]
    try:
        root = V.split_path(target) if isinstance(target, str) else None
        if not root:
            return _no("target is a path")
        bones, meshes, page = [], [], 0
        while True:
            q = T._rbx("query", root=root, page=page, pageSize=500)
            if not q.get("ok"):
                return dict(q, ok=False)
            for m in q.get("matches") or []:
                if m.get("className") == "Bone":
                    bones.append(m["path"])
                elif m.get("className") == "MeshPart":
                    meshes.append(m["path"])
            if not q.get("more"):
                break
            page += 1
        if not bones:
            return _no("no Bone under %s: the import made no skeleton to check" % V.join_path(root), scanned=0)
        brows, route = G._many([{"target": b} for b in bones], props=["Name", "WorldCFrame"])
        if brows is None:
            return route
        got_bones = {}
        for row in brows:
            cf = G._cf((row.get("values") or {}).get("WorldCFrame"))
            if cf and (row.get("values") or {}).get("Name"):
                got_bones.setdefault(row["values"]["Name"], cf["position"])
        mrows, _r = G._many([{"target": m} for m in meshes], props=["Name", "Size"]) if meshes else ([], None)
        got_meshes = {}
        for row in mrows or []:
            v = row.get("values") or {}
            size = G._v3(v.get("Size"))
            if size and v.get("Name"):
                got_meshes.setdefault(v["Name"], size)
        fb = [{"name": b["name"], "parent": b["parent"], "rbx": F.rbx_point(b["position"])} for b in sc.bones]
        fm = {m["name"]: F.rbx_box(m)[0] for m in sc.meshes}
        out = rig_compare(fb, fm, got_bones, got_meshes, tolerance)
        out.update(target=V.join_path(root), boneCount=len(bones), meshPartCount=len(meshes), readRoute=route,
                   skinnedSize=F.skinned_size_check(sc))
        if not out["ok"] and "error" not in out:
            out["error"] = "; ".join(out["failed"])
        return out
    finally:
        if placed and not keep:
            T.rbx_destroy(placed)
