"""MifBlender ops: game-asset bakes - vertex animation textures, Pivot Painter 2 data, hair cards, and a whole bake set out
of process (1.3 gap audit, 2026-10-19).

THE GAP. Unreal's bake_vertex_animation bakes SKELETAL animation only, so a cloth, a fluid surface or a destruction made in
Blender reached the engine only as Alembic; a tree built here reached Unreal with no per-branch wind data for the engine's
Pivot Painter 2 functions; a groom had no route to cards; and a bake set ran map by map on the main thread, where a 4K
multi-map bake passes the bridge's ceiling and reports a timeout while it keeps running.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import struct
import subprocess
import tempfile
import time

import bpy

from . import childproc
from .ops_common import (MifOpError, caller_deadline, caller_gave_up, check_output_path, check_writable_dir, get_object,
                         reject_unknown, take, take_bool, take_float, take_int)


def _folder(params, tail):
    raw = take(params, "folder", required=True, kind=str)
    d = os.path.abspath(bpy.path.abspath(raw))
    try:
        check_writable_dir(d)
    except (OSError, ValueError) as exc:   # ValueError: a NUL in the path
        raise MifOpError("the folder '%s' cannot be written (%s). %s" % (d, exc, tail))
    return d


def _write_image(np, name, rgba, path, fmt):
    """A float Non-Color image from (h, w, 4), saved through save_image as EXR 32 or PNG 16; removed after."""
    from .ops_image import op_save_image                       # noqa: PLC0415
    h, w = rgba.shape[:2]
    im = bpy.data.images.new(name, w, h, alpha=True, float_buffer=True)
    try:
        im.colorspace_settings.name = "Non-Color"
        im.pixels.foreach_set(np.ascontiguousarray(rgba, dtype=np.float32).ravel())
        im.update()
        req = {"image": im.name, "filepath": path, "format": fmt, "colorDepth": 32 if fmt == "OPEN_EXR" else 16}
        saved = op_save_image(req)
    finally:
        bpy.data.images.remove(im)
    return saved


# --------------------------------------------------------------------------------------------- bake_vertex_animation

_VAT_KEYS = ("object", "frameRange", "folder", "name", "format", "normals", "unrealAxes", "maxFrames")


def op_bake_vertex_animation(params):
    """Bake a deforming mesh over a frame range into vertex animation textures - a position-offset texture and a normal
    texture, one row per frame and one column per vertex - and a rest mesh whose UV channel VAT_UV gives each vertex its
    column; for the engine's vertex-animation material path. The pair of Unreal's bake_vertex_animation (skeletal only).

    params:
      object (str)          a mesh deformed by its modifiers (cloth, a fluid surface of fixed topology, shape keys,
                            geometry nodes, an armature) - evaluated per frame
      frameRange ([s, e])   default the scene's; maxFrames (default 1000) caps it
      folder (str), name    the files <name>_Offset.<ext>, <name>_Normal.<ext> (name default VAT_<object>)
      format (str)          OPEN_EXR (default: 32-bit float offsets, exact) | PNG (16-bit, offsets normalized to the bounds)
      normals (bool)        write the normal texture (default true)
      unrealAxes (bool)     default true: offsets in Unreal's frame - centimeters, Y negated - as every send here
    A changing vertex count is refused, naming the frame (remesh to fixed topology, or use Alembic). The rest mesh is a new
    object <name>_Rest at the first frame's shape. Reply: vertices, frames, the offset bounds (min / max per axis) a PNG
    is normalized to, the files and the rest object.
    """
    reject_unknown(params, _VAT_KEYS, "bake_vertex_animation")
    tail = "NOTHING was written."
    import numpy as np                                          # noqa: PLC0415
    obj = get_object(take(params, "object", required=True, kind=str), want_mesh=True)
    sc = bpy.context.scene
    fr = params.get("frameRange") or [sc.frame_start, sc.frame_end]
    if not isinstance(fr, list) or len(fr) != 2 or any(isinstance(v, bool) or not isinstance(v, int) for v in fr) \
            or fr[1] < fr[0]:
        raise MifOpError("frameRange is [start, end] whole frames, got %r. %s" % (fr, tail))
    nf = fr[1] - fr[0] + 1
    if nf > take_int(params, "maxFrames", default=1000):
        raise MifOpError("%d frames is over maxFrames (default 1000) - one texture row per frame. %s" % (nf, tail))
    fmt = str(take(params, "format", default="OPEN_EXR", kind=str)).upper()
    if fmt not in ("OPEN_EXR", "PNG"):
        raise MifOpError("format is OPEN_EXR or PNG, got %r. %s" % (fmt, tail))
    folder = _folder(params, tail)
    name = take(params, "name", default="VAT_%s" % obj.name, kind=str)
    ue = take_bool(params, "unrealAxes", default=True)
    want_n = take_bool(params, "normals", default=True)
    # THE FILE NAMES ARE CHECKED BEFORE THE BAKE (2026-10-28, the version matrix's corrupted-value pass): a name holding
    # a control character was refused by save_image only after the frames were evaluated and the rest mesh made, and
    # the refusal left that mesh in the file.
    ext = ".exr" if fmt == "OPEN_EXR" else ".png"
    for part in ("_Offset",) + (("_Normal",) if want_n else ()):
        check_output_path(name + part, os.path.join(folder, name + part + ext), "baked")
    saved_frame = sc.frame_current
    pos, nrm, counts = [], [], []
    rest_mesh = None
    try:
        for k, f in enumerate(range(fr[0], fr[1] + 1)):
            sc.frame_set(f)
            dg = bpy.context.evaluated_depsgraph_get()
            ev = obj.evaluated_get(dg)
            me = ev.to_mesh()
            try:
                n = len(me.vertices)
                counts.append(n)
                if counts[0] != n:
                    raise MifOpError("the mesh has %d vertices at frame %d and %d at frame %d - a vertex animation texture "
                                     "needs one topology; remesh to a fixed one, or send it as Alembic. %s"
                                     % (counts[0], fr[0], n, f, tail))
                if n > 16384:
                    raise MifOpError("%d vertices is over 16384, the widest texture this writes. %s" % (n, tail))
                a = np.empty(n * 3, dtype=np.float32)
                me.vertices.foreach_get("co", a)
                pos.append(a.reshape(n, 3))
                if want_n:
                    b = np.empty(n * 3, dtype=np.float32)
                    me.vertices.foreach_get("normal", b)
                    nrm.append(b.reshape(n, 3))
                if k == 0:
                    rest_mesh = bpy.data.meshes.new_from_object(ev, depsgraph=dg)
            finally:
                ev.to_mesh_clear()
    except Exception:
        if rest_mesh is not None:
            bpy.data.meshes.remove(rest_mesh)
        raise
    finally:
        sc.frame_set(saved_frame)
    P = np.stack(pos)                       # (frames, verts, 3)
    off = P - P[0]
    if ue:
        off = off * np.array([100.0, -100.0, 100.0], dtype=np.float32)
    lo, hi = off.reshape(-1, 3).min(axis=0), off.reshape(-1, 3).max(axis=0)
    rgba = np.ones((nf, P.shape[1], 4), dtype=np.float32)
    if fmt == "PNG":
        span = np.where(hi - lo > 1e-9, hi - lo, 1.0)
        rgba[..., :3] = (off - lo) / span
    else:
        rgba[..., :3] = off
    files = {}
    try:
        s = _write_image(np, name + "_Offset", rgba, os.path.join(folder, name + "_Offset" + ext), fmt)
        files["offset"] = s["file"]
        if want_n:
            N = np.stack(nrm)
            if ue:
                N = N * np.array([1.0, -1.0, 1.0], dtype=np.float32)
            rn = np.ones((nf, P.shape[1], 4), dtype=np.float32)
            rn[..., :3] = N * 0.5 + 0.5
            s = _write_image(np, name + "_Normal", rn, os.path.join(folder, name + "_Normal" + ext), fmt)
            files["normal"] = s["file"]
    except Exception:
        # nothing is linked yet: the rest mesh is the one datablock this call has made
        bpy.data.meshes.remove(rest_mesh)
        raise
    # THE REST MESH: the first frame's shape, with VAT_UV holding each vertex's column (and row 0)
    nv = P.shape[1]
    uv = rest_mesh.uv_layers.new(name="VAT_UV")
    loops_v = np.empty(len(rest_mesh.loops), dtype=np.int32)
    rest_mesh.loops.foreach_get("vertex_index", loops_v)
    co = np.zeros((len(loops_v), 2), dtype=np.float32)
    co[:, 0] = (loops_v + 0.5) / nv
    co[:, 1] = 0.5 / nf
    uv.data.foreach_set("uv", co.ravel())
    rest = bpy.data.objects.new(name + "_Rest", rest_mesh)
    rest.matrix_world = obj.matrix_world.copy()
    for c in obj.users_collection:
        c.objects.link(rest)
    moved = float(np.abs(off).max())
    out = {"object": obj.name, "vertices": nv, "frames": nf, "frameRange": fr, "format": fmt, "files": files,
           "fps": round(sc.render.fps / sc.render.fps_base, 6),
           "restObject": rest.name, "uvLayer": "VAT_UV", "uvLayerIndex": len(rest_mesh.uv_layers) - 1,
           "bounds": {"min": [round(float(c), 6) for c in lo], "max": [round(float(c), 6) for c in hi]},
           "units": "cm, Unreal axes (Y negated)" if ue else "Blender units, Blender axes",
           "encoding": ("offset = position - first frame, raw floats" if fmt == "OPEN_EXR" else
                        "offset normalized to bounds: value = (offset - min) / (max - min)"),
           "maxOffset": round(moved, 6)}
    if moved < 1e-6:
        out["warning"] = "no vertex moved over the range - the mesh is not animated there"
    return out


# --------------------------------------------------------------------------------------------- bake_pivot_painter

_PP_KEYS = ("object", "collection", "folder", "name", "format", "uvLayer", "parentIndexEncoding")


def _int_as_float(i):
    return struct.unpack("<f", struct.pack("<i", int(i)))[0]


def op_bake_pivot_painter(params):
    """Bake each element of a model - each loose part of a mesh, or each object of a collection - to Pivot Painter 2 data:
    a pivot position + parent index texture, an X axis + X extent texture, and a UV channel holding each element's pixel.

    params:
      object (str)              a mesh whose LOOSE PARTS are the elements (a tree built with build_tree / build_plant:
                                trunk, branches, leaves) - the hierarchy is inferred: each part's parent is the larger part
                                nearest its pivot, its pivot the part's point nearest that parent (the root part: its
                                lowest point); or
      collection (str)          objects as elements, the hierarchy from their parenting, each pivot its origin and its X
                                axis its local X
      folder (str), name        <name>_PivotPos_ParentIndex.exr and <name>_XVector_XExtent.<ext> (name default PP_<model>)
      format (str)              OPEN_EXR (default; the pivot texture is always 32-bit float EXR) - the X-vector texture
                                is written PNG 16 when format is PNG
      uvLayer (str)             default "PivotPainter"
      parentIndexEncoding (str) INT_AS_FLOAT (default: Pivot Painter 2's - the index's integer bits read as a float) |
                                FLOAT (the index as a plain number)
    Encoding (Unreal units: centimeters, Y negated): pivot texture RGB = pivot position, A = the parent's index (an element
    with no parent points at itself); X texture RGB = the X axis * 0.5 + 0.5, A = the X extent / 2048 cm. Reply: elements,
    hierarchy depth, per element its parent and extent, the textures and the UV layer.
    """
    reject_unknown(params, _PP_KEYS, "bake_pivot_painter")
    tail = "NOTHING was written."
    if (params.get("object") is None) == (params.get("collection") is None):
        raise MifOpError("give object (loose parts are the elements) or collection (objects are). %s" % tail)
    enc = str(take(params, "parentIndexEncoding", default="INT_AS_FLOAT", kind=str)).upper()
    if enc not in ("INT_AS_FLOAT", "FLOAT"):
        raise MifOpError("parentIndexEncoding is INT_AS_FLOAT or FLOAT, got %r. %s" % (enc, tail))
    fmt = str(take(params, "format", default="OPEN_EXR", kind=str)).upper()
    if fmt not in ("OPEN_EXR", "PNG"):
        raise MifOpError("format is OPEN_EXR or PNG, got %r. %s" % (fmt, tail))
    folder = _folder(params, tail)
    uv_name = take(params, "uvLayer", default="PivotPainter", kind=str)
    # AFTER THE REFUSALS: a call that is refused must not need numpy or mathutils (the offline refusal sweep has neither).
    import numpy as np                                          # noqa: PLC0415
    from mathutils import Vector                               # noqa: PLC0415
    from mathutils.kdtree import KDTree                        # noqa: PLC0415
    to_ue = Vector((100.0, -100.0, 100.0))
    elements = []      # dicts: pivot (world), xaxis (world unit), extent (world), parent index, verts or object

    if params.get("object") is not None:
        obj = get_object(take(params, "object", kind=str), want_mesh=True)
        me = obj.data
        model = obj.name
        import bmesh                                            # noqa: PLC0415
        bm = bmesh.new()
        bm.from_mesh(me)
        bm.verts.ensure_lookup_table()
        seen, parts = set(), []
        for v in bm.verts:
            if v.index in seen:
                continue
            stack, comp = [v], []
            seen.add(v.index)
            while stack:
                x = stack.pop()
                comp.append(x.index)
                for e in x.link_edges:
                    o = e.other_vert(x)
                    if o.index not in seen:
                        seen.add(o.index)
                        stack.append(o)
            parts.append(comp)
        bm.free()
        mw = obj.matrix_world
        wco = [mw @ v.co for v in me.vertices]
        parts.sort(key=len, reverse=True)
        trees = []
        for comp in parts:
            t = KDTree(len(comp))
            for i in comp:
                t.insert(wco[i], i)
            t.balance()
            trees.append(t)
        for k, comp in enumerate(parts):
            parent = k
            pivot = min((wco[i] for i in comp), key=lambda p: p.z)
            if k > 0:
                best = None
                for j in range(k):                 # larger parts only: the hierarchy runs from big to small
                    for i in comp:
                        co, _idx, d = trees[j].find(wco[i])
                        if best is None or d < best[0]:
                            best = (d, j, wco[i])
                parent, pivot = best[1], best[2]
            far = max((wco[i] for i in comp), key=lambda p: (p - pivot).length)
            xa = far - pivot
            elements.append({"name": "part %d" % k, "verts": comp, "parent": parent, "pivot": pivot,
                             "xaxis": xa.normalized() if xa.length > 1e-9 else Vector((1, 0, 0)), "extent": xa.length})
        targets = [(obj, {i: k for k, comp in enumerate(parts) for i in comp})]
    else:
        coll = bpy.data.collections.get(take(params, "collection", kind=str))
        if coll is None:
            raise MifOpError("no collection named '%s'. %s" % (params["collection"], tail))
        objs = [o for o in coll.all_objects if o.type == "MESH"]
        if not objs:
            raise MifOpError("collection '%s' holds no mesh. %s" % (coll.name, tail))
        objs.sort(key=lambda o: len(o.parent_recursive) if hasattr(o, "parent_recursive") else 0)
        index = {o.name: k for k, o in enumerate(objs)}
        model = coll.name
        for o in objs:
            mw = o.matrix_world
            xa = (mw.to_3x3() @ Vector((1, 0, 0)))
            ext = max(((mw @ v.co) - mw.translation).dot(xa.normalized()) for v in o.data.vertices) if len(
                o.data.vertices) else 0.0
            p = o.parent
            while p is not None and p.name not in index:
                p = p.parent
            elements.append({"name": o.name, "parent": index[p.name] if p is not None else index[o.name],
                             "pivot": mw.translation.copy(), "xaxis": xa.normalized(), "extent": max(ext, 0.0)})
        targets = [(o, {i: index[o.name] for i in range(len(o.data.vertices))}) for o in objs]
    n = len(elements)
    w = int(math.ceil(math.sqrt(n)))
    h = int(math.ceil(n / float(w)))
    piv = np.zeros((h, w, 4), dtype=np.float32)
    xv = np.zeros((h, w, 4), dtype=np.float32)
    depth = 0
    for k, el in enumerate(elements):
        x, y = k % w, k // w
        p = Vector([el["pivot"][i] * to_ue[i] for i in range(3)])
        a = Vector((el["xaxis"].x, -el["xaxis"].y, el["xaxis"].z))
        piv[y, x, :3] = list(p)
        piv[y, x, 3] = _int_as_float(el["parent"]) if enc == "INT_AS_FLOAT" else float(el["parent"])
        xv[y, x, :3] = [c * 0.5 + 0.5 for c in a]
        xv[y, x, 3] = min(1.0, el["extent"] * 100.0 / 2048.0)
        d, j = 0, k
        while elements[j]["parent"] != j and d < n:
            j = elements[j]["parent"]
            d += 1
        depth = max(depth, d)
    name = take(params, "name", default="PP_%s" % model, kind=str)
    files = {"pivotPosParentIndex": _write_image(np, name + "_PivotPos", piv,
                                                 os.path.join(folder, name + "_PivotPos_ParentIndex.exr"),
                                                 "OPEN_EXR")["file"],
             "xVectorXExtent": _write_image(np, name + "_XVector", xv,
                                            os.path.join(folder, name + "_XVector_XExtent" +
                                                         (".exr" if fmt == "OPEN_EXR" else ".png")), fmt)["file"]}
    uv_indices = set()
    for o, vmap in targets:
        me = o.data
        uv = me.uv_layers.get(uv_name) or me.uv_layers.new(name=uv_name)
        for li, lp in enumerate(me.loops):
            k = vmap[lp.vertex_index]
            uv.data[li].uv = ((k % w + 0.5) / w, (k // w + 0.5) / h)
        uv_indices.add(me.uv_layers.find(uv_name))
    rows = [{"element": el["name"], "parent": el["parent"], "extentCm": round(el["extent"] * 100.0, 4)}
            for el in elements[:200]]
    return {"model": model, "elements": n, "hierarchyDepth": depth, "textureSize": [w, h], "files": files,
            "uvLayer": uv_name, "parentIndexEncoding": enc, "elementsList": rows,
            # the layer's index, the UV channel it becomes in Unreal; null when the collection's meshes hold it at different ones
            "uvLayerIndex": uv_indices.pop() if len(uv_indices) == 1 else None,
            "units": "cm, Unreal axes (Y negated)", "extentEncoding": "A = X extent / 2048 cm"}


# --------------------------------------------------------------------------------------------- hair_to_cards

_HC_KEYS = ("object", "cards", "segments", "width", "taper", "name", "atlas", "iterations")


def _strands(obj):
    """World-space point lists, one per strand: a Curves object's curves, or a mesh's hair particles (evaluated)."""
    mw = obj.matrix_world
    if obj.type == "CURVES":
        cd = obj.data
        pts = [mw @ p.position for p in cd.points]
        out = []
        for c in cd.curves:
            out.append(pts[c.first_point_index:c.first_point_index + c.points_length])
        return out
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    out = []
    for ps in ev.particle_systems:
        if ps.settings.type != "HAIR":
            continue
        # co_hair gives each step of a strand as drawn (the root moved onto the surface) - hair_keys' own coordinates
        # are in the particle's hair space, which put every root at one point (measured on 5.0.1)
        steps = 2 ** ps.settings.display_step
        for i in range(len(ps.particles)):
            out.append([ps.co_hair(ev, particle_no=i, step=s).copy() for s in range(steps + 1)])
    return out


def op_hair_to_cards(params):
    """Turn hair - a Curves object (4.2+) or a mesh's hair particles - into game hair cards: strands clumped into N cards by
    their roots, a ribbon per card along its clump's mean strand, tapered toward the tip and turned to face out, each card
    in its own atlas cell (V 0 at the root to 1 at the tip).

    params:
      object (str)        the hair
      cards (int)         cards made (default 24; at most one per strand)
      segments (int)      ribbon segments along each card (default 6)
      width (float)       card width at the root, world units (default: twice the clump's spread, at least 1 mm)
      taper (float)       the tip's width as a fraction of the root's (default 0.3)
      atlas ([cols, rows])  the atlas grid (default the smallest square holding every card)
      name (str)          the new mesh object (default <object>_Cards)
    Reply: cards, triangles, the atlas grid and cell size, strands read. A strand-atlas bake (color, alpha, depth from the
    strands) is not made here - bake_texture onto the cards from the hair does it.
    """
    reject_unknown(params, _HC_KEYS, "hair_to_cards")
    tail = "NOTHING was made."
    from mathutils import Vector                               # noqa: PLC0415
    obj = get_object(take(params, "object", required=True, kind=str))
    if obj.type not in ("CURVES", "MESH"):
        raise MifOpError("'%s' is a %s - give a Curves object or a mesh with hair particles. %s" % (obj.name, obj.type, tail))
    strands = [s for s in _strands(obj) if len(s) >= 2]
    if not strands:
        raise MifOpError("'%s' has no hair strands (no curves, no hair particle system). %s" % (obj.name, tail))
    n_cards = min(take_int(params, "cards", default=24), len(strands))
    seg = take_int(params, "segments", default=6)
    if n_cards < 1 or not 1 <= seg <= 64:
        raise MifOpError("cards is 1 or more and segments 1-64. %s" % tail)
    taper = take_float(params, "taper", default=0.3)
    roots = [s[0] for s in strands]
    centre = sum(roots, Vector()) / len(roots)
    # k-means on the roots, seeded by the farthest-point rule so the clumps spread over the scalp
    seeds = [roots[0]]
    while len(seeds) < n_cards:
        seeds.append(max(roots, key=lambda r: min((r - s).length for s in seeds)))
    assign = [0] * len(roots)
    for _it in range(take_int(params, "iterations", default=8)):
        assign = [min(range(n_cards), key=lambda k: (r - seeds[k]).length) for r in roots]
        for k in range(n_cards):
            mem = [roots[i] for i, a in enumerate(assign) if a == k]
            if mem:
                seeds[k] = sum(mem, Vector()) / len(mem)

    def resample(s, m):
        L = [0.0]
        for a, b in zip(s, s[1:]):
            L.append(L[-1] + (b - a).length)
        tot = L[-1] or 1.0
        out = []
        for i in range(m + 1):
            t = tot * i / m
            j = max(0, min(len(s) - 2, next((q for q in range(len(L) - 1) if L[q + 1] >= t), len(L) - 2)))
            f = (t - L[j]) / max(L[j + 1] - L[j], 1e-12)
            out.append(s[j].lerp(s[j + 1], min(1.0, max(0.0, f))))
        return out
    atlas = params.get("atlas")
    if atlas is not None:
        if not isinstance(atlas, list) or len(atlas) != 2 or any(not isinstance(v, int) or v < 1 for v in atlas) \
                or atlas[0] * atlas[1] < n_cards:
            raise MifOpError("atlas is [cols, rows] holding every card (%d), got %r. %s" % (n_cards, atlas, tail))
        cols, rows = atlas
    else:
        cols = int(math.ceil(math.sqrt(n_cards)))
        rows = int(math.ceil(n_cards / cols))
    verts, faces, uvs = [], [], []
    made = 0
    for k in range(n_cards):
        mem = [strands[i] for i, a in enumerate(assign) if a == k]
        if not mem:
            continue
        rs = [resample(s, seg) for s in mem]
        guide = [sum((r[i] for r in rs), Vector()) / len(rs) for i in range(seg + 1)]
        spread = max(((r[0] - guide[0]).length for r in rs), default=0.0)
        w0 = take_float(params, "width", default=None) or max(2.0 * spread, 0.001)
        out_dir = (guide[0] - centre)
        out_dir = out_dir.normalized() if out_dir.length > 1e-9 else Vector((0, 0, 1))
        cell_u, cell_v = (made % cols) / cols, (made // cols) / rows
        base = len(verts)
        for i, p in enumerate(guide):
            tan = (guide[min(i + 1, seg)] - guide[max(i - 1, 0)])
            tan = tan.normalized() if tan.length > 1e-12 else Vector((0, 0, 1))
            side = tan.cross(out_dir)
            if side.length < 1e-6:
                side = tan.orthogonal()
            side.normalize()
            half = 0.5 * w0 * (1.0 + (taper - 1.0) * i / seg)
            verts += [p - side * half, p + side * half]
            v = i / seg
            uvs += [(cell_u, cell_v + v / rows), (cell_u + 1.0 / cols, cell_v + v / rows)]
        for i in range(seg):
            a = base + 2 * i
            faces.append((a, a + 1, a + 3, a + 2))
        made += 1
    card_name = take(params, "name", default="%s_Cards" % obj.name, kind=str)
    me = bpy.data.meshes.new(card_name)
    me.from_pydata([tuple(v) for v in verts], [], faces)
    uvl = me.uv_layers.new(name="UVMap")
    for f in me.polygons:
        for li in f.loop_indices:
            uvl.data[li].uv = uvs[me.loops[li].vertex_index]
    me.update()
    cards = bpy.data.objects.new(me.name, me)
    for c in obj.users_collection:
        c.objects.link(cards)
    return {"object": obj.name, "cardsObject": cards.name, "cards": made, "strands": len(strands),
            "requestedName": card_name, "nameWasSuffixed": cards.name != card_name,
            "triangles": 2 * len(faces), "segments": seg, "atlas": [cols, rows],
            "cellSize": [round(1.0 / cols, 6), round(1.0 / rows, 6)], "taper": taper,
            "note": "UV V runs 0 at the root to 1 at the tip inside each card's cell; bake_texture onto the cards bakes "
                    "the strands into the atlas"}


# --------------------------------------------------------------------------------------------- bake_texture_set

_BTS_KEYS = ("object", "types", "folder", "pattern", "width", "height", "samples", "margin", "uvLayer", "source",
             "extrusion", "maxRayDistance", "cage", "packOrm", "outOfProcess", "jobId", "greenChannel", "memoryLimitGB",
             "bakedMaterial", "assignBaked", "singleSlot", "computeBackend", "wait", "aoAgainst", "include",
             "pieces", "explode", "match", "fixNormals", "checkIslands")
_BTS_SUFFIX = {"AO": "AO", "NORMAL": "N", "DIFFUSE": "BC", "BASE_COLOR": "BC", "ROUGHNESS": "R", "METALLIC": "M",
               "EMIT": "E",
               "CURVATURE": "Curv", "ALPHA": "A", "MATERIAL_ID": "ID", "HEIGHT": "H", "POSITION": "Pos",
               "THICKNESS": "Thick", "COMBINED": "C", "GLOSSY": "G", "SHADOW": "S"}
_JOBS = {}
_JOB_SCRIPT = r'''
import json, sys, traceback
sys.path.insert(0, @@ADDON@@)
import bpy
from MifBlender import ops_material, ops_image, ops_render
spec = json.load(open(@@SPEC@@, encoding="utf-8"))
rows = []
used = {"backend": None, "device": "CPU", "devices": [], "note": "no computeBackend asked: the default, the CPU"}
if spec.get("computeBackend"):
    _sc = bpy.context.scene
    _sc.render.engine = "CYCLES"
    used = ops_render._still_backend_apply(_sc, ops_render._still_backend(_sc, {"computeBackend": spec["computeBackend"]}))
    used.pop("note", None)
for req in spec["bakes"]:
    _t0 = __import__("time").time()
    try:
        r = ops_material.op_bake_texture(req["params"])
        rows.append({"type": req["type"], "ok": True, "file": r.get("savedTo"), "signature": r.get("savedSignature"),
                     "seconds": round(__import__("time").time() - _t0, 2), "idlePieces": r.get("idlePieces"),
                     "warnings": r.get("warnings"), "islandCoverage": r.get("islandCoverage"),
                     "normalCheck": r.get("normalCheck"), "blobFix": r.get("blobFix")})
    except Exception as exc:
        rows.append({"type": req["type"], "ok": False, "error": str(exc)[:600]})
    json.dump({"rows": rows, "done": False, "computeUsed": used}, open(@@OUT@@, "w", encoding="utf-8"))
orm = None
if spec.get("orm"):
    try:
        r = ops_image.op_channel_pack_image(spec["orm"])
        orm = {"ok": True, "file": r.get("file"), "readBack": r.get("readBack")}
    except Exception as exc:
        orm = {"ok": False, "error": str(exc)[:600]}
json.dump({"rows": rows, "orm": orm, "done": True, "computeUsed": used}, open(@@OUT@@, "w", encoding="utf-8"))
'''


def _parse_match(match, obj, params, tail):
    """bake_texture_set match, checked with nothing made: {by: LOOSE_PARTS} (object and one source, split into their
    loose parts at bake time) or {by: NAME, lowTag, highTag} (scene objects paired now). minShare (LOOSE_PARTS): the
    share of a high part's box that must lie inside a low part's (default 0.5)."""
    if not isinstance(match, dict) or str(match.get("by", "")).upper() not in ("LOOSE_PARTS", "NAME"):
        raise MifOpError("match is {by: LOOSE_PARTS} or {by: NAME, lowTag, highTag}, got %r. %s" % (match, tail))
    by = str(match["by"]).upper()
    bad = sorted(set(match) - ({"by", "minShare"} if by == "LOOSE_PARTS" else {"by", "lowTag", "highTag"}))
    if bad:
        raise MifOpError("match %s takes no %s. %s" % (by, bad, tail))
    if by == "LOOSE_PARTS":
        src = params.get("source")
        if not isinstance(src, str):
            raise MifOpError("match LOOSE_PARTS splits object and ONE source (the joined high) into loose parts - "
                             "pass source as one name. %s" % tail)
        high = get_object(src, want_mesh=True)
        share = match.get("minShare", 0.5)
        if isinstance(share, bool) or not isinstance(share, (int, float)) or not 0 < share <= 1:
            raise MifOpError("match minShare is above 0 and at most 1, got %r. %s" % (share, tail))
        return {"by": by, "low": obj.name, "high": high.name, "minShare": float(share)}
    lt, ht = match.get("lowTag"), match.get("highTag")
    if not (isinstance(lt, str) and lt and isinstance(ht, str) and ht and lt != ht):
        raise MifOpError("match NAME pairs objects whose names differ by lowTag and highTag (e.g. '_low' and '_high'). "
                         "%s" % tail)
    if params.get("source") is not None:
        raise MifOpError("match NAME finds each low's source by name - leave source out. %s" % tail)
    lows = sorted(o.name for o in bpy.context.scene.objects if o.type == "MESH" and lt in o.name)
    pieces, unmatched = [], []
    for n in lows:
        h = bpy.data.objects.get(n.replace(lt, ht, 1))
        if h is None or h.type != "MESH":
            unmatched.append(n)
        else:
            pieces.append({"object": n, "source": h.name})
    highs = {p["source"] for p in pieces}
    lone_high = sorted(o.name for o in bpy.context.scene.objects if o.type == "MESH" and ht in o.name
                       and o.name not in highs)
    if not pieces:
        raise MifOpError("match NAME found no pair: no mesh named with '%s' has a twin named with '%s'. %s"
                         % (lt, ht, tail))
    if obj.name not in [p["object"] for p in pieces]:
        raise MifOpError("object '%s' is not one of the matched lows (%s) - it names the files and the material. %s"
                         % (obj.name, ", ".join(p["object"] for p in pieces[:10]), tail))
    pieces.sort(key=lambda p: p["object"] != obj.name)       # the named object first: bake_texture's object
    return {"by": by, "pieces": pieces, "report": {"by": by, "pairs": [[p["object"], p["source"]] for p in pieces],
                                                   "unmatchedLow": unmatched, "unmatchedHigh": lone_high}}


def _box(o):
    from mathutils import Vector                                # noqa: PLC0415
    pts = [o.matrix_world @ Vector(c) for c in o.bound_box]
    return [min(p[i] for p in pts) for i in range(3)], [max(p[i] for p in pts) for i in range(3)]


def _split_loose(obj, tag):
    """Temporary copies of obj's loose parts, in the scene collection, named tag_NNN."""
    from .ops_common import select_only, selection_restore, selection_snapshot   # noqa: PLC0415
    cp = obj.copy()
    cp.data = obj.data.copy()
    cp.name = tag
    bpy.context.scene.collection.objects.link(cp)
    snap = selection_snapshot()
    try:
        select_only([cp])
        bpy.context.view_layer.objects.active = cp
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.separate(type="LOOSE")
        bpy.ops.object.mode_set(mode="OBJECT")
        parts = [o for o in bpy.context.view_layer.objects if o.select_get()]
    finally:
        selection_restore(snap)
    for k, o in enumerate(sorted(parts, key=lambda o: tuple(round(v, 4) for v in _box(o)[0]))):
        o.name = "%s_%03d" % (tag, k)
        o.hide_render = o.hide_viewport = False
    return sorted(parts, key=lambda o: o.name)


def _build_match(plan):
    """(pieces, temporary objects, originals hidden, the report) for a LOOSE_PARTS match; NAME's pieces as parsed."""
    if plan["by"] == "NAME":
        return plan["pieces"], [], [], plan["report"]
    low, high = bpy.data.objects[plan["low"]], bpy.data.objects[plan["high"]]
    temps = []
    try:
        lows = _split_loose(low, "MifMatchL")
        temps += lows
        highs = _split_loose(high, "MifMatchH")
        temps += highs
    except Exception:
        _drop_match(temps, [])
        raise
    boxes = {o.name: _box(o) for o in lows + highs}
    span = max(max(b[1][i] - b[0][i] for b in boxes.values()) for i in range(3)) or 1.0
    pad = 0.01 * span                                           # a flat part still has a volume

    def vol(lo, hi):
        return max(0.0, hi[0] - lo[0]) * max(0.0, hi[1] - lo[1]) * max(0.0, hi[2] - lo[2])
    got = {o.name: [] for o in lows}
    shares = {}
    for h in highs:
        hl, hh = [v - pad for v in boxes[h.name][0]], [v + pad for v in boxes[h.name][1]]
        best, best_share = None, 0.0
        for lo_ in lows:
            ll, lh = [v - pad for v in boxes[lo_.name][0]], [v + pad for v in boxes[lo_.name][1]]
            inter = vol([max(a, b) for a, b in zip(hl, ll)], [min(a, b) for a, b in zip(hh, lh)])
            share = inter / max(vol(hl, hh), 1e-18)
            if share > best_share:
                best, best_share = lo_, share
        if best is not None and best_share >= plan["minShare"]:
            got[best.name].append(h.name)
            shares[h.name] = round(best_share, 3)
    pieces = [{"object": n, "source": s} for n, s in got.items() if s]
    unmatched_low = [n for n, s in got.items() if not s]
    unmatched_high = [h.name for h in highs if h.name not in shares]
    hidden = []
    for o in (low, high):
        if not o.hide_render:
            o.hide_render = True                                 # the joined originals sit out the bake
            hidden.append(o)
    report = {"by": "LOOSE_PARTS", "lowParts": len(lows), "highParts": len(highs),
              "pairs": [[p["object"], p["source"]] for p in pieces][:200], "unmatchedLow": unmatched_low,
              "unmatchedHigh": unmatched_high, "minShare": plan["minShare"],
              "note": "the parts are temporary copies of '%s' and '%s', removed after the bake; their UVs are the low's, "
                      "so the maps are the low's" % (low.name, high.name)}
    if not pieces:
        _drop_match(temps, hidden)
        raise MifOpError("match LOOSE_PARTS paired no part: %d low and %d high parts, none of the highs lies %.0f%% "
                         "inside a low part's box. NOTHING was baked." % (len(lows), len(highs), 100 * plan["minShare"]))
    return pieces, temps, hidden, report


def _drop_match(temps, hidden):
    from .ops_common import remove_objects                     # noqa: PLC0415
    meshes = [o.data for o in temps]
    present = set(bpy.data.objects.keys())
    remove_objects([o for o in temps if o.name in present])
    for m in meshes:
        try:
            if m.users == 0:
                bpy.data.meshes.remove(m)
        except ReferenceError:
            pass
    for o in hidden:
        o.hide_render = False


def _sweep_bake_set_folders():
    """Remove the bake_set folders whose owner Blender (the pid in the folder's name) has gone - nothing can poll them."""
    root = os.path.join(tempfile.gettempdir(), "MifBridge", "bake_set")
    try:
        names = os.listdir(root)
    except OSError:
        return 0
    gone = 0
    for name in names:
        parts = name.split("_")
        if len(parts) < 3 or not parts[2].isdigit():
            continue
        pid = int(parts[2])
        if pid == os.getpid() or childproc.alive(pid):
            continue
        path = os.path.join(root, name)
        try:
            if time.time() - os.path.getmtime(path) < 60:
                continue
        except OSError:
            continue
        shutil.rmtree(path, ignore_errors=True)
        gone += 0 if os.path.exists(path) else 1
    return gone


def _take_slot(label, wait_s):
    """(slot id or None, seconds waited, the jobs holding every slot). With wait_s, queue for a slot that long - never
    past the caller's deadline less a margin, so the reply still reaches the caller."""
    started = time.time()
    sid, held = childproc.acquire("blender bake", label)
    if sid is not None or not wait_s:
        return sid, 0.0, held
    end = started + wait_s
    dl = caller_deadline()
    if dl is not None:
        end = min(end, dl - 20.0)
    while sid is None and time.time() < end:
        time.sleep(min(1.0, max(0.05, end - time.time())))
        sid, held = childproc.acquire("blender bake", label)
    return sid, time.time() - started, held


def _parse_wait(params, tail):
    """wait: true (up to 600 s) or seconds, for the heavy-job slot; inside the caller's deadline either way."""
    raw = params.get("wait")
    if raw is None or raw is False:
        return 0.0
    if raw is True:
        return 600.0
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not 0 < raw <= 3600:
        raise MifOpError("wait is true or the seconds to queue for the heavy-job slot (above 0, at most 3600), got %r. %s"
                         % (raw, tail))
    return float(raw)


def _bake_set_status(job_id):
    job = _JOBS.get(job_id)
    if job is None:
        return {"unknownJob": True, "jobId": job_id, "knownJobs": sorted(_JOBS),
                "error": "no bake-set job '%s' in this Blender (never started here, or Blender restarted)" % job_id}
    code = job["popen"].poll()
    try:
        res = json.load(open(job["out"], encoding="utf-8"))
    except (OSError, ValueError):
        res = {"rows": [], "done": False}
    rows = res.get("rows") or []
    # mapsDone COUNTS THE MAPS THAT BAKED (FOUND 2026-11-18): it counted rows, so five failed maps read 5/5
    failed = [r for r in rows if not r.get("ok")]
    out = {"jobId": job_id, "running": code is None, "exitCode": code, "mapsExpected": len(job["types"]),
           "mapsDone": len(rows) - len(failed), "mapsTried": len(rows), "mapsFailed": [r.get("type") for r in failed],
           "maps": rows, "orm": res.get("orm"),
           "elapsedSeconds": round(time.time() - job["started"], 1), "log": job["log"],
           "computeUsed": res.get("computeUsed")}
    if job.get("watch") is not None:
        out["job"] = job["watch"].report()
    if code is not None:
        fh = job.pop("logHandle", None)
        if fh is not None:
            fh.close()
        out["complete"] = bool(res.get("done")) and code == 0 and all(r.get("ok") for r in res.get("rows") or [])
        if out["complete"] and job.get("baked") is not None:
            # THE BAKED MATERIAL IS MADE HERE, in this Blender, on the first status call that sees the child finish -
            # the child baked a copy of the session and its materials die with it
            if job.get("bakedResult") is None:
                try:
                    job["bakedResult"] = _wire_baked_material(job["baked"], job["files"],
                                                              (res.get("orm") or {}).get("file")
                                                              if (res.get("orm") or {}).get("ok") else None)
                except MifOpError as exc:
                    job["bakedResult"] = {"error": str(exc)}
            out["bakedMaterial"] = job["bakedResult"]
        if not out["complete"]:
            msg = "the bake process ended (exit %s) with %d of %d maps done%s%s" % (
                code, out["mapsDone"], len(job["types"]), "" if res.get("done") else " - read the log",
                "".join("; %s failed: %s" % (r.get("type"), str(r.get("error") or "")[:300]) for r in failed))
            out["error"] = msg
            # A FAILED JOB IS ok:false, the status riding in the refusal's payload: it read ok:true with every map failed
            exc = MifOpError(msg)
            exc.payload = out
            raise exc
    return out


def op_bake_texture_set(params):
    """Bake several maps from one setup (high-to-low with source, or self) in one job - by default OUT OF PROCESS in a
    second Blender, so a large set does not hold this one past the bridge's ceiling - with a jobId and a status call.

    params:
      object (str), types (list)   the low poly and the maps: AO, NORMAL, DIFFUSE, ROUGHNESS, METALLIC, EMIT, CURVATURE,
                                   ALPHA, MATERIAL_ID, HEIGHT, POSITION, THICKNESS, COMBINED, GLOSSY, SHADOW
      folder (str), pattern (str)  where and how the files are named: {object} and {suffix} (default T_{object}_{suffix};
                                   suffixes BC, N, AO, R, M, E, ... ) - PNG, EXR for HEIGHT and POSITION
      width, height, samples, margin, uvLayer, source, extrusion, maxRayDistance, cage, greenChannel   once for every map,
                                   as bake_texture takes them; include too, and aoAgainst for the AO map
                                   (SOURCES, the default with a source: every other geometry object hidden for it)
      packOrm (bool)               also write T_{object}_ORM from the AO, ROUGHNESS and METALLIC maps (constants for a map
                                   not baked: occlusion 1, roughness 0.5, metallic 0)
      outOfProcess (bool)          default true: the session is saved as a copy to a temp .blend and a second Blender bakes
                                   it (one of the machine-wide heavy-job slots, under a memory ceiling - memoryLimitGB);
                                   false bakes here, now
      jobId (str)                  alone: the job's status - maps done, each file and its pixel signature, the ORM
      bakedMaterial (str)          also make a material of that name with the baked maps wired into its Principled
                                   inputs (BASE_COLOR / DIFFUSE to base color, ROUGHNESS, METALLIC, NORMAL, EMIT, ALPHA) -
                                   what export_material_textures and the send to Unreal carry; a procedural material is
                                   carried as nothing. Out of process it is made by the status call that sees the job
                                   finish. assignBaked (bool) also puts it on the object. It reads the UV layer the bake
                                   wrote: uvMap in the reply names it when that is not the render-active one.
      singleSlot (bool)            with assignBaked, default true: the object is left with ONE slot holding the baked
                                   material - every face moved to it, the other slots removed - because the bake already
                                   carries each old slot's look and a game export wants one slot. false keeps every slot
                                   (each holding the baked material) and the faces' slot indices. A mesh another object
                                   also uses keeps its slots (removing them would change that object too): singleSlotNote
      computeBackend (str)         the job's device, the names render_still takes (CPU, OPTIX, CUDA, HIP, ONEAPI, METAL;
                                   NONE is Blender's own word for the CPU).
                                   Default: the CPU, as before. Checked before the session copy; a backend with no device
                                   here is refused, never a quiet CPU bake. The status reply's computeUsed says what ran.
      wait (bool or seconds)       queue for the machine-wide heavy-job slot (MIF_MAX_CHILD_JOBS) instead of refusing while
                                   another job holds it: true waits up to 600 s, a number that long, never past the
                                   caller's deadline less 20 s; slotWait in the reply says how long. A pool worker takes
                                   the same slot - its bake is one more Blender's memory on the same machine.
    BASE_COLOR is the albedo: a DIFFUSE bake is the diffuse color pass, base color x (1 - metallic), so a worn metal's
    bare edges come out black in it - bake BASE_COLOR for a metal.
    """
    reject_unknown(params, _BTS_KEYS, "bake_texture_set")
    tail = "NOTHING was baked."
    if params.get("jobId") is not None:
        extra = [k for k in params if k != "jobId"]
        if extra:
            raise MifOpError("jobId asks a job's status and takes nothing else (got %s). %s" % (extra, tail))
        return _bake_set_status(take(params, "jobId", kind=str))
    from .ops_material import _BAKE_TYPES                      # noqa: PLC0415
    obj = get_object(take(params, "object", required=True, kind=str), want_mesh=True)
    types = params.get("types")
    if not isinstance(types, list) or not types:
        raise MifOpError("types is a non-empty list of bake types. %s" % tail)
    types = [str(t).upper() for t in types]
    bad = [t for t in types if t not in _BAKE_TYPES or t.startswith("_")]
    if bad:
        raise MifOpError("types %s are not bake types; use %s. %s"
                         % (bad, ", ".join(t for t in _BTS_SUFFIX), tail))
    if len(set(types)) != len(types):
        raise MifOpError("types names a map twice. %s" % tail)
    if "DIFFUSE" in types and "BASE_COLOR" in types:
        raise MifOpError("DIFFUSE and BASE_COLOR both write the BC map - BASE_COLOR is the albedo (a DIFFUSE bake is "
                         "black wherever the surface is metal); pass one. %s" % tail)
    # computeBackend: the job's Blender starts with factory preferences, so it bakes on the CPU unless told. The name is
    # checked here against this machine's devices, before the session copy; a device-less backend would bake on the CPU
    # in silence.
    backend = None
    if params.get("computeBackend") is not None:
        from types import SimpleNamespace                      # noqa: PLC0415
        from .ops_render import _still_backend                 # noqa: PLC0415
        if not take_bool(params, "outOfProcess", default=True):
            raise MifOpError("computeBackend is for the out-of-process job; this Blender's own device is set with "
                             "render_still computeBackend. %s" % tail)
        # a bake runs Cycles whatever the scene renders with, so the check is asked as for a Cycles scene; nothing is
        # written, and a backend with no device on this machine is refused here, naming the ones that have one
        try:
            checked = _still_backend(SimpleNamespace(render=SimpleNamespace(engine="CYCLES")),
                                     {"computeBackend": params["computeBackend"]})
        except MifOpError as exc:
            raise MifOpError(str(exc).replace("NOTHING was rendered.", "") + " " + tail) from None
        backend = checked[1]
    # A HIDDEN TARGET OR SOURCE IS REFUSED HERE, before the session copy (FOUND 2026-11-18): the job started anyway, every
    # map failed in the child after a minute and the status read ok
    from .ops_material import bake_visibility_problems         # noqa: PLC0415
    raw_src = params.get("source")
    src_names = [raw_src] if isinstance(raw_src, str) else (raw_src if isinstance(raw_src, list) else [])
    probs = bake_visibility_problems(obj, "target")
    for row in params.get("pieces") if isinstance(params.get("pieces"), list) else []:
        if isinstance(row, dict):
            src_names = src_names + ([row["object"]] if isinstance(row.get("object"), str) else []) + (
                [row["source"]] if isinstance(row.get("source"), str) else
                [s for s in row.get("source") or [] if isinstance(s, str)])
    for n in src_names:
        so = bpy.data.objects.get(n) if isinstance(n, str) else None
        if so is not None:
            probs += bake_visibility_problems(so, "source")
    if probs:
        raise MifOpError("a bake cannot use a hidden object: %s. %s" % ("; ".join(probs), tail))
    # PER-PART PAIRING (1.4, FOUND 2026-11-18): a mech of ~110 interpenetrating parts baked as one low and one high with
    # one extrusion caught its neighbours' highs (solid blocks in the normal map; a cape's red on armour), and the set
    # had no pieces. pieces and explode go to every map as bake_texture takes them; match pairs them here -
    # LOOSE_PARTS splits object and source into their loose parts (temporary copies, removed after) and gives each low
    # part the high parts that lie mostly inside its box; NAME pairs scene objects whose names differ by lowTag/highTag.
    pieces_in, match = params.get("pieces"), params.get("match")
    if pieces_in is not None or match is not None:
        clash = [k for k in ("source", "cage") if params.get(k) is not None and pieces_in is not None]
        if clash or (pieces_in is not None and match is not None):
            raise MifOpError("pieces names each low part's own source and cage; match builds the pieces - pass pieces, "
                             "or match with object and source, not %s. %s" % (clash or ["pieces", "match"], tail))
        if "HEIGHT" in types:
            raise MifOpError("a HEIGHT map is composed of three bakes of one object, not per piece. %s" % tail)
    match_plan = _parse_match(match, obj, params, tail) if match is not None else None
    if params.get("explode") is not None and pieces_in is None and match is None:
        raise MifOpError("explode moves the pieces apart for the bake - pass pieces or match too. %s" % tail)
    baked = _baked_plan(params, obj, types, tail)
    folder = _folder(params, tail)
    pattern = take(params, "pattern", default="T_{object}_{suffix}", kind=str)
    shared = {k: params[k] for k in ("width", "height", "samples", "margin", "uvLayer", "source", "extrusion",
                                     "maxRayDistance", "cage", "include", "pieces", "explode") if params.get(k) is not None}
    if match_plan is not None:
        shared.pop("source", None)
    if params.get("aoAgainst") is not None and ("AO" not in types or not (shared.get("source") or pieces_in or match)):
        raise MifOpError("aoAgainst chooses what the AO map's rays may hit in a high-to-low bake - pass AO in types and "
                         "a source. %s" % tail)
    bakes = []
    for t in types:
        ext = ".exr" if t in ("HEIGHT", "POSITION") else ".png"
        fp = os.path.join(folder, pattern.replace("{object}", obj.name).replace("{suffix}", _BTS_SUFFIX[t]) + ext)
        # BEFORE ANY BAKE: the pattern is the caller's, and a control character in it would otherwise surface in
        # the child, after the session copy and the launch
        check_output_path(fp, fp, "baked")
        p = dict(shared, type=t, filepath=fp, imageName="MifSet_%s" % t)
        if not shared.get("pieces") and match_plan is None:
            p["object"] = obj.name
        if t == "AO" and params.get("aoAgainst") is not None:
            p["aoAgainst"] = params["aoAgainst"]
        # EVERY MAP'S ISLANDS ARE CHECKED (1.4): whole islands came back at the background and the set answered ok
        p["checkIslands"] = take_bool(params, "checkIslands", default=True)
        fix_n = take_bool(params, "fixNormals", default=False)
        if t == "NORMAL" and fix_n:
            p["blobFix"] = True
        if t == "NORMAL" and params.get("greenChannel") is not None:
            p["greenChannel"] = params["greenChannel"]
        if t in ("HEIGHT", "POSITION"):
            p["float"] = True
        if t == "HEIGHT" and not shared.get("source"):
            raise MifOpError("a HEIGHT map measures a high poly - pass source. %s" % tail)
        bakes.append({"type": t, "params": p})
    orm = None
    if take_bool(params, "packOrm", default=False):
        files = {b["type"]: b["params"]["filepath"] for b in bakes}
        chans = {}
        for ch, t, dflt in (("R", "AO", 1.0), ("G", "ROUGHNESS", 0.5), ("B", "METALLIC", 0.0)):
            chans[ch] = {"image": os.path.basename(files[t]), "channel": "R"} if t in files else dflt
        orm = {"channels": chans, "name": "MifSet_ORM",
               "filepath": os.path.join(folder, pattern.replace("{object}", obj.name).replace("{suffix}", "ORM") + ".png"),
               "_load": {c: files[t] for c, t in (("R", "AO"), ("G", "ROUGHNESS"), ("B", "METALLIC")) if t in files}}
        check_output_path(orm["filepath"], orm["filepath"], "baked")
    # READ BEFORE THE CHILD STARTS: memoryLimitGB was read after Popen, so a malformed one refused with a second
    # Blender already baking and holding the child slot, and nothing told the caller its jobId
    mem = take_float(params, "memoryLimitGB", default=None)
    mem_bytes = None if mem is None else childproc.memory_limit_bytes(mem)
    wait_s = _parse_wait(params, tail)
    oop = take_bool(params, "outOfProcess", default=True)
    if wait_s and not oop:
        raise MifOpError("wait queues for the out-of-process job's slot; outOfProcess false bakes here, now. %s" % tail)
    if backend is not None:
        for b in bakes:
            b["params"]["device"] = "GPU" if backend != "NONE" else "CPU"
    if not take_bool(params, "outOfProcess", default=True):
        from .ops_image import op_channel_pack_image           # noqa: PLC0415
        from .ops_material import op_bake_texture              # noqa: PLC0415
        rows = []
        # PROGRESS PER MAP, IN PROCESS (FOUND 2026-11-18): five maps from a 349k-vertex source held the worker 1,106 s
        # with nothing to read. This Blender answers nothing while it bakes, so each finished map is written to a file
        # any process can read - %TEMP%/MifBridge/bake_set/inprocess_<this Blender's pid>.json - and each row has its
        # seconds.
        prog_path = os.path.join(tempfile.gettempdir(), "MifBridge", "bake_set", "inprocess_%d.json" % os.getpid())
        t_all = time.time()

        def progress(state, current=None):
            try:
                os.makedirs(os.path.dirname(prog_path), exist_ok=True)
                with open(prog_path, "w", encoding="utf-8") as fh:
                    json.dump({"object": obj.name, "state": state, "mapsExpected": len(bakes), "mapsDone": len(rows),
                               "maps": rows, "current": current, "elapsedSeconds": round(time.time() - t_all, 1),
                               "pid": os.getpid(), "at": time.strftime("%Y-%m-%dT%H:%M:%S")}, fh)
            except OSError:
                pass
        temps, hid, match_rep = [], [], None
        try:
            if match_plan is not None:
                mp, temps, hid, match_rep = _build_match(match_plan)
                for b in bakes:
                    b["params"]["pieces"] = mp
            for b in bakes:
                progress("baking", b["type"])
                t0 = time.time()
                r = op_bake_texture(b["params"])
                rows.append({"type": b["type"], "ok": True, "file": r.get("savedTo"),
                             "signature": r.get("savedSignature"), "seconds": round(time.time() - t0, 2),
                             "warnings": r.get("warnings"), "islandCoverage": r.get("islandCoverage"),
                             "normalCheck": r.get("normalCheck"), "blobFix": r.get("blobFix")})
                im = bpy.data.images.get(r["image"])
                if im is not None:
                    im.name = os.path.basename(b["params"]["filepath"])
                if r.get("idlePieces"):
                    rows[-1]["idlePieces"] = r["idlePieces"]
        except MifOpError as exc:
            progress("failed", str(exc)[:300])
            raise
        finally:
            _drop_match(temps, hid)
        progress("done")
        orm_r = None
        if orm is not None:
            orm.pop("_load")
            orm_r = op_channel_pack_image(orm)
            orm_r = {"ok": True, "file": orm_r.get("file"), "readBack": orm_r.get("readBack")}
        out = {"object": obj.name, "outOfProcess": False, "maps": rows, "orm": orm_r, "complete": True,
               "seconds": round(time.time() - t_all, 2), "progressFile": prog_path, "match": match_rep}
        if baked is not None:
            out["bakedMaterial"] = _wire_baked_material(baked, {r["type"]: r["file"] for r in rows},
                                                        (orm_r or {}).get("file"))
        return out
    # OUT OF PROCESS: a copy of THIS session's state (not the file on disk) is what the child bakes.
    # THE SLOT IS TAKEN FIRST (FOUND 2026-11-18): the copy, job.py and spec.json were written before childproc.acquire,
    # and a busy refusal left them - one worker retrying every 20 s left 58 folders, and bake_set held 34 GB across
    # sessions. Now the slot comes first (queued for with wait), every refusal after the folder is made removes it, and
    # each start sweeps the folders whose owner Blender has gone.
    _sweep_bake_set_folders()
    sid, waited, held = _take_slot(obj.name, wait_s)
    if sid is None:
        raise MifOpError(childproc.busy_message("bake_texture_set", held) + (
            " Waited %.0f s for it." % waited if waited else " Pass wait true (or a number of seconds) to queue for the "
            "slot inside this call.") + " outOfProcess false bakes here, now, outside the limit - it holds this Blender "
            "for the whole bake, beside the job holding the slot, which is what the limit is there to stop. " + tail)
    stamp = "%s_%d" % (time.strftime("%Y%m%d_%H%M%S"), os.getpid())
    root = os.path.join(tempfile.gettempdir(), "MifBridge", "bake_set")
    n_ = 1
    while os.path.exists(os.path.join(root, stamp)):          # two starts in one second shared a folder
        n_ += 1
        stamp = "%s_%d_%d" % (time.strftime("%Y%m%d_%H%M%S"), os.getpid(), n_)
    work = os.path.join(root, stamp)
    try:
        return _start_bake_job(work, stamp, sid, obj, types, bakes, orm, backend, baked, mem_bytes, waited, tail,
                               match_plan)
    except BaseException:
        childproc.release(sid)
        shutil.rmtree(work, ignore_errors=True)
        raise


def _start_bake_job(work, stamp, sid, obj, types, bakes, orm, backend, baked, mem_bytes, waited, tail, match_plan=None):
    """Write the session copy and the job into `work` and start the child on slot `sid` - the caller frees the slot and
    removes the folder when this raises."""
    os.makedirs(work, exist_ok=True)
    blend = os.path.join(work, "session.blend")
    # a LOOSE_PARTS match's temporary parts exist for the copy only: made, saved with it, and removed here
    temps, hid, match_rep = [], [], None
    try:
        if match_plan is not None:
            mp, temps, hid, match_rep = _build_match(match_plan)
            for b in bakes:
                b["params"]["pieces"] = mp
        bpy.ops.wm.save_as_mainfile(filepath=blend, copy=True)
    except RuntimeError as exc:
        raise MifOpError("could not write the session copy for the bake (%s). %s" % (exc, tail))
    finally:
        _drop_match(temps, hid)
    snapped = time.time()
    # NEVER STARTED FOR A CALLER WHO HAS GONE (FOUND 2026-11-07). The dispatcher refuses a request that arrives after its
    # caller's deadline; this asks again once the copy is on disk, because writing a large session takes seconds and the
    # caller may have given up during it - and then, told "timed out", changed the scene the child would bake.
    late = caller_gave_up(snapped)
    if late is not None:
        raise MifOpError("the caller stopped waiting %.1f s ago, while the session copy was being written - no bake was "
                         "started, and the copy was deleted. %s" % (late, tail))
    if orm is not None:
        # the child loads the baked maps by path before packing
        load = orm.pop("_load")
        orm["channels"] = {c: ({"image": os.path.basename(load[c]), "channel": "R"} if c in load else v)
                           for c, v in orm["channels"].items()}
    spec = {"bakes": bakes, "orm": orm, "computeBackend": backend}
    spec_path, out_path = os.path.join(work, "spec.json"), os.path.join(work, "result.json")
    json.dump(spec, open(spec_path, "w", encoding="utf-8"))
    addon = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = os.path.join(work, "job.py")
    body = (_JOB_SCRIPT.replace("@@ADDON@@", repr(addon)).replace("@@SPEC@@", repr(spec_path))
            .replace("@@OUT@@", repr(out_path)))
    if orm is not None:
        # each baked map is renamed to its file's name in the child, so the packer finds it by that name
        body = body.replace('rows.append({"type": req["type"], "ok": True,',
                            'bpy.data.images[r["image"]].name = __import__("os").path.basename(req["params"]["filepath"])'
                            '\n        rows.append({"type": req["type"], "ok": True,')
    open(script, "w", encoding="utf-8").write(body)
    exe = bpy.app.binary_path
    log_path = os.path.join(work, "bake.log")
    try:
        log_fh = open(log_path, "wb")
        popen = subprocess.Popen([exe, "-b", blend, "--factory-startup", "--python", script],
                                 **childproc.hidden(dict(stdout=log_fh, stderr=subprocess.STDOUT,
                                                         stdin=subprocess.DEVNULL, cwd=work)))
    except Exception as exc:                                    # noqa: BLE001
        raise MifOpError("could not start %s (%s). %s" % (exe, exc, tail))
    watch = childproc.watch(popen, sid, mem_bytes, "blender bake")
    job_id = "bakeset_%s" % stamp
    _JOBS[job_id] = {"popen": popen, "watch": watch, "logHandle": log_fh, "log": log_path, "out": out_path,
                     "types": types, "started": time.time(), "baked": baked,
                     "files": {b["type"]: b["params"]["filepath"] for b in bakes}}
    dl = caller_deadline()
    return {"jobId": job_id, "object": obj.name, "outOfProcess": True, "pid": popen.pid, "maps": types,
            "files": {b["type"]: b["params"]["filepath"] for b in bakes},
            "orm": orm["filepath"] if orm else None, "sessionCopy": blend, "log": log_path,
            # WHICH STATE THE JOB BAKES, said in the reply: the copy was written before this answer, so nothing the
            # caller changes after it reaches the bake
            "snapshot": {"takenBeforeReply": True, "at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(snapped)),
                         "callerDeadline": (time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(dl))
                                            if dl is not None else None),
                         "secondsToSpare": round(dl - snapped, 1) if dl is not None else None,
                         "note": "the job bakes this session as it was at 'at'; a later change is not in it. A start "
                                 "that reaches Blender after its caller's deadline is refused and starts nothing."},
            "bakedMaterial": ({"name": baked["name"], "pending": "made by the status call that sees the job finish"}
                              if baked is not None else None),
            "slotWait": {"waitedSeconds": round(waited, 1)} if waited else None, "match": match_rep,
            "note": "baking in a second Blender from a copy of this session; poll bake_texture_set {jobId}"}


# THE BAKED MATERIAL (1.4, 2026-10-30). A procedural look - set_material_properties' weathering, a noise in the
# roughness - reaches Unreal as nothing: export_material_textures carries images and lists node chains in notCarried.
# The bake makes the images; this makes the material that reads them, so the send carries the look.
_BAKED_INPUT = {"BASE_COLOR": "baseColor", "DIFFUSE": "baseColor", "ROUGHNESS": "roughness", "METALLIC": "metallic",
                "NORMAL": "normal", "EMIT": "emissive", "ALPHA": "alpha"}


def _baked_plan(params, obj, types, tail):
    name = take(params, "bakedMaterial", default=None, kind=str)
    assign = take_bool(params, "assignBaked", default=False)
    single = take_bool(params, "singleSlot") if params.get("singleSlot") is not None else None
    if single is not None and not assign:
        raise MifOpError("singleSlot says how assignBaked leaves the object's slots - pass assignBaked true too. %s"
                         % tail)
    if name is None:
        if assign:
            raise MifOpError("assignBaked puts the baked material on the object - pass bakedMaterial too. %s" % tail)
        return None
    name = name.strip()
    if not name:
        raise MifOpError("bakedMaterial is the new material's name, got an empty one. %s" % tail)
    if bpy.data.materials.get(name) is not None:
        raise MifOpError("a material named '%s' already exists - bakedMaterial makes a new one; pick another name. %s"
                         % (name, tail))
    wired = [t for t in types if t in _BAKED_INPUT]
    if not wired:
        raise MifOpError("bakedMaterial wires %s into a material, and types has none of them. %s"
                         % (", ".join(sorted(_BAKED_INPUT)), tail))
    green = str(params.get("greenChannel") or "OPENGL").upper()
    # THE MATERIAL READS THE LAYER THE BAKE WROTE. With no UV Map node an image follows the render-active layer, so a bake onto a lightmap
    # layer came back as stripes on any mesh whose first layer tiles
    baked_on = params.get("uvLayer") or (obj.data.uv_layers.active.name if obj.data.uv_layers.active else None)
    render_on = next((u.name for u in obj.data.uv_layers if u.active_render), None)
    return {"name": name, "object": obj.name, "assign": assign, "single": single is not False, "green": green,
            "uvMap": baked_on if baked_on != render_on else None}


def _wire_orm(mat, orm_file, plan):
    """The ORM's G and B into Roughness and Metallic through a Separate Color node, and R, the occlusion, into the glTF
    occlusion slot (a Principled BSDF has no AO input). The image is loaded once, as data."""
    from .ops_material import _principled_of               # noqa: PLC0415
    nt = mat.node_tree
    bsdf = _principled_of(mat)
    want = os.path.normcase(os.path.abspath(orm_file))
    im = next((i for i in bpy.data.images if i.filepath and
               os.path.normcase(os.path.abspath(bpy.path.abspath(i.filepath))) == want), None)
    if im is None:
        im = bpy.data.images.load(orm_file, check_existing=True)
    offered = {i.identifier for i in bpy.types.ColorManagedInputColorspaceSettings.bl_rna.properties["name"].enum_items}
    data = next((w for w in ("Non-Color", "Non-Colour", "Raw", "Generic Data") if w in offered), None)
    if data is not None:
        im.colorspace_settings.name = data
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.name = tex.label = "Baked ORM"
    tex.image = im
    tex.location = (bsdf.location.x - 700, bsdf.location.y - 250)
    try:
        sep = nt.nodes.new("ShaderNodeSeparateColor")            # 3.3 and later
        outs = sep.outputs["Red"], sep.outputs["Green"], sep.outputs["Blue"]
    except RuntimeError:
        sep = nt.nodes.new("ShaderNodeSeparateRGB")
        outs = sep.outputs["R"], sep.outputs["G"], sep.outputs["B"]
    sep.name = sep.label = "Baked ORM split"
    sep.location = (bsdf.location.x - 350, bsdf.location.y - 250)
    nt.links.new(tex.outputs["Color"], sep.inputs[0])
    nt.links.new(outs[1], bsdf.inputs["Roughness"])
    nt.links.new(outs[2], bsdf.inputs["Metallic"])
    # R, the occlusion: no Principled input takes it, so it goes to the "glTF Material Output" group's Occlusion, the
    # one place a .glb carries it from (gltf_occlusion); an FBX has no occlusion slot either way
    from .ops_material import gltf_occlusion               # noqa: PLC0415
    occ = gltf_occlusion(mat, outs[0])
    if plan.get("uvMap"):
        uv = nt.nodes.new("ShaderNodeUVMap")
        uv.uv_map = plan["uvMap"]
        uv.location = (tex.location.x - 250, tex.location.y)
        nt.links.new(uv.outputs["UV"], tex.inputs["Vector"])
    return {"file": orm_file, "image": im.name, "colorSpace": im.colorspace_settings.name,
            "channels": {"R": "occlusion (glTF Occlusion, %s)" % occ, "G": "roughness", "B": "metallic"}}


def _wire_baked_material(plan, files, orm_file=None):
    """Create plan['name'] and wire each baked file into its Principled input; assign it when asked. With orm_file, the
    material reads occlusion, roughness and metallic off the packed ORM and the separate AO, R and M maps are not referenced."""
    from .ops_material import op_create_material, op_set_material_texture   # noqa: PLC0415
    made = op_create_material({"name": plan["name"]})
    mat = bpy.data.materials.get(made["name"])
    wired, skipped = {}, {}
    try:
        for t, fp in files.items():
            which = _BAKED_INPUT.get(t)
            if t == "AO" and orm_file:
                # WITH A PACKED ORM ITS RED IS THE OCCLUSION (merge of the two 2026-11-26 hand-off fixes): _wire_orm
                # sends the ORM's R to the glTF occlusion slot, so the separate AO map is not referenced - one image
                # for occlusion, roughness and metallic, as the packed set was made for
                skipped[t] = "read from the packed ORM's red instead"
                continue
            if t == "AO":
                # AO INTO glTF's OCCLUSION SLOT (FOUND 2026-11-18): a Principled BSDF has no AO input, so it goes to the
                # "glTF Material Output" group, which a .glb writes as occlusionTexture (an FBX has no such slot)
                from .ops_material import gltf_occlusion                       # noqa: PLC0415
                r = op_set_material_texture({"material": mat.name, "file": fp, "nodeName": "Baked AO", "link": False,
                                             "colorspace": "Non-Color"})
                wired["ao"] = {"map": t, "file": fp, "image": r.get("image"), "colorSpace": r.get("colorSpace"),
                               "to": "glTF Occlusion (%s)" % gltf_occlusion(mat, mat.node_tree.nodes[r["imageNode"]]
                                                                        .outputs["Color"])}
                continue
            if which is None:
                skipped[t] = "no Principled input takes a %s map" % t
                continue
            if orm_file and t in ("ROUGHNESS", "METALLIC"):
                # THE MATERIAL READS THE ORM IT PACKED (FOUND 2026-11-18): it read the separate R and M maps, so the
                # first export of a packed set shipped four 2048 maps where two would do
                skipped[t] = "read from the packed ORM instead"
                continue
            call = {"material": mat.name, "file": fp, "input": which, "nodeName": "Baked %s" % t}
            if plan.get("uvMap"):
                call["uvMap"] = plan["uvMap"]
            # IN PROCESS THE BAKED IMAGE IS ALREADY HERE: wired as it is rather than loaded a second time. Found by its
            # FILE - its name is the file's only until an earlier bake of the same object holds that name
            want = os.path.normcase(os.path.abspath(fp))
            held = next((im for im in reversed(list(bpy.data.images)) if im.filepath and
                         os.path.normcase(os.path.abspath(bpy.path.abspath(im.filepath))) == want), None)
            if held is not None:
                call.pop("file")
                call["image"] = held.name
            if t == "NORMAL":
                call["normalConvention"] = "DIRECTX" if plan["green"] == "DIRECTX" else "OPENGL"
            if t == "ALPHA":
                # AN ALPHA BAKE IS A GRAY MASK IN THE IMAGE'S COLOR, its alpha channel opaque (it goes through an
                # emission): wired from the Alpha output as set_material_texture's alpha input does by default, every
                # pixel read 1 and a cut-out card drew as a solid quad (FOUND 2026-11-26)
                call["fromSocket"] = "Color"
            r = op_set_material_texture(call)
            wired[which] = {"map": t, "file": fp, "image": r.get("image"), "colorSpace": r.get("colorSpace")}
        orm_row = _wire_orm(mat, orm_file, plan) if orm_file else None
    except (MifOpError, RuntimeError):
        bpy.data.materials.remove(mat)
        raise
    out = {"name": mat.name, "wired": wired, "notWired": skipped or None, "assigned": None, "uvMap": plan.get("uvMap"),
           "orm": orm_row}
    if plan["assign"]:
        obj = bpy.data.objects.get(plan["object"])
        if obj is None:
            out["assigned"] = {"error": "the object '%s' is gone" % plan["object"]}
        else:
            out["assigned"] = _assign_baked(obj, mat, plan.get("single", True))
    return out


def _assign_baked(obj, mat, single):
    """Put the baked material on obj: in every slot, then (single) down to ONE slot with every face on it.

    ONE SLOT BY DEFAULT (1.4, found 2026-11-19): a piece joined from parts with 6 materials came back with 6 slots all
    holding the baked material, and a game export needs one - it took assign_material_to_faces and clean_mesh
    removeUnusedSlots by hand. Every slot holds the same material either way, so the render is the same; what goes is
    the faces' old slot indices, which point at materials no longer on the object. The slot count is read back."""
    before = len(obj.material_slots)
    if not obj.material_slots:
        obj.data.materials.append(mat)
    for slot in obj.material_slots:
        slot.material = mat
    note = None
    me = obj.data
    if single and len(obj.material_slots) > 1:
        if getattr(me, "users", 1) > 1:
            note = ("the mesh '%s' is used by %d objects, so its %d slots were kept (each holds the baked material): "
                    "removing them would change the other objects too. Give this object its own copy of the mesh first "
                    "for one slot."
                    % (me.name, me.users, len(obj.material_slots)))
        else:
            polys = getattr(me, "polygons", None)
            if polys is not None and len(polys):
                polys.foreach_set("material_index", [0] * len(polys))
            while len(me.materials) > 1:
                me.materials.pop(index=len(me.materials) - 1)
            me.update()
            # an OBJECT-linked slot holds its own material, so the one left is written again
            for slot in obj.material_slots:
                slot.material = mat
    slots = len(obj.material_slots)
    return {"object": obj.name, "slots": slots, "slotsBefore": before,
            "singleSlot": slots == 1, "singleSlotNote": note}


# --------------------------------------------------------------------------------------------- roblox_pbr_bake
# FOUND ON A GAME: a kit was moved off a palette atlas by a hand-written Cycles library - a
# fresh unwrap and pack, texel density measured, five passes rewired through an Emission shader, JPG saves and a swap to
# one baked material before export. Every step was already a MifBlender op (uv_unwrap, bake_texture, the baked
# material bake_texture_set makes); this is them in one call, with what Roblox's SurfaceAppearance reads: ColorMap
# (sRGB), NormalMap (tangent, OpenGL only: "Roblox only supports OpenGL format - Tangent Space normal maps", the
# SurfaceAppearance page), RoughnessMap, MetalnessMap, and EmissiveMaskContent - a grayscale mask times EmissiveTint.
# The density is texture size x sqrt(UV area / surface area in studs); 1 Blender metre is 1 stud under export_mesh's
# profile roblox (measured in Studio, ops_mesh). A 2048 set over a 285-stud wall gave about 3 texels per stud on the
# game, props 15 to 60: fine features under about 2 texels do not survive.
_RPB_KEYS = ("object", "size", "folder", "pattern", "uvLayer", "unwrap", "angleLimitDeg", "packMargin",
             "samples", "margin", "studsPerUnit", "material", "emissive")
_RPB_PASSES = (("BASE_COLOR", "BC", "ColorMap"), ("NORMAL", "N", "NormalMap"), ("ROUGHNESS", "R", "RoughnessMap"),
               ("METALLIC", "M", "MetalnessMap"))


def _texel_density(obj, layer, size, studs_per_unit):
    """texels per stud over the mesh (size x sqrt(UV area / world area in studs)), and per face: 5th, 50th, 95th
    percentile; faces with no UV area counted."""
    import bmesh                                                # noqa: PLC0415
    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        bm.transform(obj.matrix_world)
        uvl = bm.loops.layers.uv.get(layer)
        if uvl is None:
            raise MifOpError("the mesh has no UV layer '%s' to measure. NOTHING was baked." % layer)
        uv_tot, area_tot, per, flat = 0.0, 0.0, [], 0
        for f in bm.faces:
            a = f.calc_area() * studs_per_unit ** 2
            uv = [l_[uvl].uv for l_ in f.loops]
            ua = 0.0
            for i in range(1, len(uv) - 1):            # a fan of triangles; UV faces are convex enough after an unwrap
                ua += abs((uv[i].x - uv[0].x) * (uv[i + 1].y - uv[0].y) - (uv[i + 1].x - uv[0].x) * (uv[i].y - uv[0].y)) / 2
            uv_tot += ua
            area_tot += a
            if a > 1e-12:
                if ua <= 1e-12:
                    flat += 1
                else:
                    per.append(size * math.sqrt(ua / a))
    finally:
        bm.free()
    if area_tot <= 1e-12:
        raise MifOpError("'%s' has no surface area to bake onto. NOTHING was baked." % obj.name)
    per.sort()

    def pct(q):
        return round(per[min(len(per) - 1, int(q * len(per)))], 3) if per else None
    return {"texelsPerStud": round(size * math.sqrt(uv_tot / area_tot), 3), "perFace": {"p5": pct(0.05), "p50": pct(0.5),
            "p95": pct(0.95)}, "uvAreaUsed": round(uv_tot, 4), "surfaceStuds2": round(area_tot, 3),
            "facesWithNoUvArea": flat, "studsPerUnit": studs_per_unit,
            "smallestFeatureStuds": round(2.0 / (size * math.sqrt(uv_tot / area_tot)), 4) if uv_tot > 0 else None}


def _emissive_mask(src_path, mask_path, tail):
    """The baked emission (colour x strength) as a grayscale mask (max of R, G, B, scaled to peak 1) and its tint (the
    mean colour where the mask is over half, scaled to peak 1). -> dict, or None when nothing emits."""
    import numpy as np                                          # noqa: PLC0415
    from .ops_common import abs_output_path                     # noqa: PLC0415
    mask_path = abs_output_path(mask_path)
    im = bpy.data.images.load(src_path, check_existing=False)
    try:
        w, h = im.size
        px = np.empty(w * h * 4, dtype=np.float32)
        im.pixels.foreach_get(px)
        px = px.reshape(-1, 4)
        m = px[:, :3].max(axis=1)
        peak = float(m.max()) if m.size else 0.0
        if peak <= 1.0 / 255:
            return None
        lit = m / peak > 0.5
        tint = px[lit, :3].mean(axis=0)
        tint = (tint / max(float(tint.max()), 1e-6)).tolist()
        out = np.empty((m.size, 4), dtype=np.float32)
        out[:, 0] = out[:, 1] = out[:, 2] = m / peak
        out[:, 3] = 1.0
        mk = bpy.data.images.new("MifRbxEmissiveMask", w, h, alpha=False)
        try:
            mk.colorspace_settings.name = "Non-Color"
            mk.pixels.foreach_set(out.ravel())
            mk.filepath_raw = mask_path
            mk.file_format = "JPEG"
            mk.save()
        finally:
            bpy.data.images.remove(mk)
    finally:
        bpy.data.images.remove(im)
    if not os.path.isfile(mask_path):
        raise MifOpError("the emissive mask was not written to %s. %s" % (mask_path, tail))
    return {"file": mask_path, "bytes": os.path.getsize(mask_path), "tint": [round(c, 4) for c in tint],
            "peak": round(peak, 4), "coverage": round(float(lit.mean()), 4)}


def op_roblox_pbr_bake(params):
    """Bake a mesh its own Roblox PBR set in one call: a fresh unwrap packed in 0-1, texels per stud measured, then
    color (sRGB), normal (tangent, OpenGL), roughness, metalness and an emissive mask baked from its current
    materials as JPG, and the mesh left with ONE material that reads them.

    params:
      object (str)            the mesh; its current materials (procedural or image) are what is baked
      size (int)              the texture size, a power of two 64-4096 (default 1024; an upload keeps 4096, RB1300)
      folder (str)            where the JPGs go; pattern (str) {object} and {suffix} (default {object}_{suffix}):
                              BC, N, R, M and E (the mask)
      uvLayer (str)           the layer the bake writes and the new material reads (default RobloxPBR); made, or
                              replaced when it exists
      unwrap (bool)           default true: uv_unwrap SMART (angleLimitDeg, default Blender's) packed into 0-1
                              (packMargin, default 0.01); false bakes onto uvLayer as it is
      samples, margin         as bake_texture (default 16 samples, margin 16 px)
      studsPerUnit (float)    default 1 (export_mesh profile roblox: 1 Blender metre = 1 stud)
      material (str)          the new material's name (default {object}_RobloxPBR); refused when it exists
      emissive (bool)         default true: bake EMIT and write the mask when anything emits
    Reply: texel (texelsPerStud overall, per face p5/p50/p95, smallestFeatureStuds at 2 texels, facesWithNoUvArea),
    maps (file, bytes, Roblox property each), emissive (the mask, tint and peak, or null when nothing emits), material
    (its name and what it reads), previousMaterials (kept in the file, no longer on the mesh), uvLayers.
    """
    reject_unknown(params, _RPB_KEYS, "roblox_pbr_bake")
    tail = "NOTHING was baked."
    from .ops_material import op_bake_texture, op_set_material_texture   # noqa: PLC0415
    from .ops_mesh import op_uv_unwrap                                   # noqa: PLC0415
    obj = get_object(take(params, "object", required=True, kind=str), want_mesh=True)
    size = take_int(params, "size", default=1024)
    if size not in (64, 128, 256, 512, 1024, 2048, 4096):
        raise MifOpError("size is a power of two from 64 to 4096 (an upload keeps up to 4096, RB1300), got %s. %s"
                         % (size, tail))
    folder = _folder(params, tail)
    pattern = take(params, "pattern", default="{object}_{suffix}", kind=str)
    layer = take(params, "uvLayer", default="RobloxPBR", kind=str)
    unwrap = take_bool(params, "unwrap", default=True)
    angle = take_float(params, "angleLimitDeg", default=None)
    pack_margin = take_float(params, "packMargin", default=0.01)
    samples = take_int(params, "samples", default=16)
    margin = take_int(params, "margin", default=16)
    spu = take_float(params, "studsPerUnit", default=1.0)
    if not spu > 0:
        raise MifOpError("studsPerUnit must be above 0, got %s. %s" % (spu, tail))
    mat_name = take(params, "material", default="%s_RobloxPBR" % obj.name, kind=str)
    emissive = take_bool(params, "emissive", default=True)
    if not unwrap and (angle is not None or params.get("packMargin") is not None):
        raise MifOpError("angleLimitDeg and packMargin shape the unwrap; with unwrap false there is none. %s" % tail)
    if bpy.data.materials.get(mat_name) is not None:
        raise MifOpError("a material named '%s' already exists - pass material with a new name. %s" % (mat_name, tail))
    prev = [s.material.name for s in obj.material_slots if s.material is not None]
    if not prev:
        raise MifOpError("'%s' has no material: there is nothing to bake from. %s" % (obj.name, tail))
    if not unwrap and obj.data.uv_layers.get(layer) is None:
        raise MifOpError("unwrap is false and '%s' has no UV layer '%s' to bake onto. %s" % (obj.name, layer, tail))
    files = {}
    for t, suffix, _prop in _RPB_PASSES + (("EMIT", "E", None),):
        fp = os.path.join(folder, pattern.replace("{object}", obj.name).replace("{suffix}", suffix) + ".jpg")
        check_output_path(fp, fp, "baked")
        files[t] = fp
    t0 = time.time()
    mesh = obj.data
    render_before = next((u.name for u in mesh.uv_layers if u.active_render), None)
    out = {"object": obj.name, "size": size, "uvLayer": layer, "previousMaterials": prev}
    if unwrap:
        uw = {"object": obj.name, "method": "SMART", "uvLayer": layer, "replace": True, "uvPack": True,
              "packMargin": pack_margin}
        if angle is not None:
            uw["angleLimitDeg"] = angle
        r = op_uv_unwrap(uw)
        out["unwrap"] = {k: r.get(k) for k in ("method", "createdLayer", "packed", "packMargin", "packedBounds", "layout")}
        # the source materials keep reading the layer they read: images with no UV Map node follow the render-active one
        for u in mesh.uv_layers:
            u.active_render = u.name == (render_before or layer)
    out["texel"] = _texel_density(obj, layer, size, spu)
    maps, times = {}, {}
    try:
        for t, suffix, prop in _RPB_PASSES + ((("EMIT", "E", None),) if emissive else ()):
            fp = files[t]
            if t == "EMIT":
                fp = os.path.join(tempfile.mkdtemp(prefix="mif_rbx_emit_"), "emit.png")
            p = {"object": obj.name, "type": t, "width": size, "height": size, "uvLayer": layer, "samples": samples,
                 "margin": margin, "filepath": fp, "imageName": "MifRbx_%s_%s" % (obj.name, suffix)}
            if t == "NORMAL":
                p["greenChannel"] = "OPENGL"
            s0 = time.time()
            r = op_bake_texture(p)
            times[t] = round(time.time() - s0, 2)
            im = bpy.data.images.get(r.get("image") or "")
            if t == "EMIT":
                if im is not None:
                    bpy.data.images.remove(im)
                em = _emissive_mask(fp, files["EMIT"], tail)
                shutil.rmtree(os.path.dirname(fp), ignore_errors=True)
                out["emissive"] = em if em is None else dict(em, property="EmissiveMaskContent",
                                                             note="EmissiveTint is tint; EmissiveStrength sets the glow "
                                                                  "(peak is the baked emission's brightest value)")
                continue
            maps[suffix] = {"file": r.get("savedTo"), "bytes": os.path.getsize(r["savedTo"]), "property": prop,
                            "colorSpace": "sRGB" if t == "BASE_COLOR" else "Non-Color"}
            if im is not None:
                im.name = os.path.basename(files[t])
    except MifOpError as exc:
        raise MifOpError("%s The UV layer '%s' is %s; the materials are unchanged." % (
            exc, layer, "new" if unwrap else "as it was"))
    if not emissive:
        out["emissive"] = None
    # ONE MATERIAL THAT READS THEM: bake_texture_set's baked material, then the emissive mask x tint
    plan = {"name": mat_name, "object": obj.name, "assign": False, "green": "OPENGL", "uvMap": layer}
    made = _wire_baked_material(plan, {t: files[t] for t, _s, _p in _RPB_PASSES})
    mat = bpy.data.materials.get(made["name"])
    if out.get("emissive"):
        op_set_material_texture({"material": mat.name, "file": files["EMIT"], "input": "emissive",
                                 "nodeName": "Baked EMIT mask", "uvMap": layer, "colorspace": "Non-Color"})
        nt = mat.node_tree
        p = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
        sock = p.inputs.get("Emission Color") or p.inputs.get("Emission")
        if sock is not None and sock.is_linked:
            src = sock.links[0].from_socket
            vm = nt.nodes.new("ShaderNodeVectorMath")
            vm.operation = "MULTIPLY"
            vm.name = vm.label = "EmissiveTint"
            vm.inputs[1].default_value = out["emissive"]["tint"]
            nt.links.new(src, vm.inputs[0])
            nt.links.new(vm.outputs[0], sock)
        if p.inputs.get("Emission Strength") is not None:
            p.inputs["Emission Strength"].default_value = 1.0
        made["wired"]["emissive"] = {"map": "EMIT mask x tint", "file": files["EMIT"]}
    mesh.materials.clear()
    mesh.materials.append(mat)
    for poly in mesh.polygons:
        poly.material_index = 0
    for u in mesh.uv_layers:
        u.active_render = u.name == layer
    mesh.uv_layers.active = mesh.uv_layers[layer]
    out.update(maps=maps, material={"name": mat.name, "reads": made["wired"]}, seconds=round(time.time() - t0, 2),
               passSeconds=times, uvLayers=[u.name for u in mesh.uv_layers],
               note="the mesh now has one material and its render UV layer is %s; the materials it had are kept in the "
                    "file. Roblox takes one UV set: bl_roblox_preflight refuses a mesh with more, so remove the others "
                    "before export if they are no longer needed" % layer)
    if len(mesh.uv_layers) > 1:
        out["otherUvLayers"] = [u.name for u in mesh.uv_layers if u.name != layer]
    return out


# --------------------------------------------------------------------------------------------- bake_modifier_texture
# FOUND 2026-11-18 (a creature build): its scales came from a legacy VORONOI texture on a DISPLACE modifier, and the
# shader's own Voronoi had a different cell field, so colour could not follow the cells; the build lightened the scale
# edges from the baked curvature instead. This samples the modifier's Texture datablock at the coordinates the modifier
# itself uses (MOD_get_texture_coords: LOCAL, GLOBAL, OBJECT, UV as uv * 2 - 1) into a color attribute or a UV image, so
# shading can follow the displaced geometry.
_BMT_KEYS = ("object", "modifier", "target", "name", "channel", "width", "height", "uvLayer", "filepath")


def _bmt_input_coords(obj, mod, tail):
    """The vertex positions as they reach `mod` (the modifiers before it applied, it and the rest switched off), local."""
    import numpy as np                                          # noqa: PLC0415
    mods = list(obj.modifiers)
    k = mods.index(mod)
    was = [(m, m.show_viewport) for m in mods[k:]]
    try:
        for m, _v in was:
            m.show_viewport = False
        dg = bpy.context.evaluated_depsgraph_get()
        dg.update()
        ev = obj.evaluated_get(dg)
        me = ev.to_mesh()
        try:
            n = len(me.vertices)
            if n != len(obj.data.vertices):
                raise MifOpError("the modifiers before '%s' change the vertex count (%d to %d), so its input cannot be "
                                 "written onto this mesh - apply them first (apply_modifier). %s"
                                 % (mod.name, len(obj.data.vertices), n, tail))
            co = np.empty(n * 3, dtype=np.float64)
            me.vertices.foreach_get("co", co)
        finally:
            ev.to_mesh_clear()
    finally:
        for m, v in was:
            m.show_viewport = v
        bpy.context.evaluated_depsgraph_get().update()
    return co.reshape(-1, 3)


def _bmt_texco(obj, mod, local, uv_per_corner):
    """The modifier's texture coordinates for local positions (n, 3), or per corner from the UV layer."""
    import numpy as np                                          # noqa: PLC0415
    from mathutils import Matrix                                # noqa: PLC0415
    kind = mod.texture_coords
    if kind == "UV":
        return uv_per_corner * 2.0 - 1.0 if uv_per_corner.shape[1] == 3 else np.c_[uv_per_corner * 2.0 - 1.0,
                                                                                      np.zeros(len(uv_per_corner))]
    mw = np.array(obj.matrix_world, dtype=np.float64)
    pts = np.c_[local, np.ones(len(local))]
    if kind == "LOCAL":
        return local
    world = pts @ mw.T
    if kind == "GLOBAL":
        return world[:, :3]
    ob = getattr(mod, "texture_coords_object", None)
    inv = np.array(ob.matrix_world.inverted() if ob is not None else Matrix.Identity(4), dtype=np.float64)
    return (world @ inv.T)[:, :3]


def op_bake_modifier_texture(params):
    """Sample a modifier's legacy Texture (Displace, Wave, Warp, the Vertex Weight ones) at the coordinates the modifier
    uses, into a color attribute (target COLOR_ATTRIBUTE, the default) or a UV image (target IMAGE), so a material can
    follow the cells or noise that shaped the geometry.

    object, modifier             the mesh and its modifier with a texture
    channel                      INTENSITY (default: what Displace moves by) or RGB
    name                         the attribute or image to write (default "<modifier>_Tex"); made when missing
    COLOR_ATTRIBUTE              a FLOAT_COLOR attribute, POINT domain (CORNER for UV coordinates), set render-active
    IMAGE                        width, height (default 512), uvLayer (default the active one), filepath to save; each
                                 texel takes the input position its UV triangle interpolates
    The modifier's input is the mesh as it reaches it - the modifiers before it applied - and must keep the base mesh's
    vertex count. The reply's readBack: the value range written, and for Displace on NORMAL the correlation between the
    written intensity and each vertex's displacement along its normal (1.0: the colour follows the geometry)."""
    import numpy as np                                          # noqa: PLC0415
    reject_unknown(params, _BMT_KEYS, "bake_modifier_texture")
    tail = "NOTHING was written."
    obj = get_object(take(params, "object", required=True, kind=str), want_mesh=True)
    mname = take(params, "modifier", required=True, kind=str)
    mod = obj.modifiers.get(mname)
    if mod is None:
        raise MifOpError("'%s' has no modifier named '%s' (it has: %s). %s"
                         % (obj.name, mname, ", ".join(m.name for m in obj.modifiers) or "none", tail))
    if not hasattr(mod, "texture") or not hasattr(mod, "texture_coords"):
        raise MifOpError("modifier '%s' (%s) takes no texture - Displace, Wave, Warp and the Vertex Weight modifiers "
                         "do. %s" % (mod.name, mod.type, tail))
    tex = mod.texture
    if tex is None:
        raise MifOpError("modifier '%s' has no texture to sample. %s" % (mod.name, tail))
    if not hasattr(tex, "evaluate"):
        raise MifOpError("this Blender's Texture has no evaluate(), so the texture cannot be sampled here. %s" % tail)
    target = str(take(params, "target", default="COLOR_ATTRIBUTE", kind=str)).upper()
    if target not in ("COLOR_ATTRIBUTE", "IMAGE"):
        raise MifOpError("target is COLOR_ATTRIBUTE (default) or IMAGE, got %r. %s" % (params.get("target"), tail))
    channel = str(take(params, "channel", default="INTENSITY", kind=str)).upper()
    if channel not in ("INTENSITY", "RGB"):
        raise MifOpError("channel is INTENSITY (default) or RGB, got %r. %s" % (params.get("channel"), tail))
    name = take(params, "name", default="%s_Tex" % mod.name, kind=str)
    me = obj.data
    uv_name = take(params, "uvLayer", default=None, kind=str)
    need_uv = target == "IMAGE" or mod.texture_coords == "UV"
    if need_uv:
        lay = uv_name or (getattr(mod, "uv_layer", "") if mod.texture_coords == "UV" else "") or (
            me.uv_layers.active.name if me.uv_layers.active else None)
        if lay is None or me.uv_layers.get(lay) is None:
            raise MifOpError("'%s' has no UV layer %s - %s needs one. %s"
                             % (obj.name, repr(lay) if lay else "", "an IMAGE target" if target == "IMAGE"
                                else "UV texture coordinates", tail))
    else:
        lay = None
    if target == "IMAGE":
        width, height = take_int(params, "width", default=512), take_int(params, "height", default=512)
        if not (1 <= width <= 4096 and 1 <= height <= 4096):
            raise MifOpError("width and height are 1 to 4096, got %dx%d. %s" % (width, height, tail))
        fp = take(params, "filepath", default=None, kind=str)
        if fp:
            check_output_path(fp, bpy.path.abspath(fp), "baked")
    else:
        clash = [k for k in ("width", "height", "filepath") if params.get(k) is not None]
        if clash:
            raise MifOpError("%s shape an IMAGE target; this is COLOR_ATTRIBUTE. %s" % (clash, tail))
    local = _bmt_input_coords(obj, mod, tail)

    def sample(texco):
        out = np.empty((len(texco), 4), dtype=np.float32)
        ev = tex.evaluate
        for i, c in enumerate(texco):
            out[i] = ev((float(c[0]), float(c[1]), float(c[2])))
        return out

    nl = len(me.loops)
    lv = np.empty(nl, dtype=np.int32)
    me.loops.foreach_get("vertex_index", lv)
    uvc = None
    if lay is not None:
        uvd = np.empty(nl * 2, dtype=np.float64)
        me.uv_layers[lay].data.foreach_get("uv", uvd)
        uvc = uvd.reshape(-1, 2)
    t0 = time.perf_counter()
    if target == "COLOR_ATTRIBUTE":
        corner = mod.texture_coords == "UV"
        vals = sample(_bmt_texco(obj, mod, local, uvc) if corner else _bmt_texco(obj, mod, local, None))
        rgba = vals if channel == "RGB" else np.repeat(vals[:, 3:4], 4, axis=1)
        rgba[:, 3] = 1.0
        ca = me.color_attributes.get(name)
        dom = "CORNER" if corner else "POINT"
        if ca is not None and (ca.domain != dom or ca.data_type != "FLOAT_COLOR"):
            me.color_attributes.remove(ca)
            ca = None
        if ca is None:
            ca = me.color_attributes.new(name, "FLOAT_COLOR", dom)
        ca.data.foreach_set("color", np.ascontiguousarray(rgba, dtype=np.float32).ravel())
        me.color_attributes.active_color = ca
        if hasattr(me.color_attributes, "render_color_index"):
            me.color_attributes.render_color_index = list(me.color_attributes).index(ca)
        me.update()
        written = {"attribute": ca.name, "domain": dom, "count": len(rgba)}
        inten = vals[:, 3]
    else:
        me.calc_loop_triangles()
        nt = len(me.loop_triangles)
        tl = np.empty(nt * 3, dtype=np.int32)
        me.loop_triangles.foreach_get("loops", tl)
        tl = tl.reshape(-1, 3)
        tri_uv = uvc[tl] * np.array([width, height])
        pos = np.zeros((height, width, 3), dtype=np.float64)
        cov = np.zeros((height, width), dtype=bool)
        corner_pos = local[lv]
        for t in range(nt):
            a, b, c = tri_uv[t]
            x0, x1 = int(max(0, np.floor(min(a[0], b[0], c[0])))), int(min(width - 1, np.ceil(max(a[0], b[0], c[0]))))
            y0, y1 = int(max(0, np.floor(min(a[1], b[1], c[1])))), int(min(height - 1, np.ceil(max(a[1], b[1], c[1]))))
            if x1 < x0 or y1 < y0:
                continue
            den = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
            if abs(den) < 1e-12:
                continue
            xs = np.arange(x0, x1 + 1) + 0.5
            ys = (np.arange(y0, y1 + 1) + 0.5)[:, None]
            l1 = ((b[1] - c[1]) * (xs - c[0]) + (c[0] - b[0]) * (ys - c[1])) / den
            l2 = ((c[1] - a[1]) * (xs - c[0]) + (a[0] - c[0]) * (ys - c[1])) / den
            l3 = 1.0 - l1 - l2
            inside = (l1 >= -1e-6) & (l2 >= -1e-6) & (l3 >= -1e-6)
            if not inside.any():
                continue
            p_ = (l1[..., None] * corner_pos[tl[t, 0]] + l2[..., None] * corner_pos[tl[t, 1]]
                  + l3[..., None] * corner_pos[tl[t, 2]])
            sub, sc = pos[y0:y1 + 1, x0:x1 + 1], cov[y0:y1 + 1, x0:x1 + 1]
            sub[inside] = p_[inside]
            sc[inside] = True
        flat = pos[cov]
        if mod.texture_coords == "UV":
            ys_, xs_ = np.nonzero(cov)
            texco = np.c_[(xs_ + 0.5) / width * 2.0 - 1.0, (ys_ + 0.5) / height * 2.0 - 1.0, np.zeros(len(xs_))]
        else:
            texco = _bmt_texco(obj, mod, flat, None)
        vals = sample(texco)
        img = np.zeros((height, width, 4), dtype=np.float32)
        img[..., 3] = 1.0
        img[cov] = vals if channel == "RGB" else np.repeat(vals[:, 3:4], 4, axis=1)
        img[cov, 3] = 1.0
        im = bpy.data.images.get(name)
        if im is None or tuple(im.size) != (width, height):
            im = bpy.data.images.new(name, width, height, float_buffer=True)
        im.colorspace_settings.name = "Non-Color" if channel == "INTENSITY" else im.colorspace_settings.name
        im.pixels.foreach_set(img.ravel())
        im.update()
        saved = None
        if fp:
            from .ops_common import abs_output_path              # noqa: PLC0415
            im.filepath_raw = abs_output_path(fp)
            im.file_format = "OPEN_EXR" if fp.lower().endswith(".exr") else "PNG"
            im.save()
            saved = im.filepath_raw
        written = {"image": im.name, "size": [width, height], "texels": int(cov.sum()), "savedTo": saved,
                   "uvLayer": lay}
        inten = vals[:, 3]
    out = {"object": obj.name, "modifier": mod.name, "modifierType": mod.type, "texture": tex.name,
           "textureType": tex.type, "textureCoords": mod.texture_coords, "channel": channel, "target": target,
           "written": written, "seconds": round(time.perf_counter() - t0, 3),
           "readBack": {"intensityMin": round(float(inten.min()), 6) if len(inten) else None,
                        "intensityMax": round(float(inten.max()), 6) if len(inten) else None,
                        "intensityMean": round(float(inten.mean()), 6) if len(inten) else None}}
    if mod.type == "DISPLACE" and mod.direction == "NORMAL" and target == "COLOR_ATTRIBUTE" and \
            mod.texture_coords != "UV":
        # THE READ-BACK: the displacement the modifier made against the intensity written, per vertex
        dg = bpy.context.evaluated_depsgraph_get()
        ev = obj.evaluated_get(dg)
        em = ev.to_mesh()
        try:
            if len(em.vertices) == len(me.vertices):
                after = np.empty(len(em.vertices) * 3)
                em.vertices.foreach_get("co", after)
                nrm = np.empty(len(me.vertices) * 3)
                me.vertices.foreach_get("normal", nrm)
                disp = ((after.reshape(-1, 3) - local) * nrm.reshape(-1, 3)).sum(axis=1)
                if disp.std() > 1e-9 and inten.std() > 1e-9:
                    out["readBack"]["displacementCorrelation"] = round(float(np.corrcoef(disp, inten)[0, 1]), 6)
        finally:
            ev.to_mesh_clear()
    return out


OPS = {
    "bake_modifier_texture": op_bake_modifier_texture,
    "roblox_pbr_bake": op_roblox_pbr_bake,
    "bake_vertex_animation": op_bake_vertex_animation,
    "bake_pivot_painter": op_bake_pivot_painter,
    "hair_to_cards": op_hair_to_cards,
    "bake_texture_set": op_bake_texture_set,
}
