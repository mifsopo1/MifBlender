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
             "bakedMaterial", "assignBaked")
_BTS_SUFFIX = {"AO": "AO", "NORMAL": "N", "DIFFUSE": "BC", "BASE_COLOR": "BC", "ROUGHNESS": "R", "METALLIC": "M",
               "EMIT": "E",
               "CURVATURE": "Curv", "ALPHA": "A", "MATERIAL_ID": "ID", "HEIGHT": "H", "POSITION": "Pos",
               "THICKNESS": "Thick", "COMBINED": "C", "GLOSSY": "G", "SHADOW": "S"}
_JOBS = {}
_JOB_SCRIPT = r'''
import json, sys, traceback
sys.path.insert(0, @@ADDON@@)
import bpy
from MifBlender import ops_material, ops_image
spec = json.load(open(@@SPEC@@, encoding="utf-8"))
rows = []
for req in spec["bakes"]:
    try:
        r = ops_material.op_bake_texture(req["params"])
        rows.append({"type": req["type"], "ok": True, "file": r.get("savedTo"), "signature": r.get("savedSignature")})
    except Exception as exc:
        rows.append({"type": req["type"], "ok": False, "error": str(exc)[:600]})
    json.dump({"rows": rows, "done": False}, open(@@OUT@@, "w", encoding="utf-8"))
orm = None
if spec.get("orm"):
    try:
        r = ops_image.op_channel_pack_image(spec["orm"])
        orm = {"ok": True, "file": r.get("file"), "readBack": r.get("readBack")}
    except Exception as exc:
        orm = {"ok": False, "error": str(exc)[:600]}
json.dump({"rows": rows, "orm": orm, "done": True}, open(@@OUT@@, "w", encoding="utf-8"))
'''


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
    out = {"jobId": job_id, "running": code is None, "exitCode": code, "mapsExpected": len(job["types"]),
           "mapsDone": len(res.get("rows") or []), "maps": res.get("rows") or [], "orm": res.get("orm"),
           "elapsedSeconds": round(time.time() - job["started"], 1), "log": job["log"]}
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
                    job["bakedResult"] = _wire_baked_material(job["baked"], job["files"])
                except MifOpError as exc:
                    job["bakedResult"] = {"error": str(exc)}
            out["bakedMaterial"] = job["bakedResult"]
        if not out["complete"]:
            out["error"] = "the bake process ended (exit %s) with %d of %d maps done%s" % (
                code, sum(1 for r in res.get("rows") or [] if r.get("ok")), len(job["types"]),
                "" if res.get("done") else " - read the log")
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
                                   as bake_texture takes them
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
                                   finish. assignBaked (bool) also puts it in every material slot of the object. It reads the
                                   UV layer the bake wrote: uvMap in the reply names it when that is not the render-active one.
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
    baked = _baked_plan(params, obj, types, tail)
    folder = _folder(params, tail)
    pattern = take(params, "pattern", default="T_{object}_{suffix}", kind=str)
    shared = {k: params[k] for k in ("width", "height", "samples", "margin", "uvLayer", "source", "extrusion",
                                     "maxRayDistance", "cage") if params.get(k) is not None}
    bakes = []
    for t in types:
        ext = ".exr" if t in ("HEIGHT", "POSITION") else ".png"
        fp = os.path.join(folder, pattern.replace("{object}", obj.name).replace("{suffix}", _BTS_SUFFIX[t]) + ext)
        # BEFORE ANY BAKE: the pattern is the caller's, and a control character in it would otherwise surface in
        # the child, after the session copy and the launch
        check_output_path(fp, fp, "baked")
        p = dict(shared, object=obj.name, type=t, filepath=fp, imageName="MifSet_%s" % t)
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
    if not take_bool(params, "outOfProcess", default=True):
        from .ops_image import op_channel_pack_image           # noqa: PLC0415
        from .ops_material import op_bake_texture              # noqa: PLC0415
        rows = []
        for b in bakes:
            r = op_bake_texture(b["params"])
            rows.append({"type": b["type"], "ok": True, "file": r.get("savedTo"), "signature": r.get("savedSignature")})
            im = bpy.data.images.get(r["image"])
            if im is not None:
                im.name = os.path.basename(b["params"]["filepath"])
        orm_r = None
        if orm is not None:
            orm.pop("_load")
            orm_r = op_channel_pack_image(orm)
            orm_r = {"ok": True, "file": orm_r.get("file"), "readBack": orm_r.get("readBack")}
        out = {"object": obj.name, "outOfProcess": False, "maps": rows, "orm": orm_r, "complete": True}
        if baked is not None:
            out["bakedMaterial"] = _wire_baked_material(baked, {r["type"]: r["file"] for r in rows})
        return out
    # OUT OF PROCESS: a copy of THIS session's state (not the file on disk) is what the child bakes
    stamp = "%s_%d" % (time.strftime("%Y%m%d_%H%M%S"), os.getpid())
    work = os.path.join(tempfile.gettempdir(), "MifBridge", "bake_set", stamp)
    os.makedirs(work, exist_ok=True)
    blend = os.path.join(work, "session.blend")
    try:
        bpy.ops.wm.save_as_mainfile(filepath=blend, copy=True)
    except RuntimeError as exc:
        raise MifOpError("could not write the session copy for the bake (%s). %s" % (exc, tail))
    snapped = time.time()
    # NEVER STARTED FOR A CALLER WHO HAS GONE (FOUND 2026-11-07). The dispatcher refuses a request that arrives after its
    # caller's deadline; this asks again once the copy is on disk, because writing a large session takes seconds and the
    # caller may have given up during it - and then, told "timed out", changed the scene the child would bake.
    late = caller_gave_up(snapped)
    if late is not None:
        shutil.rmtree(work, ignore_errors=True)
        raise MifOpError("the caller stopped waiting %.1f s ago, while the session copy was being written - no bake was "
                         "started, and the copy was deleted. %s" % (late, tail))
    if orm is not None:
        # the child loads the baked maps by path before packing
        load = orm.pop("_load")
        orm["channels"] = {c: ({"image": os.path.basename(load[c]), "channel": "R"} if c in load else v)
                           for c, v in orm["channels"].items()}
    spec = {"bakes": bakes, "orm": orm}
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
    sid, held = childproc.acquire("blender bake", obj.name)
    if sid is None:
        raise MifOpError(childproc.busy_message("bake_texture_set", held) + " " + tail)
    log_path = os.path.join(work, "bake.log")
    try:
        log_fh = open(log_path, "wb")
        popen = subprocess.Popen([exe, "-b", blend, "--factory-startup", "--python", script],
                                 **childproc.hidden(dict(stdout=log_fh, stderr=subprocess.STDOUT,
                                                         stdin=subprocess.DEVNULL, cwd=work)))
    except Exception as exc:                                    # noqa: BLE001
        childproc.release(sid)
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
            "note": "baking in a second Blender from a copy of this session; poll bake_texture_set {jobId}"}


# THE BAKED MATERIAL (1.4, 2026-10-30). A procedural look - set_material_properties' weathering, a noise in the
# roughness - reaches Unreal as nothing: export_material_textures carries images and lists node chains in notCarried.
# The bake makes the images; this makes the material that reads them, so the send carries the look.
_BAKED_INPUT = {"BASE_COLOR": "baseColor", "DIFFUSE": "baseColor", "ROUGHNESS": "roughness", "METALLIC": "metallic",
                "NORMAL": "normal", "EMIT": "emissive", "ALPHA": "alpha"}


def _baked_plan(params, obj, types, tail):
    name = take(params, "bakedMaterial", default=None, kind=str)
    assign = take_bool(params, "assignBaked", default=False)
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
    return {"name": name, "object": obj.name, "assign": assign, "green": green, "uvMap": baked_on if baked_on != render_on else None}


def _wire_baked_material(plan, files):
    """Create plan['name'] and wire each baked file into its Principled input; assign it when asked."""
    from .ops_material import op_create_material, op_set_material_texture   # noqa: PLC0415
    made = op_create_material({"name": plan["name"]})
    mat = bpy.data.materials.get(made["name"])
    wired, skipped = {}, {}
    try:
        for t, fp in files.items():
            which = _BAKED_INPUT.get(t)
            if which is None:
                skipped[t] = "no Principled input takes a %s map" % t
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
            r = op_set_material_texture(call)
            wired[which] = {"map": t, "file": fp, "image": r.get("image"), "colorSpace": r.get("colorSpace")}
    except MifOpError:
        bpy.data.materials.remove(mat)
        raise
    out = {"name": mat.name, "wired": wired, "notWired": skipped or None, "assigned": None, "uvMap": plan.get("uvMap")}
    if plan["assign"]:
        obj = bpy.data.objects.get(plan["object"])
        if obj is None:
            out["assigned"] = {"error": "the object '%s' is gone" % plan["object"]}
        else:
            if not obj.material_slots:
                obj.data.materials.append(mat)
            for slot in obj.material_slots:
                slot.material = mat
            out["assigned"] = {"object": obj.name, "slots": len(obj.material_slots)}
    return out


OPS = {
    "bake_vertex_animation": op_bake_vertex_animation,
    "bake_pivot_painter": op_bake_pivot_painter,
    "hair_to_cards": op_hair_to_cards,
    "bake_texture_set": op_bake_texture_set,
}
