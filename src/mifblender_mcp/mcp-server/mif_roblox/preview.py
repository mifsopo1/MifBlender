"""One Blender object to a Studio mesh preview in one call (mif_preview_in_roblox).

Blender side: MifBlender's read_mesh_data (WORLD space, UVs per face corner) - no run_python. This side: each polygon fanned
into triangles, positions to Roblox axes (Blender X, Y, Z with Z up is Roblox X, -Z, Y... written as (x, z, -y)), V flipped, a flat normal per triangle, then rbx_mesh_preview, and the counts and
span compared. Blender metres are studs (RB45).

NEVER RAISE.
"""
import json
import os
import tempfile


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def arrays_from_mesh_file(data, uv_layer=None):
    """(vertices, triangles, corner_uvs, corner_normals) in Roblox axes from a mifbridge.mesh/1 file's dict."""
    verts = []
    for x, y, z in data["vertices"]:
        verts += [x, z, -y]
    uvs_all = data.get("uvs") or {}
    layer = uvs_all.get(uv_layer) if uv_layer else (next(iter(uvs_all.values())) if uvs_all else None)
    tris, cuv, cn = [], [], []
    for fi, face in enumerate(data["faces"]):
        fuv = layer[fi] if layer else None
        for k in range(1, len(face) - 1):
            corners = (0, k, k + 1)
            ids = [face[c] for c in corners]
            tris += ids
            if fuv:
                for c in corners:
                    u, v = fuv[c]
                    cuv += [u, 1.0 - v]
            p = [verts[i * 3:i * 3 + 3] for i in ids]
            n = _cross([p[1][j] - p[0][j] for j in range(3)], [p[2][j] - p[0][j] for j in range(3)])
            ln = (n[0] ** 2 + n[1] ** 2 + n[2] ** 2) ** 0.5 or 1.0
            cn += [n[0] / ln, n[1] / ln, n[2] / ln] * 3
    return verts, tris, (cuv if layer else None), cn


def preview(blender_call, rbx_mesh_preview, obj, name=None, parent="Workspace.MifScratch", position=None, uv_layer=None):
    try:
        path = os.path.join(tempfile.gettempdir(), "mif_preview_%d.json" % os.getpid())
        r = blender_call("read_mesh_data", object=obj, file=path, space="WORLD", uvs=True, looseEdges=False)
        if not r.get("ok"):
            return {"ok": False, "step": "read_mesh_data", "error": r.get("error")}
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        os.remove(path)
        verts, tris, cuv, cn = arrays_from_mesh_file(data, uv_layer)
        b = r.get("bounds") or {}
        span = None
        if b.get("min") and b.get("max"):
            mn, mx = b["min"], b["max"]
            span = [mx[0] - mn[0], mx[2] - mn[2], mx[1] - mn[1]]          # Roblox X, Y, Z
        res = rbx_mesh_preview(verts, tris, name=name or obj, parent=parent, position=position, corner_uvs=cuv,
                               corner_normals=cn)
        out = {"ok": bool(res.get("ok")), "object": obj, "blender": {"vertices": data.get("vertexCount"),
               "polygons": data.get("faceCount"), "triangles": len(tris) // 3, "spanStuds": span, "uvs": cuv is not None},
               "roblox": {k: res.get(k) for k in ("path", "faces", "vertices", "meshSize", "error")}}
        problems = []
        if res.get("faces") != len(tris) // 3:
            problems.append("Studio holds %s faces, Blender's polygons fan into %d triangles" % (res.get("faces"), len(tris) // 3))
        ms = res.get("meshSize")
        if span and isinstance(ms, list) and not all(abs(a - b2) <= max(1e-3, 1e-3 * b2) for a, b2 in zip(ms, span)):
            problems.append("MeshSize %s against Blender's span %s" % (ms, span))
        if problems or not res.get("ok"):
            out.update(ok=False, error="; ".join(problems) or res.get("error"))
        out["verified"] = out["ok"]
        return out
    except Exception as exc:  # noqa: BLE001 - never raise
        return {"ok": False, "error": "the preview failed: %r" % (exc,)}
