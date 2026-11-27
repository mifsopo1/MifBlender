"""An FBX file's meshes, materials, UVs and skin read for Roblox, from the file alone (standard library only).

The kit and mesh-pipeline tools (rbx_mesh_preflight, rbx_asset_upload's mesh-data dedupe and skinned-size guard, the
batch upload, the kit checks) all read FBX files before anything reaches Studio; this is their one reader. It parses
binary FBX with mif_fbx_hulls' node reader (Blender's exporter writes binary); an ASCII file raises ValueError naming it.

WHAT STUDIO DOES WITH A FILE, measured, and used here:
  - Axes: Roblox (X, Y, Z) = file (-X, +Y, -Z) for a file whose GlobalSettings say up +Y, front +Z, coord +X - the
    settings both Blender's default export and MifBlender's profile roblox write. So
    the file's +Z (Blender's -Y with forward -Z) is Roblox's front, -Z. A file with other axis settings has no measured
    mapping: rbx_axes() answers None and the callers refuse rather than guess.
  - Scale: one stud per file unit after the nodes' own transforms, whatever UnitScaleFactor says. Blender's default
    export writes metres with node scale 100 and UnitScaleFactor 1 (100 studs per Blender metre, RB1472); profile roblox
    writes metres with no node scale and UnitScaleFactor 100 (1 stud per Blender metre, RB45, RB1156). Both files read
    here as the sizes Studio gave.
  - Size: a Model upload whose largest MeshPart axis would pass 2048 studs is shrunk as a whole to 2048 (RB1023), which
    on a skinned rig leaves the bones out of the meshes' space (RB1472).

load(path) -> Scene; Scene.summary() is JSON-safe. Raises OSError/ValueError on an unreadable file; nothing else raises.
"""
import hashlib
import math
import struct
import zlib

ROBLOX_MAX_STUDS = 2048.0           # RB1023: the whole-model shrink brings the largest part axis to exactly this
MEASURED_AXES = {"UpAxis": 1, "UpAxisSign": 1, "FrontAxis": 2, "FrontAxisSign": 1, "CoordAxis": 0, "CoordAxisSign": 1}


# ------------------------------------------------------------------------------------------------------- matrix math
def mat_mul(a, b):
    """4x4 row-major product a * b."""
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def mat_identity():
    return [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]


def mat_point(m, p):
    return [m[i][0] * p[0] + m[i][1] * p[1] + m[i][2] * p[2] + m[i][3] for i in range(3)]


def mat_det3(m):
    return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))


def _translate(t):
    m = mat_identity()
    m[0][3], m[1][3], m[2][3] = float(t[0]), float(t[1]), float(t[2])
    return m


def _scale(s):
    m = mat_identity()
    m[0][0], m[1][1], m[2][2] = float(s[0]), float(s[1]), float(s[2])
    return m


def _euler_xyz(deg):
    """FBX's default rotation order eEulerXYZ: X applied first, so R = Rz * Ry * Rx."""
    x, y, z = (math.radians(float(d)) for d in deg)
    cx, sx, cy, sy, cz, sz = math.cos(x), math.sin(x), math.cos(y), math.sin(y), math.cos(z), math.sin(z)
    rx = [[1, 0, 0, 0], [0, cx, -sx, 0], [0, sx, cx, 0], [0, 0, 0, 1]]
    ry = [[cy, 0, sy, 0], [0, 1, 0, 0], [-sy, 0, cy, 0], [0, 0, 0, 1]]
    rz = [[cz, -sz, 0, 0], [sz, cz, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    return mat_mul(rz, mat_mul(ry, rx))


def _from_fbx_columns(v):
    """An FBX 16-double matrix (column-major) as 4x4 row-major."""
    return [[float(v[c * 4 + r]) for c in range(4)] for r in range(4)]


def rbx_point(p):
    """A file-world point on Roblox axes (RB1156)."""
    return [-p[0], p[1], -p[2]]


# ------------------------------------------------------------------------------------------------------------ reading
def _p70(node):
    out = {}
    for c in node["children"]:
        if c["name"] == "Properties70":
            for p in c["children"]:
                if p["name"] == "P" and p["props"]:
                    out[p["props"][0]] = p["props"][4:]
    return out


def _child(node, name):
    return next((c for c in node["children"] if c["name"] == name), None)


def _child_value(node, name, default=None):
    c = _child(node, name)
    return c["props"][0] if c and c["props"] else default


class Scene:
    """What load() read: models, meshes, materials, skin, bones. Arrays stay here; summary() is the JSON answer."""

    def __init__(self):
        self.path = None
        self.unit_scale_cm = 1.0
        self.axis_settings = {}
        self.models = {}          # id -> {name, type, local (4x4), parent}
        self.meshes = []          # [{name, modelId, world, positions (file world), polygons [[vertex index]], ...}]
        self.textures = []        # [{name, sha256}] embedded images (Video Content)
        self.material_textures = {}   # material name -> [{property, texture, fileName, relativeFilename, embedded bytes}]
        self.bones = []           # [{name, world (file world 4x4 at bind), parent}]
        self.notes = []

    def rbx_axes(self):
        """True when this file's axis settings are the ones Studio's mapping was measured on (RB1156)."""
        return all(self.axis_settings.get(k) == v for k, v in MEASURED_AXES.items())

    def summary(self):
        meshes = []
        for m in self.meshes:
            row = {k: m[k] for k in ("name", "vertexCount", "polygons", "triangles", "quads", "ngons", "largestPolygon",
                                     "materials", "uvSets", "skinned", "bones", "boxMin", "boxMax", "sizeStuds",
                                     "signedVolume", "mirroredNode", "shells")}
            meshes.append(row)
        return {"file": self.path, "format": "binary", "unitScaleCm": self.unit_scale_cm,
                "axisSettings": self.axis_settings, "measuredAxes": self.rbx_axes(), "meshes": meshes,
                "bones": [{"name": b["name"], "parent": b["parent"], "position": [round(v, 6) for v in b["position"]]}
                          for b in self.bones],
                "embeddedTextures": len(self.textures), "notes": self.notes}


def _node_local(p70, notes, name):
    t = p70.get("Lcl Translation") or [0, 0, 0]
    r = p70.get("Lcl Rotation") or [0, 0, 0]
    s = p70.get("Lcl Scaling") or [1, 1, 1]
    pre = p70.get("PreRotation") or [0, 0, 0]
    post = p70.get("PostRotation") or [0, 0, 0]
    order = (p70.get("RotationOrder") or [0])[0]
    if order not in (0, None):
        notes.append("%s: rotation order %s is not XYZ; its turn is read as XYZ" % (name, order))
    for key in ("RotationPivot", "ScalingPivot", "RotationOffset", "ScalingOffset"):
        v = p70.get(key)
        if v and any(abs(float(x)) > 1e-9 for x in v[:3]):
            notes.append("%s: %s is set and not applied" % (name, key))
    m = mat_mul(_translate(t), mat_mul(_euler_xyz(pre), mat_mul(_euler_xyz(r), _scale(s))))
    if any(abs(float(x)) > 1e-9 for x in post[:3]):
        notes.append("%s: PostRotation is set and not applied" % name)
    gt, gr, gs = p70.get("GeometricTranslation"), p70.get("GeometricRotation"), p70.get("GeometricScaling")
    geo = mat_mul(_translate(gt or [0, 0, 0]), mat_mul(_euler_xyz(gr or [0, 0, 0]), _scale(gs or [1, 1, 1])))
    return m, geo


def _layer_values(geom, element, values_name, index_name, n_polyverts, poly_vert_to_vertex, poly_of_pv, width):
    """Per polygon-vertex values of a LayerElement (UV, normal), resolved through its mapping and reference, or None."""
    layers = [c for c in geom["children"] if c["name"] == element]
    out = []
    for layer in layers:
        vals = _child_value(layer, values_name)
        if not isinstance(vals, list):
            continue
        mapping = _child_value(layer, "MappingInformationType", "ByPolygonVertex")
        ref = _child_value(layer, "ReferenceInformationType", "Direct")
        idx = _child_value(layer, index_name) if ref in ("IndexToDirect", "Index") else None
        tuples = [vals[i:i + width] for i in range(0, len(vals) - width + 1, width)]
        res = []
        for pv in range(n_polyverts):
            if mapping == "ByPolygonVertex":
                k = pv
            elif mapping in ("ByVertex", "ByVertice", "ByControlPoint"):
                k = poly_vert_to_vertex[pv]
            elif mapping == "ByPolygon":
                k = poly_of_pv[pv]
            else:  # AllSame
                k = 0
            if idx is not None:
                k = idx[k] if 0 <= k < len(idx) else -1
            res.append(tuples[k] if 0 <= k < len(tuples) else None)
        out.append(res)
    return out


# What a cut or damaged file raises inside the node reader and the walks below: answered as ValueError, as promised.
DAMAGED = (struct.error, IndexError, KeyError, TypeError, AttributeError, OverflowError, RecursionError, zlib.error)


def load(path):
    """The Scene of a binary FBX. A cut or damaged file raises ValueError naming it, never struct.error or IndexError:
    every caller catches (OSError, ValueError) and answers "did not read"."""
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:18] != b"Kaydara FBX Binary":
        raise ValueError("%s is not a binary FBX (an ASCII FBX or another format); Blender's exporter writes binary" % path)
    try:
        return _load(path, data)
    except DAMAGED as exc:
        why = "%s is cut short or damaged (%d bytes): %s: %s" % (path, len(data), type(exc).__name__, exc)
        raise ValueError(why) from exc


def _load(path, data):
    from mif_fbx_hulls import _binary_top  # noqa: PLC0415 - the MCP server's own FBX node reader
    top = _binary_top(data)
    sc = Scene()
    sc.path = path
    gs = _p70(top["GlobalSettings"]) if "GlobalSettings" in top else {}
    sc.unit_scale_cm = float((gs.get("UnitScaleFactor") or [1.0])[0])
    sc.axis_settings = {k: int(gs[k][0]) for k in MEASURED_AXES if gs.get(k)}
    objects = (top.get("Objects") or {"children": []})["children"]
    geoms, materials, skins, clusters, videos, video_bytes, texnodes = {}, {}, {}, {}, {}, {}, {}
    for n in objects:
        if len(n["props"]) < 2:
            continue
        oid, name = n["props"][0], str(n["props"][1]).split("\x00\x01")[0]
        kind = n["props"][2] if len(n["props"]) > 2 else ""
        if n["name"] == "Model":
            local, geo = _node_local(_p70(n), sc.notes, name)
            sc.models[oid] = {"name": name, "type": kind, "local": local, "geo": geo, "parent": None}
        elif n["name"] == "Geometry" and kind == "Mesh":
            geoms[oid] = n
        elif n["name"] == "Material":
            materials[oid] = name
        elif n["name"] == "Deformer" and kind == "Skin":
            skins[oid] = name
        elif n["name"] == "Deformer" and kind == "Cluster":
            clusters[oid] = {"name": name, "link": _child_value(n, "TransformLink"),
                             "indexes": _child_value(n, "Indexes") or [], "weights": _child_value(n, "Weights") or []}
        elif n["name"] == "Video":
            content = _child_value(n, "Content")
            if isinstance(content, (bytes, bytearray)) and content:
                videos[oid] = {"name": name, "sha256": hashlib.sha256(bytes(content)).hexdigest()}
                video_bytes[oid] = bytes(content)
        elif n["name"] == "Texture":
            texnodes[oid] = {"name": name, "fileName": _child_value(n, "FileName"),
                             "relativeFilename": _child_value(n, "RelativeFilename")}
    links, plinks = [], []
    for c in (top.get("Connections") or {"children": []})["children"]:
        if c["name"] == "C" and len(c["props"]) >= 3 and c["props"][0] == "OO":
            links.append((c["props"][1], c["props"][2]))
        elif c["name"] == "C" and len(c["props"]) >= 4 and c["props"][0] == "OP":
            plinks.append((c["props"][1], c["props"][2], str(c["props"][3])))
    # Each material's textures by the property they feed (OP texture -> material, "EmissiveColor" ...), each texture's
    # image from its Video (OO video -> texture): embedded bytes when the export embedded them, else the file it names.
    video_of = {owner: child for child, owner in links if child in video_bytes and owner in texnodes}
    for tex, owner, prop in plinks:
        if tex in texnodes and owner in materials:
            t = texnodes[tex]
            sc.material_textures.setdefault(materials[owner], []).append({
                "property": prop, "texture": t["name"], "fileName": t["fileName"],
                "relativeFilename": t["relativeFilename"], "embedded": video_bytes.get(video_of.get(tex))})
    geom_of, mats_of, skin_of, clusters_of, bone_of = {}, {}, {}, {}, {}
    for child, owner in links:
        if child in sc.models and owner in sc.models:
            sc.models[child]["parent"] = owner
        elif child in geoms and owner in sc.models:
            geom_of.setdefault(owner, child)
        elif child in materials and owner in sc.models:
            mats_of.setdefault(owner, []).append(materials[child])
        elif child in skins and owner in geoms:
            skin_of[owner] = child
        elif child in clusters and owner in skins:
            clusters_of.setdefault(owner, []).append(child)
        elif child in sc.models and owner in clusters:
            bone_of[owner] = child
    sc.textures = sorted(videos.values(), key=lambda v: v["name"])

    def world(mid, seen=None):
        seen = seen or set()
        m = sc.models[mid]
        if m["parent"] in sc.models and m["parent"] not in seen:
            seen.add(mid)
            return mat_mul(world(m["parent"], seen), m["local"])
        return m["local"]

    for mid, mdl in sorted(sc.models.items(), key=lambda kv: kv[1]["name"]):
        if mdl["type"] == "LimbNode":
            w = world(mid)
            parent = sc.models.get(mdl["parent"], {})
            sc.bones.append({"name": mdl["name"], "parent": parent.get("name") if parent.get("type") == "LimbNode" else None,
                             "world": w, "position": [w[0][3], w[1][3], w[2][3]]})
    # a bind-pose TransformLink is the bone's own world matrix when skinned; prefer it to the node chain
    for cid, cl in clusters.items():
        bone = sc.models.get(bone_of.get(cid))
        if bone and isinstance(cl["link"], list) and len(cl["link"]) == 16:
            link = _from_fbx_columns(cl["link"])
            for b in sc.bones:
                if b["name"] == bone["name"]:
                    b["world"], b["position"] = link, [link[0][3], link[1][3], link[2][3]]
    for mid, mdl in sorted(sc.models.items(), key=lambda kv: kv[1]["name"]):
        gid = geom_of.get(mid)
        if gid is None or mdl["name"].upper().startswith("UCX_"):
            continue
        g = geoms[gid]
        verts = _child_value(g, "Vertices") or []
        pvi = _child_value(g, "PolygonVertexIndex") or []
        w = mat_mul(world(mid), mdl["geo"])
        local_pts = [verts[i:i + 3] for i in range(0, len(verts) - 2, 3)]
        pts = [mat_point(w, p) for p in local_pts]
        polys, cur = [], []
        for v in pvi:
            if v < 0:
                cur.append(~v)
                polys.append(cur)
                cur = []
            else:
                cur.append(v)
        if cur:
            sc.notes.append("%s: the last polygon has no end marker" % mdl["name"])
            polys.append(cur)
        pv_to_v = [v for p in polys for v in p]
        poly_of_pv = [i for i, p in enumerate(polys) for _ in p]
        uvs = _layer_values(g, "LayerElementUV", "UV", "UVIndex", len(pv_to_v), pv_to_v, poly_of_pv, 2)
        normals = _layer_values(g, "LayerElementNormal", "Normals", "NormalsIndex", len(pv_to_v), pv_to_v, poly_of_pv, 3)
        mat_layer = next((c for c in g["children"] if c["name"] == "LayerElementMaterial"), None)
        mat_idx = _child_value(mat_layer, "Materials") if mat_layer else None
        mat_map = _child_value(mat_layer, "MappingInformationType", "AllSame") if mat_layer else None
        names = mats_of.get(mid, [])
        if isinstance(mat_idx, list) and mat_map == "ByPolygon":
            used = sorted(set(i for i in mat_idx if 0 <= i < len(names)))
        else:
            used = [0] if names else []
        skin = skin_of.get(gid)
        bones = sorted(sc.models[bone_of[c]]["name"] for c in clusters_of.get(skin, []) if bone_of.get(c) in sc.models)
        tris = sum(max(0, len(p) - 2) for p in polys)
        bmin = [min(p[k] for p in pts) for k in range(3)] if pts else [0, 0, 0]
        bmax = [max(p[k] for p in pts) for k in range(3)] if pts else [0, 0, 0]
        vol = 0.0
        for p in polys:
            for i in range(1, len(p) - 1):
                a, b, c = pts[p[0]], pts[p[i]], pts[p[i + 1]]
                vol += (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                        + a[2] * (b[0] * c[1] - b[1] * c[0])) / 6.0
        sc.meshes.append({
            "name": mdl["name"], "modelId": mid, "world": w, "positions": pts, "polyList": polys,
            "uvLayers": uvs, "normalLayers": normals, "materialIndex": mat_idx if mat_map == "ByPolygon" else None,
            "vertexCount": len(pts), "polygons": len(polys), "triangles": tris,
            "quads": sum(1 for p in polys if len(p) == 4), "ngons": sum(1 for p in polys if len(p) > 4),
            "largestPolygon": max((len(p) for p in polys), default=0),
            "materials": [names[i] for i in used], "uvSets": len(uvs), "skinned": bool(bones), "bones": bones,
            "clusters": [{"bone": sc.models[bone_of[c]]["name"], "indexes": clusters[c]["indexes"],
                          "weights": clusters[c]["weights"]} for c in clusters_of.get(skin, []) if bone_of.get(c) in sc.models],
            "boxMin": [round(v, 6) for v in bmin], "boxMax": [round(v, 6) for v in bmax],
            "sizeStuds": [round(bmax[k] - bmin[k], 6) for k in range(3)],
            "signedVolume": round(vol, 9), "mirroredNode": mat_det3(w) < 0,
            "shells": shells(pts, polys),
        })
    return sc


def shells(pts, polys, digits=6):
    """The mesh's connected parts, each [{faces, closed, signedVolume, boxMin, boxMax}], from file-world points.

    Vertices are welded by position (rounded to `digits` decimals), so a part split along a UV seam or a hard edge is
    still one part. A part is closed when every edge is used by exactly two of its faces; only then does its signed volume
    say which way its faces wind (positive: outward, L122). An open part answers closed false and no verdict. The axis
    mapping to Roblox (-X, +Y, -Z) keeps the sign: its determinant is +1."""
    key_of, keys = {}, []
    for p in pts:
        k = tuple(round(float(c), digits) for c in p)
        keys.append(key_of.setdefault(k, len(key_of)))
    parent = list(range(len(key_of)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for poly in polys:
        if not poly:
            continue
        r0 = find(keys[poly[0]])
        for v in poly[1:]:
            r = find(keys[v])
            if r != r0:
                parent[r] = r0
    groups = {}
    for fi, poly in enumerate(polys):
        if poly:
            groups.setdefault(find(keys[poly[0]]), []).append(fi)
    out = []
    for faces in sorted(groups.values(), key=lambda f: f[0]):
        edges, vol = {}, 0.0
        used = set()
        for fi in faces:
            poly = polys[fi]
            ks = [keys[v] for v in poly]
            for i in range(len(ks)):
                a, b = ks[i], ks[(i + 1) % len(ks)]
                if a != b:
                    e = (a, b) if a < b else (b, a)
                    edges[e] = edges.get(e, 0) + 1
            for i in range(1, len(poly) - 1):
                a, b, c = pts[poly[0]], pts[poly[i]], pts[poly[i + 1]]
                vol += (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                        + a[2] * (b[0] * c[1] - b[1] * c[0])) / 6.0
            used.update(poly)
        closed = bool(edges) and all(n == 2 for n in edges.values())
        bp = [pts[v] for v in used]
        out.append({"faces": len(faces), "closed": closed, "signedVolume": round(vol, 9) if closed else None,
                    "boxMin": [round(min(p[k] for p in bp), 6) for k in range(3)],
                    "boxMax": [round(max(p[k] for p in bp), 6) for k in range(3)]})
    return out


# ------------------------------------------------------------------------------------------------- derived answers
def rbx_box(mesh):
    """(size, center) of a mesh's box on Roblox axes, studs. The MeshPart is placed by this center (L76)."""
    a, b = rbx_point(mesh["boxMin"]), rbx_point(mesh["boxMax"])
    lo, hi = [min(a[k], b[k]) for k in range(3)], [max(a[k], b[k]) for k in range(3)]
    return [round(hi[k] - lo[k], 6) for k in range(3)], [round((hi[k] + lo[k]) / 2, 6) for k in range(3)]


def scene_rbx_box(sc):
    """(size, center) of every mesh together on Roblox axes."""
    if not sc.meshes:
        return [0, 0, 0], [0, 0, 0]
    pts = []
    for m in sc.meshes:
        pts += [rbx_point(m["boxMin"]), rbx_point(m["boxMax"])]
    lo = [min(p[k] for p in pts) for k in range(3)]
    hi = [max(p[k] for p in pts) for k in range(3)]
    return [round(hi[k] - lo[k], 6) for k in range(3)], [round((hi[k] + lo[k]) / 2, 6) for k in range(3)]


def _pack(fmt, values):
    return struct.pack("<%d%s" % (len(values), fmt), *values)


def mesh_hash(sc, digits=6):
    """sha256 of what Studio would build from the file - every mesh's name, world transform, positions, polygons, UVs,
    normals, material names and skin, plus every embedded image's bytes - rounded to `digits` decimals so float noise
    does not count. The file's own bytes (timestamps, ids) are left out: Blender writes different bytes for an identical
    scene on every export (RB1475). Animation curves are left out too: a Model upload brings no animation (RB1026).

    None when the scene holds no mesh: Studio builds nothing from it, and the hash of nothing would match every other
    mesh-less file (a cut file, a bones-only export) so a dedupe would answer one file's asset id for another."""
    if not sc.meshes:
        return None
    h = hashlib.sha256()
    q = 10 ** digits

    def r(vals):
        return [int(round(float(v) * q)) for v in vals]

    for m in sorted(sc.meshes, key=lambda m: m["name"]):
        h.update(b"MESH\x00" + m["name"].encode("utf-8") + b"\x00")
        h.update(_pack("q", r(v for row in m["world"] for v in row)))
        h.update(_pack("q", r(v for p in m["positions"] for v in p)))
        for p in m["polyList"]:
            h.update(_pack("i", [len(p)] + list(p)))
        for layer in m["uvLayers"]:
            h.update(b"UV\x00" + _pack("q", r(v for t in layer for v in (t or (0, 0)))))
        for layer in m["normalLayers"]:
            h.update(b"N\x00" + _pack("q", r(round(v, 5) for t in layer for v in (t or (0, 0, 0)))))
        h.update(("MAT\x00" + "\x00".join(m["materials"])).encode("utf-8"))
        if m["materialIndex"]:
            h.update(_pack("i", m["materialIndex"]))
        for cl in sorted(m["clusters"], key=lambda c: c["bone"]):
            h.update(b"SKIN\x00" + cl["bone"].encode("utf-8") + b"\x00")
            h.update(_pack("i", list(cl["indexes"])))
            h.update(_pack("q", r(cl["weights"])))
    for b in sc.bones:
        h.update(b"BONE\x00" + b["name"].encode("utf-8") + b"\x00" + (b["parent"] or "").encode("utf-8"))
        h.update(_pack("q", r(v for row in b["world"] for v in row)))
    for t in sc.textures:
        h.update(b"TEX\x00" + t["sha256"].encode("ascii"))
    return h.hexdigest()


def skinned_size_check(sc):
    """{skinned, largestMesh, largestStuds, overLimit, limit} - a skinned file whose largest mesh axis passes 2048 studs is
    shrunk on import with its bones left at full size (RB1023, RB1472)."""
    largest, lname = 0.0, None
    for m in sc.meshes:
        s = max(m["sizeStuds"]) if m["sizeStuds"] else 0.0
        if s > largest:
            largest, lname = s, m["name"]
    skinned = any(m["skinned"] for m in sc.meshes)
    return {"skinned": skinned, "largestMesh": lname, "largestStuds": round(largest, 3),
            "overLimit": largest > ROBLOX_MAX_STUDS, "limit": ROBLOX_MAX_STUDS,
            "shrinkFactor": round(ROBLOX_MAX_STUDS / largest, 6) if largest > ROBLOX_MAX_STUDS else 1.0}


def emissive_textures(sc):
    """[{material, property, texture, image, embedded, found}] for every texture a material feeds into an emissive
    property (Blender's exporter writes EmissiveColor; EmissiveFactor and Emissive are read too). An FBX upload turns
    each one into SurfaceAppearance.EmissiveMaskContent rbxassetid://6, an asset that does not exist (RB2766). image is
    where the picture is: "embedded" (the export embedded it), else the file the texture names - RelativeFilename beside
    the FBX first, then FileName - and found says whether that file is on this machine."""
    import os  # noqa: PLC0415
    here = os.path.dirname(os.path.abspath(sc.path)) if sc.path else ""
    out = []
    for mat in sorted(sc.material_textures):
        for t in sc.material_textures[mat]:
            if not t["property"].lower().startswith("emissive"):
                continue
            row = {"material": mat, "property": t["property"], "texture": t["texture"], "embedded": bool(t["embedded"])}
            if t["embedded"]:
                row.update(image="embedded", found=True)
            else:
                cands = []
                if isinstance(t["relativeFilename"], str) and t["relativeFilename"]:
                    cands.append(os.path.normpath(os.path.join(here, t["relativeFilename"].replace("\\", os.sep))))
                if isinstance(t["fileName"], str) and t["fileName"]:
                    cands.append(t["fileName"])
                hit = next((c for c in cands if os.path.isfile(c)), None)
                row.update(image=hit or (cands[0] if cands else None), found=bool(hit))
            out.append(row)
    return out


def emissive_image_bytes(sc, row):
    """The picture of one emissive_textures row as bytes, or None when it is neither embedded nor on this machine."""
    for t in sc.material_textures.get(row["material"], []):
        if t["texture"] == row["texture"] and t["property"] == row["property"]:
            if t["embedded"]:
                return t["embedded"]
            break
    if row.get("found") and row.get("image") not in (None, "embedded"):
        with open(row["image"], "rb") as fh:
            return fh.read()
    return None


def preview_payload(sc):
    """What rbx_mesh_preview takes to show this file in Studio without an upload, built from the file itself: every mesh
    merged into one {vertices, triangles, cornerUVs, cornerNormals, triangleCount, meshes}, on Roblox axes in studs.

    Points and normals go through the measured import mapping (file (-X, +Y, -Z), RB1156); its determinant is +1, so each
    polygon's winding is kept and fanned into triangles. A UV's V is flipped (1 - v): Roblox's image origin is the top
    (RB2138). cornerNormals is None when a mesh carries no normal layer, cornerUVs None when one has no UV layer. A file
    whose axis settings are not the measured ones answers None: rbx_axes() says so and the caller refuses."""
    if not sc.rbx_axes():
        return None
    verts, tris, uvs, nrms = [], [], [], []
    have_uv = all(m["uvLayers"] for m in sc.meshes)
    have_n = all(m["normalLayers"] for m in sc.meshes)
    for m in sc.meshes:
        base = len(verts) // 3
        for p in m["positions"]:
            verts += rbx_point(p)
        uv = m["uvLayers"][0] if m["uvLayers"] else None
        nl = m["normalLayers"][0] if m["normalLayers"] else None
        pv = 0
        for poly in m["polyList"]:
            corners = list(range(pv, pv + len(poly)))
            for i in range(1, len(poly) - 1):
                for c, v in ((corners[0], poly[0]), (corners[i], poly[i]), (corners[i + 1], poly[i + 1])):
                    tris.append(base + v)
                    if have_uv:
                        t = uv[c] or (0.0, 0.0)
                        uvs += [float(t[0]), 1.0 - float(t[1])]
                    if have_n:
                        n = nl[c] or (0.0, 1.0, 0.0)
                        nrms += rbx_point(n)
            pv += len(poly)
    return {"vertices": verts, "triangles": tris, "cornerUVs": uvs if have_uv else None,
            "cornerNormals": nrms if have_n else None, "triangleCount": len(tris) // 3,
            "meshes": [m["name"] for m in sc.meshes]}
