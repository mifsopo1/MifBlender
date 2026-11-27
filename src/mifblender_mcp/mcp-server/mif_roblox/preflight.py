"""A Blender mesh checked against Roblox's general mesh limits before it is exported and uploaded (bl_roblox_preflight).

Every limit is Roblox's own published words, cited per rule below with the page and its sentence (read 2026-11-20, the internal notes): the mesh specifications page for triangles, watertight, n-gons, bone influences and the root bone; the texture
specifications page for one material, one UV set and UVs in 0-1. Texture size is stated two ways on the page (4096 and
1024); uploads settled it: an upload keeps 4096 (RB1300 a Decal at 4096 and at 2048, measured by render; RB1027 a
SurfaceAppearance map at 4096 sharper than at 1024), so the limit is 4096. It is reported, not enforced, because this
check reads the mesh, not its images (list_images says each image's sizePx). Avatar items have their own, stricter rules:
bl_roblox_ugc_check (RB50: MaxTextureSize 1024, some maps 2048).

Every measurement is a MifBlender op (mesh_stats, mesh_quality, list_vertex_groups); a check whose measurement is
missing fails and says so. Avatar items have their own, stricter rules: bl_roblox_ugc_check.

check(call, obj) takes `call(op, **params) -> dict`, the Blender bridge, so the server and a worker both use it.

HEADLESS, NO BRIDGE (L126): inside any Blender - a `blender -b` batch script, or Blender's own Python console - call
check_object(obj) with a Blender object or its name. It runs MifBlender's own op functions in that process through
blender_call(), which drops unset parameters and adds ok to each answer the way the socket bridge does (an op's raw
answer carries no ok, so check() would call every mesh failed without it). Or run this file as the script:

    blender -b scene.blend -P tools/mcp-server/mif_roblox/preflight.py -- [object ...] [--out report.json]

which checks the named meshes (every mesh object when none is named), prints one PREFLIGHT_JSON line, and exits 0 when
every mesh passes, 1 when any fails or errs, 2 when there was no mesh to check.
NEVER RAISE.
"""

MESH_PAGE = "https://create.roblox.com/docs/art/modeling/specifications"
TEXTURE_PAGE = "https://create.roblox.com/docs/art/modeling/texture-specifications"

RULES = {
    "triangles": {"limit": 20000, "page": MESH_PAGE, "section": "Geometry: Budgets",
                  "says": "Individual meshes can not exceed 20,000 triangles."},
    "watertight": {"limit": "0 non-manifold, boundary or loose edges", "page": MESH_PAGE, "section": "Geometry: Watertight",
                   "says": "All geometry must be watertight without exposed holes or backfaces."},
    "noNgons": {"limit": 0, "page": MESH_PAGE, "section": "Geometry: No N-gons",
                "says": "Meshes must be in quads where possible."},
    "maxInfluences": {"limit": 4, "page": MESH_PAGE, "section": "Rigging and skinning: Max influences",
                      "says": "A vertex can not be influenced by more than 4 bones or joints."},
    "noRootInfluence": {"limit": 0, "page": MESH_PAGE, "section": "Rigging and skinning: No Root influences",
                        "says": "Do not apply influences to the Root bone or joint."},
    "oneMaterial": {"limit": 1, "page": TEXTURE_PAGE, "section": "Texture requirements",
                    "says": "Mesh objects can only have one material assigned."},
    "oneUvSet": {"limit": 1, "page": TEXTURE_PAGE, "section": "UV mapping",
                 "says": "Studio doesn't allow for multiple UV sets."},
    "uvIn01": {"limit": 0, "page": TEXTURE_PAGE, "section": "UV mapping",
               "says": "All UVs must be created within a 0:1 space."},
}
LIMITS = {
    "textureSize": {"limit": 4096, "page": TEXTURE_PAGE, "enforced": False,
                    "says": ["Roblox supports up to 4096x4096 pixel texture resolutions (4K).",
                             "Roblox supports up to 1024x1024 pixel spaces for texture maps."],
                    "measured": "an upload keeps 4096: a 4096 and a 2048 Decal each kept their own resolution, judged by "
                                "the width of a checker's edge in a render (ledger RB1300), and a 4096 SurfaceAppearance "
                                "map drew sharper than the same map at 1024 (RB1027); no Studio read answers an image's "
                                "stored size (RB1301)",
                    "why": "the preflight reads the mesh, not its images: list_images gives each image's sizePx. Avatar "
                           "items: bl_roblox_ugc_check (1024, some maps 2048, RB50)"},
}
UNCONFIRMED = {}


def check(call, obj):
    """{ok, pass, object, checks {rule: {pass, measured, limit, cite}}, failed, limits (known, not checked here), unconfirmed}."""
    try:
        if not isinstance(obj, str) or not obj:
            return {"ok": False, "ran": False, "error": "object is the name of a Blender mesh"}
        st = call("mesh_stats", object=obj)
        if not st.get("ok"):
            return {"ok": False, "error": "mesh_stats on %s: %s" % (obj, st.get("error"))}
        q = call("mesh_quality", object=obj, gameReady={"triangleBudget": RULES["triangles"]["limit"]},
                 conventions=False)
        if not q.get("ok"):
            return {"ok": False, "error": "mesh_quality on %s: %s" % (obj, q.get("error"))}
        vg = call("list_vertex_groups", object=obj, limit=1)
        if not vg.get("ok"):
            return {"ok": False, "error": "list_vertex_groups on %s: %s" % (obj, vg.get("error"))}
        meas = ((q.get("gameReady") or {}).get("measured")) or {}
        checks = {}

        def put(name, ok, measured):
            r = RULES[name]
            checks[name] = {"pass": bool(ok), "measured": measured, "limit": r["limit"],
                            "cite": {"page": r["page"], "section": r["section"], "says": r["says"]}}

        tris = (st.get("base") or {}).get("triangles")
        put("triangles", isinstance(tris, int) and tris <= RULES["triangles"]["limit"], tris)
        wt = {k: meas.get(k) for k in ("nonManifoldEdges", "looseEdges", "looseVerts")}
        wt["boundaryEdges"] = (((q.get("checks") or {}).get("boundaryEdges")) or {}).get("count")
        put("watertight", wt["nonManifoldEdges"] == 0 and wt["boundaryEdges"] == 0 and wt["looseEdges"] == 0, wt)
        put("noNgons", meas.get("ngons") == 0, meas.get("ngons"))
        mats = st.get("materialSlots")
        nmat = len(mats) if isinstance(mats, list) else mats if isinstance(mats, int) else None
        put("oneMaterial", nmat is not None and nmat <= 1, nmat)
        uvs = st.get("uvLayers")
        put("oneUvSet", isinstance(uvs, list) and len(uvs) == 1, uvs)
        out01 = meas.get("uvLoopsOutside01")
        put("uvIn01", isinstance(uvs, list) and len(uvs) >= 1 and out01 == 0, out01)
        inf = vg.get("influences")
        if vg.get("count", 0) == 0 or inf is None:
            skin = {"maxPerVertex": 0, "verticesOver4": 0, "armature": None, "note": "no vertex groups: not skinned"}
            put("maxInfluences", True, skin)
            put("noRootInfluence", True, {"verticesOnRoot": 0, "rootBones": []})
        else:
            put("maxInfluences", isinstance(inf.get("maxPerVertex"), int) and inf["maxPerVertex"] <= 4,
                {k: inf.get(k) for k in ("maxPerVertex", "verticesOver4", "verticesAtMax", "armature", "counted")})
            roots_known = inf.get("armature") is not None
            put("noRootInfluence", roots_known and inf.get("verticesOnRoot") == 0,
                {"verticesOnRoot": inf.get("verticesOnRoot"), "rootBones": inf.get("rootBones"),
                 "armature": inf.get("armature"),
                 "note": None if roots_known else "vertex groups but no Armature modifier: the root bone is unknown"})
        failed = sorted(k for k, v in checks.items() if not v["pass"])
        return {"ok": True, "pass": not failed, "object": obj, "checks": checks, "failed": failed,
                "limits": LIMITS, "unconfirmed": UNCONFIRMED}
    except Exception as exc:  # noqa: BLE001 - never raise
        return {"ok": False, "error": "the preflight failed: %r" % (exc,)}


# ------------------------------------------------------------------------------------- headless, inside Blender (L126)
ADDON_PACKAGE = "MifBlender"
BLENDER_OPS = ("ops_mesh", "ops_rig")     # mesh_stats and mesh_quality; list_vertex_groups


def _addon_dir():
    import os
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "blender-addon"))


def blender_call(package=ADDON_PACKAGE):
    """call(op, **params) -> dict over MifBlender's op functions in THIS Blender process, answering as the socket bridge
    does: unset (None) parameters dropped, {"ok": True, "endpoint": op, ...the op's answer}, a refusal {"ok": False, error}.

    The add-on is the one already imported, else the copy in this checkout's tools/blender-addon (the same tree as this
    file, so the rules and the ops agree), else an installed one. sys.dont_write_bytecode is set first so nothing is
    written into the tree. Raises ImportError when no MifBlender can be imported (check_object turns that into ok false)."""
    import importlib
    import os
    import sys
    sys.dont_write_bytecode = True
    if package not in sys.modules:
        here = _addon_dir()
        if os.path.isdir(os.path.join(here, package)) and here not in sys.path:
            sys.path.insert(0, here)
    common = importlib.import_module(package + ".ops_common")
    table = {}
    for name in BLENDER_OPS:
        table.update(importlib.import_module("%s.%s" % (package, name)).OPS)

    def call(op, **params):
        fn = table.get(op)
        if fn is None:
            return {"ok": False, "endpoint": op, "error": "%s is not one of the ops loaded here (%s)" % (op, ", ".join(BLENDER_OPS))}
        body = {k: v for k, v in params.items() if v is not None}
        try:
            res = fn(body)
        except common.MifOpError as exc:
            return dict(getattr(exc, "payload", None) or {}, ok=False, endpoint=op, error=str(exc))
        except Exception as exc:  # noqa: BLE001 - an op that raises is a failed call, as over the socket
            return {"ok": False, "endpoint": op, "error": "%s raised %s: %s" % (op, type(exc).__name__, exc)}
        out = {"ok": True, "endpoint": op}
        if isinstance(res, dict):
            out.update(res)
        elif res is not None:
            out["result"] = res
        return out
    return call


def check_object(obj, package=ADDON_PACKAGE):
    """check() on a Blender object (or its name) in this Blender process, with no bridge listening. Same answer as
    bl_roblox_preflight, plus headless true."""
    try:
        name = obj if isinstance(obj, str) else getattr(obj, "name", None)
        if not isinstance(name, str) or not name:
            return {"ok": False, "ran": False, "error": "obj is a Blender object or the name of one"}
        try:
            call = blender_call(package)
        except ImportError as exc:
            return {"ok": False, "ran": False, "error": "MifBlender could not be imported here: %s" % exc}
        return dict(check(call, name), headless=True)
    except Exception as exc:  # noqa: BLE001 - never raise
        return {"ok": False, "error": "the headless preflight failed: %r" % (exc,)}


def _main(argv):
    """The script entry: blender -b scene.blend -P preflight.py -- [object ...] [--out report.json]."""
    import json
    import bpy
    args = argv[argv.index("--") + 1:] if "--" in argv else []
    out_path = None
    if "--out" in args:
        i = args.index("--out")
        out_path = args[i + 1] if i + 1 < len(args) else None
        args = args[:i] + args[i + 2:]
    names = args or sorted(o.name for o in bpy.data.objects if o.type == "MESH")
    results = [check_object(n) for n in names]
    failed = [r.get("object") or n for n, r in zip(names, results) if not (r.get("ok") and r.get("pass"))]
    report = {"ok": bool(results) and not failed, "checked": len(results), "failedObjects": failed, "results": results,
              "error": None if results else "no mesh objects to check: nothing was checked"}
    text = json.dumps(report, default=str)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(text)
    print("PREFLIGHT_JSON " + text)
    return 2 if not results else 1 if failed else 0


if __name__ == "__main__":
    import sys as _sys
    _code = _main(_sys.argv)
    _sys.stdout.flush()
    _sys.exit(_code)
