"""Avatar-item rules checked in Blender, before anything is uploaded (bl_roblox_ugc_check).

The limits are Roblox's own, as Studio validates them: AvatarCreationService:GetValidationRules, stored per Studio build
by rbx_ugc_rules in mif_roblox/ugc_rules/<build>.json. Blender metres are Roblox studs with the bridge's
FBX settings (RB45). Every measurement is a MifBlender op (mesh_stats, mesh_quality, list_objects); nothing is guessed:
a check whose measurement is missing fails and says so.

check(call, obj, asset_type) takes `call(op, **params) -> dict`, the Blender bridge, so the server and a worker both use it.
NEVER RAISE.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RULES_DIR = os.path.join(HERE, "ugc_rules")
BODY_TYPES = ("DynamicHead", "Torso", "LeftArm", "RightArm", "LeftLeg", "RightLeg")


def _version_key(name):
    """A build file's name as a version: "0.1000.0.1" is newer than "0.748.0.7481526", which a string sort gets wrong."""
    return [(int(p), "") if p.isdigit() else (-1, p) for p in name[:-5].split(".")]


def latest_rules():
    """(build, rules) of the newest stored build by version number, or (None, None)."""
    try:
        files = sorted((f for f in os.listdir(RULES_DIR) if f.endswith(".json")), key=_version_key)
    except OSError:
        return None, None
    if not files:
        return None, None
    with open(os.path.join(RULES_DIR, files[-1]), encoding="utf-8") as fh:
        return files[-1][:-5], json.load(fh)


def asset_types(rules):
    acc = [k.split(".")[-1] for k in (rules.get("AccessoryRules") or {})]
    body = [k.split(".")[-1] for k in ((rules.get("MeshRules") or {}).get("BodyPartMaxTriangles") or {})]
    return sorted(acc), sorted(body)


def _limits(rules, asset_type):
    key = "Enum.AssetType." + asset_type
    mesh = rules.get("MeshRules") or {}
    acc = (rules.get("AccessoryRules") or {}).get(key)
    out = {"kind": "accessory" if acc else ("body" if key in (mesh.get("BodyPartMaxTriangles") or {}) else None)}
    if out["kind"] == "accessory":
        out["maxTriangles"] = mesh.get("AccessoryMaxTriangles")
        att = (acc.get("Attachments") or [{}])[0]
        out["box"] = att.get("Size")
        out["attachment"] = att.get("Name")
        out["rigidAllowed"] = acc.get("RigidAllowed")
    elif out["kind"] == "body":
        out["maxTriangles"] = mesh["BodyPartMaxTriangles"][key]
    tex = ((rules.get("TextureRules") or {}).get("AssetTextureLimits") or {}).get(key) or {}
    out["maxTexture"] = tex.get("ColorMapSize") or (rules.get("TextureRules") or {}).get("MaxTextureSize")
    out["boundsMin"], out["boundsMax"] = mesh.get("MeshPartBoundsMin"), mesh.get("MeshPartBoundsMax")
    return out


def check(call, obj, asset_type, rules=None, layered=None):
    """The item's checks against its type's limits. {ok, pass, type, build, limits, checks, failed}."""
    try:
        build = None
        if rules is None:
            build, rules = latest_rules()
        if not rules:
            return {"ok": False, "error": "no stored Roblox rules: run rbx_ugc_rules with Studio connected first"}
        lim = _limits(rules, asset_type)
        if lim["kind"] is None:
            acc, body = asset_types(rules)
            return {"ok": False, "refusedRule": "type", "accessoryTypes": acc, "bodyTypes": body,
                    "error": "%s is not an accessory or body-part type in Roblox's rules" % asset_type}
        st = call("mesh_stats", object=obj)
        if not st.get("ok"):
            return {"ok": False, "error": "mesh_stats on %s: %s" % (obj, st.get("error"))}
        q = call("mesh_quality", object=obj, gameReady={"triangleBudget": int(lim["maxTriangles"])}, conventions=False)
        if not q.get("ok"):
            return {"ok": False, "error": "mesh_quality on %s: %s" % (obj, q.get("error"))}
        meas = ((q.get("gameReady") or {}).get("measured")) or {}
        tris = (st.get("base") or {}).get("triangles")
        size = ((st.get("bboxWorld") or {}).get("size"))
        mats = st.get("materialSlots")
        checks = {}

        def put(name, ok, measured, limit):
            checks[name] = {"pass": bool(ok), "measured": measured, "limit": limit}

        put("triangles", isinstance(tris, int) and tris <= lim["maxTriangles"], tris, lim["maxTriangles"])
        # BOUNDARY EDGES COUNT (FOUND 2026-11-20): mesh_quality keeps them out of gameReady.measured, so an open plane
        # passed "closed"; they are read from its checks.
        closed = {k: meas.get(k) for k in ("nonManifoldEdges", "looseVerts", "looseEdges")}
        closed["boundaryEdges"] = (((q.get("checks") or {}).get("boundaryEdges")) or {}).get("count")
        # a missing measurement fails like any other (the module's rule): no default of 0 for looseEdges or looseVerts
        put("closed", all(v == 0 for v in closed.values()),
            closed, "0 non-manifold, 0 boundary, 0 loose")
        put("noNgons", meas.get("ngons") == 0, meas.get("ngons"), 0)
        put("uvIn01", meas.get("uvLoopsOutside01") == 0, meas.get("uvLoopsOutside01"), 0)
        put("scaleApplied", meas.get("scale") == [1.0, 1.0, 1.0], meas.get("scale"), [1.0, 1.0, 1.0])
        nmat = len(mats) if isinstance(mats, list) else mats if isinstance(mats, int) else None
        put("oneMaterial", nmat is not None and nmat <= 1, nmat, 1)
        if size and lim.get("box"):
            box = [lim["box"]["x"], lim["box"]["y"], lim["box"]["z"]]
            # Blender X, Y, Z (Z up) are Roblox X, -Z, Y: compare Blender (x, z, y) with the box's (x, y, z)
            rsize = [size[0], size[2], size[1]]
            put("insideAttachmentBox", all(a <= b + 1e-6 for a, b in zip(rsize, box)), rsize, box)
        elif lim.get("box"):
            put("insideAttachmentBox", False, None, lim["box"])
        if size and lim.get("boundsMax") is not None:
            put("meshPartBounds", all(lim["boundsMin"] <= s for s in size) and max(size) <= lim["boundsMax"] * 1.0001,
                size, [lim["boundsMin"], lim["boundsMax"]])
        if lim["kind"] == "accessory" and (layered or not lim.get("rigidAllowed")):
            names = [o.get("name") for o in (call("list_objects").get("objects") or [])]
            put("cages", obj + "_InnerCage" in names and obj + "_OuterCage" in names,
                [n for n in names if n and n.startswith(obj + "_")], [obj + "_InnerCage", obj + "_OuterCage"])
        failed = sorted(k for k, v in checks.items() if not v["pass"])
        return {"ok": True, "pass": not failed, "type": asset_type, "kind": lim["kind"], "build": build, "limits": lim,
                "checks": checks, "failed": failed, "object": obj}
    except Exception as exc:  # noqa: BLE001 - never raise
        return {"ok": False, "error": "the check failed: %r" % (exc,)}
