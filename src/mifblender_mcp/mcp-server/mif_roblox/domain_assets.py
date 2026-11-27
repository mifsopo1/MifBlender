"""Assets already on the person's account, read in Studio: the imported MeshPart's pivot and the file's axis mapping.

    rbx_asset_probe   an EXISTING asset loaded into Workspace.MifScratch (InsertService:LoadAsset, a read of an asset the
                      person owns), each MeshPart's Size, MeshSize and PivotOffset reported, and - given the file it was
                      uploaded from - which file axis became which Roblox axis; then removed. rbx_asset_upload's
                      probe: true runs it on the asset it just made. Nothing is uploaded here.

WHY. Every piece of a set placed together needs the same pivot rule, and the upload answer says nothing of the pivot or
the facing (spec "Extend rbx_asset_upload: a probe that reports the imported MeshPart's pivot and axis mapping").

THE AXIS MAPPING IS MEASURED, NOT ASSUMED. The file's vertices as the file holds them (mif_fbx_hulls.read_mesh_bounds: its
own axes and units, before any importer turns them) give a box per axis around the file origin. Studio makes the file
origin the MeshPart's pivot (measured on 0.748: PivotOffset puts the pivot at the file origin), so the MeshPart's box in
pivot space is Size around -PivotOffset. Each of the 48 axis permutations with signs is scaled by the ratio of the summed
extents and held against that box; every one inside the tolerance is kept, so an axis whose box is symmetric about the
origin answers both signs (the box cannot tell them apart) instead of a guess. Roblox's front is -Z (CFrame.LookVector).

Members (0.748 dump): InsertService:LoadAsset, BasePart.Size, BasePart.PivotOffset, MeshPart.MeshSize. The placed copy is
removed with rbx_destroy (Parent = nil, undoable) and its absence read back.

NEVER RAISE: every failure is {"ok": False, ...}.
"""
import itertools
import os

__all__ = ["rbx_asset_probe"]

MAX_MESHES = 50
TOL_FRACTION = 0.01       # 1% of the largest extent
TOL_MIN = 1e-3            # studs
AXES = "XYZ"


def _T():
    from mif_roblox import tools
    return tools


def _vec(v):
    if isinstance(v, dict) and all(k in v for k in "xyz"):
        return [float(v["x"]), float(v["y"]), float(v["z"])]
    if isinstance(v, list) and len(v) == 3:
        return [float(x) for x in v]
    return None


def pivot_space_box(size, pivot_offset):
    """The MeshPart's box in its pivot's space, ([min x, y, z], [max x, y, z]), or (None, why). size is Size, pivot_offset
    {position, rotation (row-major 3x3)}. A point c of the part's space is R^T (c - p) in the pivot's."""
    s = _vec(size)
    po = pivot_offset if isinstance(pivot_offset, dict) else {}
    p, r = po.get("position"), po.get("rotation")
    if s is None or not (isinstance(p, list) and len(p) == 3 and isinstance(r, list) and len(r) == 9):
        return None, "Size or PivotOffset did not read as a vector and a CFrame"
    if any(min(abs(x), abs(abs(x) - 1)) > 1e-4 for x in r):
        return None, "PivotOffset is turned by other than a multiple of 90 degrees, so the box depends on the turn"
    pts = []
    for sx, sy, sz in itertools.product((-0.5, 0.5), repeat=3):
        c = [sx * s[0] - p[0], sy * s[1] - p[1], sz * s[2] - p[2]]
        pts.append([sum(r[j * 3 + i] * c[j] for j in range(3)) for i in range(3)])
    return ([min(q[i] for q in pts) for i in range(3)], [max(q[i] for q in pts) for i in range(3)]), None


def axis_mapping(file_min, file_max, box_min, box_max):
    """Which file axis (with sign) became each Roblox axis, from the file's vertex box and the MeshPart's pivot-space box.
    -> {axisMap {X: ["+X"], ...}, settled, studsPerFileUnit, maxError, tolerance, front}: every candidate inside the
    tolerance is listed, so an axis the boxes cannot tell answers more than one."""
    fe = [file_max[i] - file_min[i] for i in range(3)]
    re_ = [box_max[i] - box_min[i] for i in range(3)]
    if sum(fe) <= 0 or sum(re_) <= 0:
        return {"settled": False, "error": "a box with no extent cannot be mapped"}
    k = sum(re_) / sum(fe)
    tol = max(TOL_MIN, TOL_FRACTION * max(re_))
    scored = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            err = 0.0
            for j in range(3):
                i, sg = perm[j], signs[j]
                lo, hi = (file_min[i], file_max[i]) if sg > 0 else (-file_max[i], -file_min[i])
                err = max(err, abs(k * lo - box_min[j]), abs(k * hi - box_max[j]))
            scored.append((err, perm, signs))
    scored.sort(key=lambda x: x[0])
    keep = [c for c in scored if c[0] <= tol]
    if not keep:
        best = scored[0]
        return {"settled": False, "studsPerFileUnit": round(k, 6), "maxError": round(best[0], 6), "tolerance": tol,
                "nearest": {AXES[j]: ("+" if best[2][j] > 0 else "-") + AXES[best[1][j]] for j in range(3)},
                "error": "no axis permutation puts the file's box on the MeshPart's within %.4g studs (nearest %.4g): "
                         "the mesh was changed on import, or this is not the file the asset was made from" % (tol, best[0])}
    amap = {}
    for j in range(3):
        opts = []
        for _e, perm, signs in keep:
            o = ("+" if signs[j] > 0 else "-") + AXES[perm[j]]
            if o not in opts:
                opts.append(o)
        amap[AXES[j]] = opts
    # Roblox's front is -Z: the file direction that became -Z is the negation of what became +Z.
    front = [("-" if o[0] == "+" else "+") + o[1] for o in amap["Z"]]
    return {"axisMap": amap, "settled": all(len(v) == 1 for v in amap.values()), "studsPerFileUnit": round(k, 6),
            "maxError": round(keep[0][0], 6), "tolerance": tol, "fileFront": front,
            "note": None if all(len(v) == 1 for v in amap.values()) else
            "an axis with more than one answer has a box symmetric about the file origin (or two equal extents), so the "
            "boxes cannot tell the options apart"}


# ---- created or reused: the mesh and image ids inside a Model, each held against when the Model was made
# Re-uploading an identical FBX makes a new Model id but points it at the OLD mesh (RB1030); only an altered file gets new
# mesh and image ids (RB1032). Asset Privacy is decided when a mesh or image is made and is not retroactive (RB1033), and
# no API answers an asset's privacy (RB1029; Open Cloud's assetPrivacy is write-only), so a reused id is judged by its
# creation time: GET /assets/v1/assets/{id} answers each child's revisionCreateTime (revision 1, its creation), measured on
# a reused mesh a day older than its Model and on new ones 7 s older (RB1650).
CONTENT_PROPS = {"MeshPart": ["MeshId", "TextureID"], "SurfaceAppearance": ["ColorMap", "NormalMap", "RoughnessMap",
                                                                            "MetalnessMap"],
                 "SpecialMesh": ["MeshId", "TextureId"], "Decal": ["Texture"]}  # Decal covers Texture (IsA)
REUSE_WINDOW = 600.0      # s: a child made up to this long before its Model is the upload's own (measured 7 s, RB1650)
CLOCK_SLACK = 120.0       # s: this machine's clock against Roblox's, when the upload's start is known
MAX_CONTENT = 200


def _asset_num(text):
    import re
    m = re.search(r"(?:rbxassetid://|[?&]id=)(\d+)", text or "") if isinstance(text, str) else None
    return int(m.group(1)) if m else None


def content_ids(T, root):
    """[{path, className, prop, id}] for every mesh and image id under root (root included), or ([], error)."""
    rows, errors = [], []
    cls0 = T.rbx_get_props(root, ["ClassName"]).get("className")
    for cls, props in CONTENT_PROPS.items():
        found = T.rbx_find(class_name=cls, root=root, limit=MAX_CONTENT)
        if not found.get("ok", True):
            errors.append("find %s: %s" % (cls, found.get("error")))
            continue
        paths = [m["path"] for m in found.get("matches") or []]
        if found.get("truncated") is True:
            # the find stopped at its limit: a verdict on the first MAX_CONTENT would call the rest checked
            errors.append("find %s: more than %d under %s (truncated), so not every id was read"
                          % (cls, MAX_CONTENT, root))
        if cls0 == cls and root not in paths:
            paths.insert(0, root)
        for p in paths:
            g = T.rbx_get_props(p, props)
            if not g.get("ok"):
                errors.append("%s: %s" % (p, g.get("error")))
                continue
            vals = g.get("values") or {}
            for prop in props:
                n = _asset_num(vals.get(prop))
                if n:
                    rows.append({"path": p, "className": cls, "prop": prop, "id": n})
    return rows, errors


# An FBX uploaded with an emissive texture comes back with SurfaceAppearance.EmissiveMaskContent rbxassetid://6, an asset
# that does not exist: every play client then fails the SurfaceAppearance's TexturePack and draws the mesh untextured
# grey, while edit mode draws it fine (RB2766). Clearing the map (an empty Content) brought the textures back (RB2766).
SA_MAPS = ("ColorMapContent", "NormalMapContent", "RoughnessMapContent", "MetalnessMapContent", "EmissiveMaskContent")
BROKEN_ASSET = 6
MAX_SURFACES = 500


def _empty_content(v):
    """True for a Content that names nothing: "" / None, {handle: None}, or the plugin's text of an empty one - an empty
    Content reads back {"type": "Content", "text": "Content{SourceType=None}"} (plugin 0.17.15, RB2821)."""
    if v in (None, ""):
        return True
    if isinstance(v, dict):
        if set(v) == {"handle"}:
            return not v["handle"]
        return v.get("type") == "Content" and "SourceType=None" in str(v.get("text") or "")
    return False


def broken_maps(T, root, clear=False, dry_run=False):
    """Every SurfaceAppearance under root (root included) whose map Contents (SA_MAPS) name rbxassetid://6, the broken
    emissive mask an FBX upload leaves (RB2766). With clear (and not dry_run), each such map is set to an empty Content
    and read back on its own: cleared means the read-back names nothing - rbx_set_props' own verdict is not used, since it
    calls an empty Content read back as Content{SourceType=None} a mismatch (spec N67). Never raises.
    -> {ok, checked, surfaceAppearances, broken: [{path, prop, value}], cleared: [{path, prop, readBack}], error?}"""
    V = T.V
    parts = V.split_path(root) if isinstance(root, str) else list(root or [])
    out = {"ok": True, "checked": True, "root": V.join_path(parts), "surfaceAppearances": 0, "broken": []}
    try:
        cls0 = T._rbx("resolve", target=parts)
        found = T._rbx("find", root=parts, name=None, className="SurfaceAppearance", limit=MAX_SURFACES)
        if not found.get("ok"):
            return dict(out, ok=False, checked=False, error="the SurfaceAppearance search failed: %s" % found.get("error"))
        paths = [list(m.get("path") or []) for m in found.get("matches") or []]
        if cls0.get("className") == "SurfaceAppearance" and parts not in paths:
            paths.insert(0, parts)
        out["surfaceAppearances"] = len(paths)
        if len(paths) >= MAX_SURFACES:
            out["truncated"] = True
        errors = []
        for p in paths:
            g = T._rbx("get_props", target=p, props=list(SA_MAPS))
            if not g.get("ok"):
                errors.append("%s: %s" % (V.join_path(p), g.get("error")))
                continue
            vals = {k: V.decode(v) for k, v in T._obj(g.get("values")).items()}
            for prop in SA_MAPS:
                v = vals.get(prop)
                if isinstance(v, str) and _asset_num(v) == BROKEN_ASSET:
                    out["broken"].append({"path": V.join_path(p), "prop": prop, "value": v})
        if errors:
            out.update(ok=False, checked=False, error="%d SurfaceAppearance(s) did not read: %s" % (len(errors),
                                                                                                    "; ".join(errors[:5])))
        if clear and not dry_run and out["broken"]:
            out["cleared"], failed = [], []
            for b in out["broken"]:
                T.rbx_set_props(b["path"], {b["prop"]: ""})
                back = T._rbx("get_props", target=V.split_path(b["path"]), props=[b["prop"]])
                v = V.decode(T._obj(back.get("values")).get(b["prop"])) if back.get("ok") else {"error": back.get("error")}
                if back.get("ok") and _empty_content(v):
                    out["cleared"].append({"path": b["path"], "prop": b["prop"], "readBack": v})
                else:
                    failed.append("%s.%s reads %r" % (b["path"], b["prop"], v))
            if failed:
                out.update(ok=False, error="clearing failed: %s" % "; ".join(failed))
    except Exception as exc:  # noqa: BLE001 - never raise
        return dict(out, ok=False, checked=False, error="the map check failed: %r" % exc)
    if out["broken"]:
        n = len(out["broken"])
        done = len(out.get("cleared") or [])
        out["warning"] = ("%d SurfaceAppearance map(s) name rbxassetid://6, the broken emissive mask an FBX upload with an "
                          "emissive texture leaves: play clients fail its TexturePack and draw the mesh grey (RB2766). %s"
                          % (n, "Cleared and read back empty: %d." % done if clear and not dry_run else
                             "clear_broken_maps true sets each to an empty Content; give the glow back by uploading the "
                             "mask as its own JPG image and setting its image id (RB2768)"))
    return out


def _seconds(iso):
    import datetime
    try:
        return datetime.datetime.fromisoformat(str(iso).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def judge_reuse(model, children, started=None, privacy_on_since=None):
    """Pure: model {revisionCreateTime}, children {id: asset_state answer}. The reference is the upload's start less
    CLOCK_SLACK when started is given, else the Model's own creation less REUSE_WINDOW; a child made at or after it is
    created, one made before it reused. privacy_on_since (ISO time the creator turned Asset Privacy on) names each reused
    child made before it as Open Use (RB1033). -> {reference, how, rows {id: {...}}, created [], reused [], unread [],
    warnings []}."""
    if started is not None:
        ref, how = float(started) - CLOCK_SLACK, "the upload's start less %ds of clock slack" % CLOCK_SLACK
    else:
        mt = _seconds((model or {}).get("revisionCreateTime"))
        if mt is None:
            return {"error": "the Model's creation time did not read (%s)" % (model or {}).get("error")}
        ref, how = mt - REUSE_WINDOW, "the Model's creation less %ds" % REUSE_WINDOW
    since = _seconds(privacy_on_since) if privacy_on_since else None
    out = {"reference": round(ref, 3), "how": how, "rows": {}, "created": [], "reused": [], "unread": [], "warnings": []}
    for cid, st in sorted(children.items()):
        st = st or {}
        t = _seconds(st.get("revisionCreateTime")) if st.get("ok") else None
        row = {"assetType": st.get("assetType"), "made": st.get("revisionCreateTime")}
        if t is None:
            row["status"] = "unread"
            row["why"] = st.get("error") or "no creation time"
            out["unread"].append(cid)
        elif str(st.get("revisionId") or "1") != "1":
            row["status"] = "unread"
            row["why"] = "revision %s: the time read is its latest revision's, not its creation" % st.get("revisionId")
            out["unread"].append(cid)
        elif t >= ref:
            row["status"] = "created"
            out["created"].append(cid)
        else:
            row["status"] = "reused"
            out["reused"].append(cid)
            if since is not None:
                row["privacy"] = "Open Use" if t < since else "Restricted"
            if since is not None and t < since:
                out["warnings"].append(
                    "%s %s was made %s, before Asset Privacy was turned on (%s), and this Model reused it: it is Open Use, "
                    "so anyone can take it (RB1030, RB1033). Upload an altered copy (one vertex or one pixel changed) to "
                    "get a new, Restricted one (RB1032)" % (row["assetType"] or "asset", cid, row["made"], privacy_on_since))
            elif since is None:
                out["warnings"].append(
                    "%s %s was made %s, before this Model, which reused it: it keeps the privacy it had then (Open Use if "
                    "that was before the creator's Asset Privacy opt-in, RB1033). Pass privacy_on_since to have it named; "
                    "an altered file makes a new one (RB1032)" % (row["assetType"] or "asset", cid, row["made"]))
        out["rows"][cid] = row
    return out


def reuse_report(T, loaded_path, model_id, started=None, privacy_on_since=None):
    """Each mesh and image id under the loaded copy of model_id, created by it or reused. Never raises."""
    from mif_roblox import opencloud as OC
    try:
        rows, errors = content_ids(T, loaded_path)
        cut = any("(truncated)" in e for e in errors)
        if not rows:
            return {"ok": False, "error": "no mesh or image id was found under %s%s" % (
                loaded_path, (": " + "; ".join(errors[:3])) if errors else "")}
        ids = sorted({r["id"] for r in rows})
        children = {i: OC.asset_state(i) for i in ids}
        model = OC.asset_state(int(model_id)) if started is None else {}
        verdict = judge_reuse(model, children, started=started, privacy_on_since=privacy_on_since)
        if verdict.get("error"):
            return {"ok": False, "error": verdict["error"], "ids": rows}
        for r in rows:
            r["status"] = verdict["rows"][r["id"]]["status"]
        verdict["ids"] = rows
        verdict["modelMade"] = (model or {}).get("revisionCreateTime")
        verdict["ok"] = not verdict["unread"] and not errors
        if cut:
            verdict["truncated"] = True
        if errors:
            verdict["readErrors"] = errors
        if verdict["unread"]:
            verdict["error"] = "%d id(s) whose creation time did not read: %s" % (
                len(verdict["unread"]), ", ".join(str(i) for i in verdict["unread"][:10]))
        elif errors:
            verdict["error"] = "; ".join(errors[:3])
        return verdict
    except Exception as exc:  # noqa: BLE001 - never raise
        return {"ok": False, "error": "the reuse check failed: %s" % OC.exc_text(exc)}


def _file_meshes(file):
    import mif_fbx_hulls as H  # noqa: PLC0415 - the MCP server's own FBX reader, standard library only
    return H.read_mesh_bounds(file)


def rbx_asset_probe(asset_id: int, file: str = None, reuse: bool = False, privacy_on_since: str = None) -> dict:
    "Roblox Studio: load an asset already on the person's account (InsertService:LoadAsset - a read, nothing is uploaded) into Workspace.MifScratch, report each MeshPart's `BasePart.Size`, `MeshPart.MeshSize` and `BasePart.PivotOffset` and its box around the pivot, then remove it (rbx_destroy, its absence read back). With file, the FBX the asset was made from, also which file axis (with sign) became each Roblox axis, the studs per file unit, and which file direction became Roblox's front (-Z): measured by holding the file's vertex box against the MeshPart's pivot-space box, every axis permutation inside 1% listed - an axis symmetric about the file origin answers both signs rather than a guess. With reuse, every mesh and image id inside it (`MeshPart.MeshId`, `MeshPart.TextureID`, `SurfaceAppearance.ColorMap`/`NormalMap`/`RoughnessMap`/`MetalnessMap`, `SpecialMesh` and `Decal` ids) is called created or reused by its creation time (Open Cloud GET /assets/v1/assets/{id}, the person's key, never shown) against the Model's own, less 600 s: an identical re-upload makes a new Model over the OLD mesh, which keeps the privacy it had (RB1030, RB1033); privacy_on_since (the ISO time the creator turned Asset Privacy on) names each reused one made before it as Open Use, with the fix (an altered file, RB1032). Refused with the reason when Studio is not running or no plugin is connected. rbx_asset_upload's probe: true runs this on the asset it made; its reuse check runs on every Model upload."
    return probe(asset_id, file=file, reuse=reuse, privacy_on_since=privacy_on_since)


def probe(asset_id, file=None, reuse=False, privacy_on_since=None, started=None, meshes=True):
    """rbx_asset_probe's body; rbx_asset_upload also calls it with the upload's start time (started) and, for the reuse
    check alone, meshes False (no MeshPart rows, so a Model with none is not a failure there)."""
    T = _T()
    if not isinstance(reuse, bool):
        return {"ok": False, "ran": False, "error": "reuse is true or false"}
    if privacy_on_since is not None and _seconds(privacy_on_since) is None:
        return {"ok": False, "ran": False, "error": "privacy_on_since is an ISO time such as 2026-11-20T05:00:00Z, got %r"
                % (privacy_on_since,)}
    if not isinstance(asset_id, int) or isinstance(asset_id, bool) or asset_id <= 0:
        return {"ok": False, "ran": False, "error": "asset_id is a positive whole number, got %r" % (asset_id,)}
    fileinfo = None
    if file is not None:
        if not (isinstance(file, str) and os.path.isfile(file) and file.lower().endswith(".fbx")):
            return {"ok": False, "ran": False, "error": "file is an existing .fbx (the file the asset was made from), got %r" % (file,)}
        try:
            fileinfo = _file_meshes(file)
        except (OSError, ValueError, KeyError, IndexError) as exc:
            return {"ok": False, "ran": False, "error": "the file could not be read as FBX: %s" % exc}
        if not fileinfo.get("meshes"):
            return {"ok": False, "ran": False, "error": "the file holds no mesh with geometry (UCX_ hulls are not meshes)"}
    st = T.rbx_status()
    if not st.get("ok"):
        return {"ok": False, "ran": False, "refusedRule": "no-studio", "editorState": st.get("editorState"),
                "error": "nothing was loaded: Roblox Studio is not running or the MifBridge plugin is not connected (%s). "
                         "Launch it with rbx_launch first" % (st.get("error") or "no answer")}
    if not (T.rbx_find(name="MifScratch", root="Workspace").get("matches")):
        made = T.rbx_create("Folder", "Workspace", name="MifScratch")
        if not made.get("ok"):
            return dict(made, ok=False, step="scratch folder")
    loaded = T.rbx_call("InsertService", "LoadAsset", [asset_id], timeout=120)
    h = (loaded.get("value") or {}).get("handle") if isinstance(loaded.get("value"), dict) else None
    if not h:
        return {"ok": False, "ran": True, "step": "LoadAsset", "assetId": asset_id, "refusedRule": loaded.get("refusedRule"),
                "error": loaded.get("error") or "LoadAsset returned %r" % (loaded.get("value"),)}
    path, perr = T._place_handle(h, ["Workspace", "MifScratch"], "Probe%d" % asset_id)
    if perr:
        T.rbx_release_handle(h)
        return {"ok": False, "ran": True, "step": "place", "assetId": asset_id, "error": perr}
    out = {"ok": True, "assetId": asset_id, "loadedAs": path}
    try:
        if reuse:
            out["reuse"] = reuse_report(T, path, asset_id, started=started, privacy_on_since=privacy_on_since)
            if not out["reuse"].get("ok"):
                out.update(ok=False, error="reuse: %s" % out["reuse"].get("error"))
        if not meshes:
            return out
        root_cls = T.rbx_get_props(path, ["ClassName"]).get("className")
        fr = T.rbx_find(class_name="MeshPart", root=path, limit=MAX_MESHES)
        found = fr.get("matches") or []
        paths = ([path] if root_cls == "MeshPart" else []) + [m["path"] for m in found]
        if fr.get("truncated") is True or len(paths) > MAX_MESHES:
            # more MeshParts than are read: the rows below are the first MAX_MESHES, never the whole Model
            out.update(ok=False, truncated=True, error=(out.get("error") + "; " if out.get("error") else "")
                       + "more than %d MeshParts under %s: only the first %d were read, so this is not a verdict on the "
                         "whole asset" % (MAX_MESHES, path, MAX_MESHES))
        rows = []
        for mp in paths[:MAX_MESHES]:
            g = T.rbx_get_props(mp, ["Name", "Size", "MeshSize", "PivotOffset"])
            v = g.get("values") or {}
            row = {"path": mp, "name": v.get("Name"), "size": v.get("Size"), "meshSize": v.get("MeshSize"),
                   "pivotOffset": v.get("PivotOffset")}
            if not g.get("ok"):
                row["error"] = g.get("error")
            else:
                box, why = pivot_space_box(v.get("Size"), v.get("PivotOffset"))
                row["pivotSpaceBox"] = {"min": [round(x, 6) for x in box[0]], "max": [round(x, 6) for x in box[1]]} if box else None
                if why:
                    row["boxNote"] = why
            rows.append(row)
        out["meshParts"] = rows
        out["meshPartCount"] = len(paths)   # at most MAX_MESHES + 1 from the find: a floor when truncated
        if not rows:
            out.update(ok=False, error=(out.get("error") + "; " if out.get("error") else "")
                       + "the asset loaded as %s with no MeshPart in it" % root_cls)
        if fileinfo is not None and rows:
            fm = fileinfo["meshes"]
            out["file"] = {"path": file, "format": fileinfo["format"], "unitScaleCm": fileinfo["unitScaleCm"],
                           "axisSettings": fileinfo["axisSettings"], "meshes": [m["name"] for m in fm]}
            for row in rows:
                pair = next((m for m in fm if m["name"] == row.get("name")), None)
                if pair is None and len(fm) == 1 and len(rows) == 1:
                    pair = fm[0]
                if pair is None:
                    row["mapping"] = {"settled": False, "error": "no mesh in the file is named %r" % row.get("name")}
                elif not row.get("pivotSpaceBox"):
                    row["mapping"] = {"settled": False, "error": row.get("boxNote") or "no pivot-space box"}
                else:
                    sc = pair.get("nodeScale") or [1.0, 1.0, 1.0]
                    lo = [min(pair["min"][i] * sc[i], pair["max"][i] * sc[i]) for i in range(3)]
                    hi = [max(pair["min"][i] * sc[i], pair["max"][i] * sc[i]) for i in range(3)]
                    m = axis_mapping(lo, hi, row["pivotSpaceBox"]["min"], row["pivotSpaceBox"]["max"])
                    m["fileMesh"] = pair["name"]
                    if pair.get("nodeRotation") and any(abs(a) > 1e-6 for a in pair["nodeRotation"]):
                        m["nodeRotation"] = pair["nodeRotation"]
                        m["rotationNote"] = "the file's node is turned; the map is from the vertices as stored to Roblox"
                    row["mapping"] = m
            maps = [r.get("mapping") or {} for r in rows]
            if any(m.get("error") for m in maps):
                out.update(ok=False, error=(out.get("error") + "; " if out.get("error") else "") + "; ".join(
                    "%s: %s" % (r["name"], (r.get("mapping") or {}).get("error"))
                    for r in rows if (r.get("mapping") or {}).get("error")))
    finally:
        gone = T.rbx_destroy(path)
        left = T.rbx_find(name=path.split(".")[-1], root="Workspace.MifScratch").get("matches") or []
        out["removed"] = bool(gone.get("ok")) and not left
        if not out["removed"]:
            out.update(ok=False, error=(out.get("error") + "; " if out.get("error") else "")
                       + "the loaded copy could not be removed from %s: %s" % (path, gone.get("error") or "still there"))
        out["verified"] = out["ok"]
    return out


# ---------------------------------------------------------------- flatten and normalize a placed model
MAX_FLAT_PARTS = 200


def _indexed_walk(node, at, out, cat):
    """Depth-first over a tree op's node: each descendant's indexed path (Name[n] for the nth child of that name, in
    GetChildren order, which the plugin resolves from 0.16.3) and class."""
    kids = node.get("children") or []
    count, seen = {}, {}
    for c in kids:
        count[c.get("name")] = count.get(c.get("name"), 0) + 1
    for c in kids:
        nm = c.get("name")
        seen[nm] = seen.get(nm, 0) + 1
        seg = nm if count[nm] == 1 else "%s[%d]" % (nm, seen[nm])
        here = at + [seg]
        out.append({"path": here, "name": nm, "className": c.get("className"), "depth": len(here)})
        _indexed_walk(c, here, out, cat)


def flatten_normalize(path, asset_id, flatten=True, longest=None):
    """Leave Model > BasePart(s): every part under the placed model moved up to it (renamed first so each name is its
    own: the first part is Body), every other child of the model removed, Body made the PrimaryPart, the model scaled so
    its longest side is `longest` studs (Model:ScaleTo through rbx_scale_model), and the asset id kept as the AssetId
    attribute. Read back: the children are parts only, PrimaryPart is Body, the extents' longest side and the attribute."""
    T = _T()
    V = T.V
    cat, err = T._catalog()
    if err:
        return err
    parts = V.split_path(path) if isinstance(path, str) else list(path)
    steps, out = [], {"path": V.join_path(parts)}
    if T._resolve(parts)[0] != "Model":
        return {"ok": False, "error": "%s is not a Model: nothing to flatten" % V.join_path(parts)}
    tree = T._rbx("tree", root=parts, depth=20, limit=5000)
    if not tree.get("ok"):
        return dict(tree, ok=False, step="tree")
    nodes = []
    _indexed_walk(tree.get("root") or {}, list(parts), nodes, cat)
    based = [n for n in nodes if n["className"] and cat.is_a(n["className"], "BasePart")]
    if not based:
        return {"ok": False, "error": "no part under %s: nothing to flatten" % V.join_path(parts), "children": len(nodes)}
    if len(based) > MAX_FLAT_PARTS:
        return {"ok": False, "refusedRule": "budget", "error": "%d parts; at most %d" % (len(based), MAX_FLAT_PARTS)}
    # Parts inside parts (a part parented to a part) move too; work deepest first so a parent's path stays good.
    final, taken = {}, set()
    for i, n in enumerate(based):
        nm = "Body" if i == 0 else n["name"]
        if nm in taken or (i and nm == "Body"):
            k = 2
            while "%s_%d" % (n["name"], k) in taken:
                k += 1
            nm = "%s_%d" % (n["name"], k)
        taken.add(nm)
        final[tuple(n["path"])] = nm
    moved = []
    if flatten:
        def order(n):                         # deepest first, and the last of same-named siblings first, so a rename
            seg = n["path"][-1]               # never shifts the Name[n] index of one still to come
            k = int(seg[seg.rindex("[") + 1:-1]) if seg.endswith("]") and "[" in seg else 0
            return (-n["depth"], -k)
        for n in sorted(based, key=order):
            src, nm = n["path"], final[tuple(n["path"])]
            if src[-1] != nm:
                r = T.TOOLS["rbx_set_props"](src, {"Name": nm})
                if not r.get("ok"):
                    return dict(r, ok=False, step="rename %s" % V.join_path(src))
        for n in sorted(based, key=lambda x: -x["depth"]):
            src, nm = n["path"], final[tuple(n["path"])]
            if src[:-1] == parts:
                continue
            here = src[:-1] + [nm]
            r = T.TOOLS["rbx_reparent"](here, parts)
            if not r.get("ok"):
                return dict(r, ok=False, step="move %s" % V.join_path(here))
            moved.append(nm)
        top = T._rbx("tree", root=parts, depth=1, limit=5000)
        removed = []
        for c in (top.get("root") or {}).get("children") or []:
            if c.get("name") in taken and c.get("className") and cat.is_a(c["className"], "BasePart"):
                continue
            r = T.TOOLS["rbx_destroy"](parts + [c["name"]], all=True)
            if not r.get("ok"):
                return dict(r, ok=False, step="remove %s" % c.get("name"))
            removed.append({"name": c.get("name"), "className": c.get("className")})
        out.update(moved=moved, removed=removed)
    body = "Body" if flatten else None
    if flatten:
        r = T.TOOLS["rbx_set_props"](parts, {"PrimaryPart": V.join_path(parts + ["Body"])})
        steps.append({"primaryPart": r.get("ok")})
        if not r.get("ok"):
            return dict(r, ok=False, step="PrimaryPart")
    if longest is not None:
        ex = T.rbx_call(parts, "GetExtentsSize")
        sz = ex.get("value") if ex.get("ok") else None
        cur = max(sz.get("x"), sz.get("y"), sz.get("z")) if isinstance(sz, dict) else None
        if not cur:
            return dict(ex, ok=False, step="GetExtentsSize", error=ex.get("error") or "the extents read %r" % (sz,))
        sc = T.TOOLS["rbx_scale_model"](parts, factor=longest / cur)
        if not sc.get("ok"):
            return dict(sc, ok=False, step="ScaleTo")
        out["scaledBy"] = round(longest / cur, 6)
    at = T.TOOLS["rbx_attributes"](parts, set={"AssetId": int(asset_id)})
    if not at.get("ok"):
        return dict(at, ok=False, step="AssetId attribute")
    # The read-back, all at once.
    problems = []
    top = T._rbx("tree", root=parts, depth=1, limit=5000)
    kids = (top.get("root") or {}).get("children") or []
    out["children"] = [{"name": c.get("name"), "className": c.get("className")} for c in kids]
    if flatten and any(not (c.get("className") and cat.is_a(c["className"], "BasePart")) for c in kids):
        problems.append("a child of the model is not a part: %s" % [c.get("name") for c in kids
                                                                     if not cat.is_a(c.get("className") or "", "BasePart")])
    if flatten:
        pp = T.TOOLS["rbx_get_props"](parts, ["PrimaryPart"])
        got = V.decode((pp.get("values") or {}).get("PrimaryPart")) if pp.get("ok") else None
        out["primaryPart"] = got
        if got != V.join_path(parts + [body]):
            problems.append("PrimaryPart reads %r, not Body" % (got,))
    ex = T.rbx_call(parts, "GetExtentsSize")
    sz = ex.get("value") if ex.get("ok") else None
    if isinstance(sz, dict):
        out["extentsSize"] = [round(sz[k], 4) for k in "xyz"]
        out["longest"] = round(max(sz.values()), 4)
        if longest is not None and abs(out["longest"] - longest) > max(1e-3, longest * 1e-3):
            problems.append("the longest side reads %s, asked %s" % (out["longest"], longest))
    a = T.TOOLS["rbx_attributes"](parts)
    out["assetIdAttribute"] = (a.get("attributes") or {}).get("AssetId") if isinstance(a.get("attributes"), dict) else a.get("attributes")
    if out["assetIdAttribute"] != int(asset_id):
        problems.append("the AssetId attribute reads %r" % (out["assetIdAttribute"],))
    out.update(ok=not problems, verified=not problems)
    if problems:
        out["error"] = "; ".join(problems)
    return out
