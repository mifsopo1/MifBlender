"""Avatar bodies built in Studio from parts the person made: the R15 rig, each part's BodyPartR15, Roblox's own budgets.

    rbx_layered_clothing_build
                      a garment the person uploaded (an FBX Model: <G> mesh with <G>_InnerCage and <G>_OuterCage, which
                      the import makes into a MeshPart holding a WrapLayer) as a layered Accessory, added to bodies of
                      several scales (or bodies already built); each body's deformed garment read from captures.
    rbx_head_check    a dynamic head's FACS poses (EditableMesh:GetFacsPoses on its own uploaded mesh) against the 17 Roblox
                      requires, and five expressions set through its FaceControls and seen to move the face in captures.
    rbx_body_build    up to 15 parts put into an R15 body: an R15 mannequin made (rbx_mannequin), each part swapped in for
                      its slot with Humanoid:ReplaceBodyPartR15, given the slot's name and any rig attachment it lacks,
                      then Humanoid:BuildRigFromAttachments; read back through Humanoid:GetBodyPartR15 per part and the
                      15 joints. Roblox's per-asset budgets (AvatarCreationService:GetValidationRules, read from this
                      Studio) refuse a part before anything is made.

MEASURED ON 0.748:
- ReplaceBodyPartR15(slot, part) answers true, moves the part into the rig under ITS OWN name, destroys the slot's old part
  with everything in it (its attachments, its WrapTarget outer cage, its joint), and adds an OriginalSize Vector3Value
  holding the size it was given; the Humanoid then rescales the part by its body scales, so the rig's Size differs.
- A part not named for its slot reads GetBodyPartR15 Unknown, and one with no NeckRigAttachment gets no Neck from
  BuildRigFromAttachments. So the tool reads every slot's attachments from the mannequin BEFORE any swap, names each part
  for its slot after the swap, and gives it each attachment it lacks at the mannequin's position scaled by the size
  ratio (a part's own attachments - from the person's export - are kept as they are).

OUTER CAGES: the swap destroys the slot's old WrapTarget but keeps the NEW part's children, so a
part that holds its own WrapTarget keeps it. An FBX uploaded as a Model makes one: each <Part>_OuterCage mesh beside a
<Part> mesh becomes a WrapTarget (named <Part>WrapTarget) under that MeshPart, CageMeshContent the cage's new mesh id and
CageOrigin set. A part with no cage is given one by `cages` {slot: a WrapTarget path to copy, or {cage: rbxassetid,
origin: [x, y, z] or a CFrame}}: a WrapTarget named <slot>WrapTarget is made under the swapped-in part with that
CageMeshContent and CageOrigin and read back. Roblox's template cage meshes will not open for reading (RB57); a cage
uploaded by the person does.

Budgets (GetValidationRules on this Studio; Studio's numbers win): MeshRules.BodyPartMaxTriangles per asset type, summed
over the asset's parts that are MeshParts (counted through rbx_mesh_inspect); BodyPartRules <asset>.Bounds.<scale>.MaxSize,
which no single part may exceed. After the build each asset's assembled box is held against its MinSize and MaxSize.

Members (0.748 dump): Humanoid:ReplaceBodyPartR15, Humanoid:GetBodyPartR15, Humanoid:BuildRigFromAttachments,
AvatarCreationService:GetValidationRules, Attachment.CFrame, BasePart.Size, BasePart.CFrame.

NEVER RAISE: every failure is {"ok": False, ...}.
"""
import itertools
import time
from typing import Any

__all__ = ["rbx_body_build", "rbx_head_check", "rbx_layered_clothing_build"]

SCRATCH = "Workspace.MifScratch"
ASSET_OF = {"Head": "DynamicHead", "UpperTorso": "Torso", "LowerTorso": "Torso",
            "LeftUpperArm": "LeftArm", "LeftLowerArm": "LeftArm", "LeftHand": "LeftArm",
            "RightUpperArm": "RightArm", "RightLowerArm": "RightArm", "RightHand": "RightArm",
            "LeftUpperLeg": "LeftLeg", "LeftLowerLeg": "LeftLeg", "LeftFoot": "LeftLeg",
            "RightUpperLeg": "RightLeg", "RightLowerLeg": "RightLeg", "RightFoot": "RightLeg"}
SLOTS = tuple(ASSET_OF)
# joint name -> (part0, part1): the R15 hierarchy a built body must hold.
R15_JOINTS = {"Root": ("HumanoidRootPart", "LowerTorso"), "Waist": ("LowerTorso", "UpperTorso"),
              "Neck": ("UpperTorso", "Head"),
              "LeftShoulder": ("UpperTorso", "LeftUpperArm"), "LeftElbow": ("LeftUpperArm", "LeftLowerArm"),
              "LeftWrist": ("LeftLowerArm", "LeftHand"),
              "RightShoulder": ("UpperTorso", "RightUpperArm"), "RightElbow": ("RightUpperArm", "RightLowerArm"),
              "RightWrist": ("RightLowerArm", "RightHand"),
              "LeftHip": ("LowerTorso", "LeftUpperLeg"), "LeftKnee": ("LeftUpperLeg", "LeftLowerLeg"),
              "LeftAnkle": ("LeftLowerLeg", "LeftFoot"),
              "RightHip": ("LowerTorso", "RightUpperLeg"), "RightKnee": ("RightUpperLeg", "RightLowerLeg"),
              "RightAnkle": ("RightLowerLeg", "RightFoot")}
SCALE_TYPES = ("Classic", "ProportionsNormal", "ProportionsSlender")


def _T():
    from mif_roblox import tools
    return tools


def _V():
    from mif_roblox import values
    return values


def _no(error, **kw):
    return dict({"ok": False, "ran": False, "error": error}, **kw)


def _vec(v):
    if isinstance(v, dict) and all(k in v for k in "xyz"):
        return [float(v["x"]), float(v["y"]), float(v["z"])]
    return None


def _cf(v):
    if isinstance(v, dict) and isinstance(v.get("position"), list) and isinstance(v.get("rotation"), list):
        return [float(x) for x in v["position"]], [float(x) for x in v["rotation"]]
    return None


def world_box(parts):
    """The axis-aligned box ([min], [max]) around boxes [(position, rotation row-major 3x3, size)]."""
    lo, hi = [float("inf")] * 3, [float("-inf")] * 3
    for pos, rot, size in parts:
        for sx, sy, sz in itertools.product((-0.5, 0.5), repeat=3):
            c = [sx * size[0], sy * size[1], sz * size[2]]
            w = [pos[i] + sum(rot[i * 3 + j] * c[j] for j in range(3)) for i in range(3)]
            lo = [min(lo[i], w[i]) for i in range(3)]
            hi = [max(hi[i], w[i]) for i in range(3)]
    return lo, hi


def budget_check(rows, rules, scale_type):
    """rows {slot: {path, size [3], triangles or None}} -> [problem] against GetValidationRules: a part larger than its
    asset's MaxSize on any axis, an asset whose counted triangles pass BodyPartMaxTriangles. Each problem names the slot
    (or every slot of the asset) and the figure."""
    body = (rules or {}).get("BodyPartRules") or {}
    tri_max = ((rules or {}).get("MeshRules") or {}).get("BodyPartMaxTriangles") or {}
    problems, tris = [], {}
    for slot, r in rows.items():
        asset = ASSET_OF[slot]
        mx = _vec((((body.get("Enum.AssetType." + asset) or {}).get("Bounds") or {}).get(scale_type) or {}).get("MaxSize"))
        if mx and r.get("size") and any(r["size"][i] > mx[i] + 1e-6 for i in range(3)):
            problems.append({"slot": slot, "part": r["path"], "asset": asset, "rule": "Bounds.%s.MaxSize" % scale_type,
                             "limit": mx, "value": r["size"],
                             "error": "%s is %s studs, larger than %s's MaxSize %s" % (slot, r["size"], asset, mx)})
        if isinstance(r.get("triangles"), int):
            tris.setdefault(asset, []).append((slot, r["triangles"]))
    for asset, got in sorted(tris.items()):
        limit = tri_max.get("Enum.AssetType." + asset)
        total = sum(n for _s, n in got)
        if isinstance(limit, (int, float)) and total > limit:
            problems.append({"slot": [s for s, _n in got], "asset": asset, "rule": "MeshRules.BodyPartMaxTriangles",
                             "limit": limit, "value": total,
                             "error": "%s has %d triangles over %s, past its budget of %d" % (
                                 asset, total, ", ".join(s for s, _n in got), limit)})
    return problems


def _cage_source(T, slot, src):
    """(CageMeshContent, CageOrigin CFrame dict, None) for one `cages` entry, or (None, None, refusal)."""
    if isinstance(src, dict) and "cage" in src:
        c, o = src.get("cage"), src.get("origin", [0, 0, 0])
        if not (isinstance(c, str) and c.startswith("rbxassetid://") and c[len("rbxassetid://"):].isdigit()):
            return None, None, _no("cages.%s.cage is an rbxassetid://<digits> mesh, got %r" % (slot, c))
        if isinstance(o, list) and len(o) == 3 and all(isinstance(x, (int, float)) for x in o):
            o = {"position": [float(x) for x in o], "rotation": [1, 0, 0, 0, 1, 0, 0, 0, 1]}
        if not _cf(o):
            return None, None, _no("cages.%s.origin is [x, y, z] or {position, rotation}, got %r" % (slot, o))
        return c, o, None
    pp, err = T._target(src, "cages.%s" % slot)
    if err:
        return None, None, err
    path = _V().join_path(pp)
    g = T.rbx_get_props(path, ["ClassName", "CageMeshContent", "CageOrigin"])
    vals = g.get("values") or {}
    cls = g.get("className") or vals.get("ClassName")
    if not g.get("ok") or cls != "WrapTarget":
        return None, None, _no("cages.%s (%s) is %s, not a WrapTarget: give a body cage's WrapTarget or {cage, origin}" % (
            slot, path, cls or g.get("error")), refusedRule="type")
    c = vals.get("CageMeshContent")
    if not (isinstance(c, str) and c.startswith("rbxassetid://")):
        return None, None, _no("cages.%s (%s) holds no uploaded cage mesh (CageMeshContent %r)" % (slot, path, c))
    return c, vals.get("CageOrigin") or {"position": [0, 0, 0], "rotation": [1, 0, 0, 0, 1, 0, 0, 0, 1]}, None


def rbx_body_build(parts: dict, name: str = "MifBody", parent: Any = SCRATCH, scale_type: str = "Classic",
                   position: list = None, cages: dict = None) -> dict:
    "Roblox Studio: an R15 avatar body from the person's own parts. parts maps R15 slots (Head, UpperTorso, LowerTorso, LeftUpperArm, LeftLowerArm, LeftHand, RightUpperArm, RightLowerArm, RightHand, LeftUpperLeg, LeftLowerLeg, LeftFoot, RightUpperLeg, RightLowerLeg, RightFoot) to BaseParts already in the place; a slot left out keeps the default body's part. Budgets first, from this Studio's `AvatarCreationService:GetValidationRules`: an asset's triangles (MeshParts counted through rbx_mesh_inspect) past MeshRules.BodyPartMaxTriangles, or a part larger than its asset's Bounds.<scale_type>.MaxSize, is refused budget per part before anything is made. Then an R15 mannequin (rbx_mannequin) under parent, each part swapped in with `Humanoid:ReplaceBodyPartR15` (unanchored if it was anchored), named for its slot and given each rig attachment it lacks (the default body's, position scaled by the size ratio; its own are kept), and `Humanoid:BuildRigFromAttachments`. Read back: `Humanoid:GetBodyPartR15` of every part equal to its slot, the given parts in their slots (their OriginalSize, which the Humanoid's body scales leave as given), the 15 R15 joints with the parts they join, and each asset's assembled box inside its MinSize and MaxSize. Outer cages: a given part keeps a WrapTarget of its own (an FBX uploaded as a Model makes one from each <Part>_OuterCage mesh), and cages {slot: a WrapTarget path to copy, or {cage: rbxassetid, origin: [x, y, z]}} fits one to a swapped-in part that has none (`BaseWrap.CageMeshContent`, `BaseWrap.CageOrigin`), read back; each part's answer says whether it holds one and which cage. Nothing is uploaded."
    T, V = _T(), _V()
    if not isinstance(parts, dict) or not parts:
        return _no("parts is {slot: path} for one or more of %s" % ", ".join(SLOTS))
    bad = sorted(k for k in parts if k not in ASSET_OF)
    if bad:
        return _no("unknown slots %s; the R15 slots are %s" % (bad, ", ".join(SLOTS)))
    if scale_type not in SCALE_TYPES:
        return _no("scale_type is one of %s" % ", ".join(SCALE_TYPES))
    if not isinstance(name, str) or not name:
        return _no("name is the body's name")
    given = {}
    for slot, p in parts.items():
        pp, err = T._target(p, "parts.%s" % slot)
        if err:
            return err
        given[slot] = pp
    leaf = [pp[-1] for pp in given.values()]
    if len(set(leaf)) != len(leaf):
        return _no("two given parts share a name (%s): each is moved into the rig under its own name first, so the names "
                   "must differ" % sorted(n for n in leaf if leaf.count(n) > 1))
    clash = [(s, pp[-1]) for s, pp in given.items() if pp[-1] in ASSET_OF and pp[-1] != s]
    clash += [(s, pp[-1]) for s, pp in given.items() if pp[-1] in ("Humanoid", "HumanoidRootPart", "Animate", "Body Colors")]
    if clash:
        return _no("a given part is named for something else the rig holds (%s): rename it first" % clash)
    if cages is not None and not isinstance(cages, dict):
        return _no("cages is {slot: a WrapTarget path, or {cage: rbxassetid, origin: [x, y, z]}}")
    cages = cages or {}
    stray = sorted(k for k in cages if k not in given)
    if stray:
        return _no("cages names slots with no given part (%s): a slot left out keeps the default body's part and its cage" % stray)
    cat, err = T._catalog()
    if err:
        return err
    cage_src = {}
    for slot, src in cages.items():
        c, o, err = _cage_source(T, slot, src)
        if err:
            return err
        cage_src[slot] = (c, o)
    # 1. read each given part and the rules; refuse on budget before anything is made
    rows = {}
    for slot, pp in given.items():
        path = V.join_path(pp)
        g = T.rbx_get_props(path, ["ClassName", "Size", "Anchored"])
        if not g.get("ok"):
            return dict(g, ok=False, ran=False, slot=slot, target=path)
        cls = g.get("className") or (g.get("values") or {}).get("ClassName")
        if not cls or not cat.is_a(cls, "BasePart"):
            return _no("%s (%s) is a %s, not a part" % (slot, path, cls), refusedRule="type")
        row = {"path": path, "className": cls, "size": _vec((g.get("values") or {}).get("Size")), "triangles": None,
               "anchored": (g.get("values") or {}).get("Anchored") is True}
        if cls == "MeshPart":
            mi = T.rbx_mesh_inspect(path)
            if mi.get("ok") and isinstance(mi.get("triangles"), int):
                row["triangles"] = mi["triangles"]
            else:
                row["trianglesNote"] = "not counted: %s" % (mi.get("error") or "no triangle count")
        if slot in cage_src and (T.rbx_query(root=path, class_name="WrapTarget", page_size=10).get("matches") or []):
            return _no("%s (%s) already holds a WrapTarget, and cages gives it another: give one or the other" % (slot, path),
                       refusedRule="cage-twice")
        rows[slot] = row
    rr = T.rbx_call("AvatarCreationService", "GetValidationRules")
    rules = rr.get("value") if rr.get("ok") and isinstance(rr.get("value"), dict) else None
    if rules is None:
        return _no("Roblox's validation rules did not read (%s): budgets cannot be checked, nothing made" % rr.get("error"),
                   step="GetValidationRules")
    problems = budget_check(rows, rules, scale_type)
    if problems:
        return {"ok": False, "ran": False, "refusedRule": "budget", "problems": problems, "parts": rows,
                "scaleType": scale_type, "error": "; ".join(p["error"] for p in problems) + ": nothing was made"}
    # 2. the mannequin, and every slot's attachments read before any swap destroys them
    mq = T.rbx_mannequin(name=name, parent=parent, rig_type="R15", position=position)
    if not mq.get("ok"):
        return dict(mq, ok=False, step="mannequin")
    rig = mq["path"]
    hum = rig + ".Humanoid"
    out = {"path": rig, "humanoid": hum, "scaleType": scale_type, "parts": {}}
    template = {}
    for slot in given:
        sp = rig + "." + slot
        g = T.rbx_get_props(sp, ["Size"])
        atts = T.rbx_query(root=sp, class_name="Attachment", page_size=200)
        if not g.get("ok") or not atts.get("ok"):
            return dict(out, ok=False, ran=True, step="template", slot=slot,
                        error="the default body's %s did not read: %s" % (slot, g.get("error") or atts.get("error")))
        rowsA = []
        for m in atts.get("matches") or []:
            if V.split_path(m["path"])[:-1] != V.split_path(sp):
                continue
            a = T.rbx_get_props(m["path"], ["CFrame"])
            cf = _cf((a.get("values") or {}).get("CFrame"))
            if cf:
                rowsA.append((V.split_path(m["path"])[-1], cf))
        template[slot] = {"size": _vec((g.get("values") or {}).get("Size")), "attachments": rowsA}
    # 3. swap, name, attach
    for slot, pp in given.items():
        rec = {"from": V.join_path(pp)}
        r = T.rbx_call(hum, "ReplaceBodyPartR15", [slot, V.join_path(pp)])
        if not r.get("ok") or r.get("value") is not True:
            return dict(out, ok=False, ran=True, step="ReplaceBodyPartR15", slot=slot,
                        error="ReplaceBodyPartR15(%s) answered %s" % (slot, r.get("error") or r.get("value")))
        here = rig + "." + pp[-1]
        if pp[-1] != slot:
            nm = T.rbx_set_props(here, {"Name": slot})
            if not nm.get("ok"):
                return dict(out, ok=False, ran=True, step="name", slot=slot, error=nm.get("error"))
        sp = rig + "." + slot
        if rows[slot].get("anchored"):
            # An anchored part stays where it was made and the joints cannot move it: a body of preview parts
            # (rbx_mesh_preview anchors its MeshPart) built that way sat collapsed at the origin (measured, RB1343).
            ua = T.rbx_set_props(sp, {"Anchored": False})
            if not ua.get("ok"):
                return dict(out, ok=False, ran=True, step="unanchor", slot=slot, error=ua.get("error"))
            rec["unanchored"] = True
        own = {V.split_path(m["path"])[-1] for m in (T.rbx_query(root=sp, class_name="Attachment", page_size=200)
                                                     .get("matches") or []) if V.split_path(m["path"])[:-1] == V.split_path(sp)}
        tsize, nsize = template[slot]["size"], rows[slot]["size"]
        ratio = [nsize[i] / tsize[i] if tsize and nsize and tsize[i] else 1.0 for i in range(3)]
        added = []
        for an, (pos, rot) in template[slot]["attachments"]:
            if an in own:
                continue
            c = T.rbx_create("Attachment", sp, name=an, props={"CFrame": {
                "position": [pos[i] * ratio[i] for i in range(3)], "rotation": rot}})
            if not c.get("ok"):
                return dict(out, ok=False, ran=True, step="attachment", slot=slot, attachment=an, error=c.get("error"))
            added.append(an)
        rec.update(attachmentsAdded=added, attachmentsKept=sorted(own), sizeRatio=[round(x, 4) for x in ratio])
        if slot in cage_src:
            c, o = cage_src[slot]
            w = T.rbx_create("WrapTarget", sp, name=slot + "WrapTarget", props={"CageMeshContent": c, "CageOrigin": o})
            if not w.get("ok"):
                return dict(out, ok=False, ran=True, step="cage", slot=slot, error=w.get("error"))
            rec["cageFitted"] = {"path": w.get("path"), "cage": c}
        out["parts"][slot] = rec
    b = T.rbx_call(hum, "BuildRigFromAttachments")
    if not b.get("ok"):
        return dict(out, ok=False, ran=True, step="BuildRigFromAttachments", error=b.get("error"))
    # 4. read back
    problems = []
    boxes = {}
    for slot in SLOTS:
        sp = rig + "." + slot
        gb = T.rbx_call(hum, "GetBodyPartR15", [sp])
        g = T.rbx_get_props(sp, ["ClassName", "Size", "CFrame", "Anchored"])
        wt = T.rbx_query(root=sp, class_name="WrapTarget", page_size=10)
        vals = g.get("values") or {}
        rec = out["parts"].setdefault(slot, {"from": "default body"})
        rec.update(bodyPartR15=gb.get("value"), className=g.get("className") or vals.get("ClassName"),
                   size=vals.get("Size"), outerCage=bool(wt.get("matches")))
        if wt.get("matches"):
            wv = T.rbx_get_props(wt["matches"][0]["path"], ["CageMeshContent", "CageOrigin"]).get("values") or {}
            rec["cage"] = {"path": wt["matches"][0]["path"], "content": wv.get("CageMeshContent"), "origin": wv.get("CageOrigin")}
        if slot in cage_src:
            want_c, want_o = cage_src[slot]
            got = rec.get("cage") or {}
            go, wo = _cf(got.get("origin")), _cf(want_o)
            if got.get("content") != want_c or not go or not wo or any(abs(go[0][i] - wo[0][i]) > 1e-4 for i in range(3)):
                problems.append("%s's fitted cage reads %s at %s, not %s at %s" % (
                    slot, got.get("content"), go and go[0], want_c, wo and wo[0]))
        if vals.get("Anchored") is True:
            problems.append("%s is anchored: the joints cannot hold it to the body" % slot)
        if gb.get("value") != slot:
            problems.append("%s reads GetBodyPartR15 %s" % (slot, gb.get("value") or gb.get("error")))
        if slot in rows:
            # The Humanoid rescales a swapped-in part by its body scales (measured: a given head 1.27 x 1.30 x 1.28 read
            # 1.24 x 1.28 x 1.24 in the rig); ReplaceBodyPartR15's OriginalSize keeps the size it was given.
            og = T.rbx_get_props(sp + ".OriginalSize", ["Value"])
            orig = _vec((og.get("values") or {}).get("Value"))
            rec["originalSize"] = (og.get("values") or {}).get("Value")
            if orig is None or any(abs(orig[i] - rows[slot]["size"][i]) > 1e-3 for i in range(3)):
                problems.append("%s in the rig is not the given part (OriginalSize %s, given %s)" % (
                    slot, rec["originalSize"] or og.get("error"), rows[slot]["size"]))
        cf = _cf(vals.get("CFrame"))
        if cf and _vec(vals.get("Size")):
            boxes.setdefault(ASSET_OF[slot], []).append((cf[0], cf[1], _vec(vals["Size"])))
    from mif_roblox import domain_anim_data as A  # noqa: PLC0415 - the rig's joints, read as rbx_rig_create reads them
    joints, _root, err = A.rig_joints(rig)
    if err:
        return dict(out, ok=False, ran=True, step="joints", error=err.get("error"))
    have = {j["joint"]: (j["part0"], j["part1"]) for j in joints.values()}
    out["joints"] = {k: {"part0": v[0], "part1": v[1]} for k, v in sorted(have.items())}
    for jn, pair in R15_JOINTS.items():
        if have.get(jn) != pair:
            problems.append("joint %s %s, wanted %s" % (jn, "joins %s" % (have[jn],) if jn in have else "missing", pair))
    body = rules.get("BodyPartRules") or {}
    out["bounds"] = {}
    for asset, bx in sorted(boxes.items()):
        lo, hi = world_box(bx)
        size = [round(hi[i] - lo[i], 4) for i in range(3)]
        lim = ((body.get("Enum.AssetType." + asset) or {}).get("Bounds") or {}).get(scale_type) or {}
        mn, mx = _vec(lim.get("MinSize")), _vec(lim.get("MaxSize"))
        inside = (mn is None or all(size[i] >= mn[i] - 1e-6 for i in range(3))) and \
                 (mx is None or all(size[i] <= mx[i] + 1e-6 for i in range(3)))
        out["bounds"][asset] = {"size": size, "minSize": mn, "maxSize": mx, "inside": inside}
        if not inside:
            problems.append("%s assembles to %s studs, outside %s..%s (%s)" % (asset, size, mn, mx, scale_type))
    out.update(ok=not problems, verified=not problems, given=sorted(given),
               fromDefaultBody=sorted(s for s in SLOTS if s not in given),
               outerCages="%d of %d parts hold an outer cage (%d fitted from cages)" % (
                   sum(1 for v in out["parts"].values() if v.get("outerCage")), len(SLOTS), len(cage_src)))
    if problems:
        out["error"] = "; ".join(problems)
    return out


# ---------------------------------------------------------------------------------------------------------------- layered
# MEASURED ON 0.748:
# - An FBX uploaded through Open Cloud as a Model, holding <G>, <G>_InnerCage and <G>_OuterCage, loads as a MeshPart <G>
#   holding a WrapLayer <G>WrapLayer: CageMeshContent the outer cage's new mesh, ReferenceMeshContent the inner cage's,
#   CageOrigin and ReferenceOrigin set, Order 1, AutoSkin Disabled. Its skin bones come in under a RootPart, as RB1025.
# - That MeshPart copied as an Accessory's Handle and added with Humanoid:AddAccessory WRAPS: the shirt is drawn fitted to a
#   slender and to a wide body (Studio still adds an AccessoryRigidConstraint), while Handle.Size reads the undeformed
#   size on both. The deformed garment is not readable as a property, so its bounds are read from the capture: the front
#   view with the garment on against the same view with it set aside, the pixels that differ boxed, and the box turned
#   into studs at the garment's depth through the camera's vertical field of view.
LAYERED_ACCESSORY_TYPES = ("TShirt", "Shirt", "Pants", "Jacket", "Sweater", "Shorts", "DressSkirt", "LeftShoe", "RightShoe")
LAYERED_BODIES = {"slender": {"BodyTypeScale": 0, "ProportionScale": 1, "WidthScale": 0.75, "HeightScale": 1.05},
                  "wide": {"WidthScale": 1, "DepthScale": 1, "BodyTypeScale": 1, "ProportionScale": 0, "HeightScale": 0.9}}
DIFF_LEVEL = 40     # a pixel is the garment's when a channel differs by more than this, garment on against set aside
MIN_PIXELS = 200    # fewer differing pixels than this: nothing was drawn
CAM_BACK = 7.0      # studs in front of the UpperTorso the camera stands
CAM_FOV = 70.0      # vertical field of view the captures are taken with


def silhouette(dressed_rgba, bare_rgba, w, h, level=DIFF_LEVEL):
    """The box (x0, y0, x1, y1) and count of pixels that differ by more than level in any channel. Pure."""
    n, x0, y0, x1, y1 = 0, w, h, -1, -1
    for y in range(h):
        row = y * w * 4
        for x in range(w):
            i = row + x * 4
            if max(abs(dressed_rgba[i] - bare_rgba[i]), abs(dressed_rgba[i + 1] - bare_rgba[i + 1]),
                   abs(dressed_rgba[i + 2] - bare_rgba[i + 2])) > level:
                n += 1
                x0, y0, x1, y1 = min(x0, x), min(y0, y), max(x1, x), max(y1, y)
    return (x0, y0, x1, y1) if n else None, n


def studs_per_pixel(h_pixels, distance, fov=CAM_FOV):
    """At distance studs from a camera whose vertical field of view is fov degrees, one pixel of an h_pixels-high frame
    spans this many studs. Pure."""
    import math
    return 2.0 * distance * math.tan(math.radians(fov) / 2.0) / float(h_pixels)


def _layer_source(T, item):
    """(handle MeshPart path, WrapLayer path, its props, None) for a garment MeshPart or an Accessory, or a refusal."""
    V = _V()
    pp, err = T._target(item, "item")
    if err:
        return None, None, None, err
    path = V.join_path(pp)
    g = T.rbx_get_props(path, ["ClassName"])
    cls = g.get("className") or (g.get("values") or {}).get("ClassName")
    if not g.get("ok"):
        return None, None, None, dict(g, ok=False, ran=False)
    mesh = path + ".Handle" if cls == "Accessory" else path
    if cls not in ("Accessory", "MeshPart"):
        return None, None, None, _no("item %s is a %s: give the garment's MeshPart (holding a WrapLayer) or an Accessory" % (
            path, cls), refusedRule="type")
    wl = T.rbx_query(root=mesh, class_name="WrapLayer", page_size=10).get("matches") or []
    wl = [m for m in wl if V.split_path(m["path"])[:-1] == V.split_path(mesh)]
    if not wl:
        return None, None, None, _no("%s holds no WrapLayer: a layered garment is a mesh with <name>_InnerCage and "
                                     "<name>_OuterCage beside it in the FBX, uploaded as a Model" % mesh,
                                     refusedRule="no-wrap-layer")
    props = T.rbx_get_props(wl[0]["path"], ["CageMeshContent", "ReferenceMeshContent", "CageOrigin", "ReferenceOrigin",
                                             "Order", "AutoSkin", "BindOffset"]).get("values") or {}
    for k in ("CageMeshContent", "ReferenceMeshContent"):
        if not (isinstance(props.get(k), str) and props[k].startswith("rbxassetid://")):
            return None, None, None, _no("%s's %s is %r, not an uploaded mesh: an EditableMesh cage is held but nothing "
                                         "wraps to it (RB63)" % (wl[0]["path"], k, props.get(k)), refusedRule="needs-upload")
    return mesh, wl[0]["path"], props, None


def _capture_bounds(T, rig, acc_path, name, at_part="UpperTorso"):
    """Front capture with the accessory on, then with it set aside (and put back): the garment's drawn box in pixels and
    in studs at the body's depth."""
    from mif_roblox import png as P
    V = _V()
    tc = T.rbx_get_props(rig + "." + at_part, ["CFrame"]).get("values") or {}
    c = (_cf(tc.get("CFrame")) or ([0, 0, 0], None))[0]
    cam = {"position": [c[0], c[1], c[2] - CAM_BACK], "look_at": c, "fov": CAM_FOV}
    on = T.rbx_screenshot(camera=cam, max_width=640, name="layered_%s" % name)
    if not on.get("ok"):
        return {"ok": False, "error": "capture: %s" % on.get("error")}
    rp = V.split_path(rig)
    aside = T.rbx_create("Folder", rp[:-1] or ["Workspace"], name=T._unique_name(rp[:-1] or ["Workspace"], "MifLayerAside"))
    leaf = V.split_path(acc_path)[-1]
    mv = T.rbx_reparent(acc_path, aside.get("path")) if aside.get("ok") else {"ok": False, "error": aside.get("error")}
    time.sleep(1.0)
    off = T.rbx_screenshot(camera=cam, max_width=640, name="layered_%s_bare" % name) if mv.get("ok") else mv
    back = T.rbx_reparent(aside["path"] + "." + leaf, rig) if mv.get("ok") else {"ok": False}
    if aside.get("ok"):
        T.rbx_destroy(aside["path"])
    out = {"files": {"on": on.get("file"), "off": off.get("file")}, "putBack": bool(back.get("ok"))}
    if not mv.get("ok") or not off.get("ok") or not back.get("ok"):
        return dict(out, ok=False, error="the set-aside view failed: %s" % (mv.get("error") or off.get("error") or back.get("error")))
    w, h, a = P.decode_rgba(open(on["file"], "rb").read())
    w2, h2, b = P.decode_rgba(open(off["file"], "rb").read())
    if (w, h) != (w2, h2):
        return dict(out, ok=False, error="the two captures differ in size (%sx%s, %sx%s)" % (w, h, w2, h2))
    box, n = silhouette(a, b, w, h)
    if not box or n < MIN_PIXELS:
        return dict(out, ok=False, pixels=n, error="the garment changes %d pixels of the front view: it is not drawn" % n)
    spp = studs_per_pixel(h, CAM_BACK)
    out.update(ok=True, pixels=n, boxPixels=list(box),
               width=round((box[2] - box[0] + 1) * spp, 3), height=round((box[3] - box[1] + 1) * spp, 3))
    return out


def rbx_layered_clothing_build(item: Any, accessory_type: str = "TShirt", bodies: dict = None, order: int = None,
                               auto_skin: str = None, bind_offset: Any = None, attachment: str = "BodyFrontAttachment",
                               name: str = "MifLayered", parent: Any = SCRATCH, at: list = None) -> dict:
    "Roblox Studio: layered clothing fitted and deformed. item is a garment the person uploaded - the MeshPart an FBX Model upload makes from <G>, <G>_InnerCage and <G>_OuterCage (it holds a `WrapLayer` with `BaseWrap.CageMeshContent` and `WrapLayer.ReferenceMeshContent`) - or an Accessory whose Handle holds one. An Accessory (`Accessory.AccessoryType` accessory_type: TShirt, Shirt, Pants, Jacket, Sweater, Shorts, DressSkirt, LeftShoe, RightShoe) is made with a copy as its Handle and an attachment named attachment; order, auto_skin and bind_offset set `WrapLayer.Order`, `WrapLayer.AutoSkin` and `WrapLayer.BindOffset` (read back). Then for each body - bodies {name: HumanoidDescription scales, or the path of a body already built (rbx_body_build)}, default slender and wide - a copy is added with `Humanoid:AddAccessory`, and its deformed bounds are read: the front view with the garment on against the same view with it set aside, the differing pixels boxed and turned into studs at the body's depth. `BasePart.Size` keeps the undeformed size on every body (measured), so the capture is the read-back. A cage that is not an uploaded mesh is refused needs-upload (an EditableMesh cage does not wrap, RB63). Nothing is uploaded."
    T, V = _T(), _V()
    if accessory_type not in LAYERED_ACCESSORY_TYPES:
        return _no("accessory_type is one of %s" % ", ".join(LAYERED_ACCESSORY_TYPES))
    if order is not None and (not isinstance(order, int) or isinstance(order, bool) or not 0 <= order <= 127):
        return _no("order is a whole number 0-127 (WrapLayer.Order: higher layers sit outside lower ones)")
    if auto_skin is not None and auto_skin not in ("Disabled", "EnabledPreserve", "EnabledOverride"):
        return _no("auto_skin is Disabled, EnabledPreserve or EnabledOverride")
    if bind_offset is not None:
        if isinstance(bind_offset, list) and len(bind_offset) == 3 and all(isinstance(x, (int, float)) for x in bind_offset):
            bind_offset = {"position": [float(x) for x in bind_offset], "rotation": [1, 0, 0, 0, 1, 0, 0, 0, 1]}
        if not _cf(bind_offset):
            return _no("bind_offset is [x, y, z] or {position, rotation}")
    if not isinstance(attachment, str) or not attachment.endswith("Attachment"):
        return _no("attachment is a body attachment name (BodyFrontAttachment, WaistCenterAttachment, ...)")
    bodies = LAYERED_BODIES if bodies is None else bodies
    if not isinstance(bodies, dict) or not bodies or not all(isinstance(k, str) and k.isidentifier() for k in bodies):
        return _no("bodies is {name: scales or a body's path}, names letters, digits and _")
    pparts, err = T._target(parent, "parent")
    if err:
        return err
    mesh, wl_path, wl, err = _layer_source(T, item)
    if err:
        return err
    for bn, spec in bodies.items():
        if isinstance(spec, str):
            hp = spec + ".Humanoid"
            if not T.rbx_get_props(hp, ["Name"]).get("ok"):
                return _no("bodies.%s (%s) has no Humanoid" % (bn, spec))
        elif not isinstance(spec, dict):
            return _no("bodies.%s is HumanoidDescription scales or a body's path" % bn)
    # 1. the accessory template
    aname = T._unique_name(pparts, name)
    acc = T.rbx_create("Accessory", pparts, name=aname, props={"AccessoryType": accessory_type})
    if not acc.get("ok"):
        return dict(acc, ok=False, step="accessory")
    apath = acc["path"]
    out = {"template": apath, "accessoryType": accessory_type, "source": {"mesh": mesh, "wrapLayer": wl_path, "props": wl}}
    c = T.rbx_clone(mesh, parent=apath, name="Handle")
    if not c.get("ok"):
        return dict(out, ok=False, ran=True, step="handle", error=c.get("error"))
    hpath = c["path"]
    T.rbx_set_props(hpath, {"Anchored": False, "CanCollide": False})
    for m in T.rbx_query(root=hpath, class_name="Motor6D", page_size=20).get("matches") or []:
        T.rbx_destroy(m["path"])  # the import's joint to its RootPart; the accessory is joined by AddAccessory
    if not (T.rbx_query(root=hpath, class_name="Attachment", page_size=50).get("matches") or []):
        a = T.rbx_create("Attachment", hpath, name=attachment)
        if not a.get("ok"):
            return dict(out, ok=False, ran=True, step="attachment", error=a.get("error"))
    hwl = hpath + "." + V.split_path(wl_path)[-1]
    want = {k: v for k, v in (("Order", order), ("AutoSkin", auto_skin), ("BindOffset", bind_offset)) if v is not None}
    if want:
        sp = T.rbx_set_props(hwl, want)
        if not sp.get("ok"):
            return dict(out, ok=False, ran=True, step="wrap layer", error=sp.get("error"))
    hs = T.rbx_get_props(hpath, ["Size"]).get("values") or {}
    out["handleSize"] = hs.get("Size")
    # 2. each body
    out["bodies"], problems = {}, []
    x0 = list(at or [2400, 200, 2000])
    for i, (bn, spec) in enumerate(bodies.items()):
        rec = {}
        if isinstance(spec, str):
            rig = spec
        else:
            mq = T.rbx_mannequin(name="%s_%s" % (aname, bn), parent=parent, rig_type="R15", scales=spec,
                                 position=[x0[0] + 12 * i, x0[1], x0[2]])
            if not mq.get("ok"):
                problems.append("%s: no mannequin: %s" % (bn, mq.get("error")))
                continue
            rig = mq["path"]
        rec["body"] = rig
        cp = T.rbx_clone(apath, parent=rig[: rig.rfind(".")] if "." in rig else "Workspace",
                         name=T._unique_name(V.split_path(rig)[:-1], "%s_%s" % (aname, bn)))
        if not cp.get("ok"):
            problems.append("%s: no copy: %s" % (bn, cp.get("error")))
            continue
        add = T.rbx_call(rig + ".Humanoid", "AddAccessory", [cp["path"]])
        worn = rig + "." + V.split_path(cp["path"])[-1]
        if not add.get("ok") or not T.rbx_get_props(worn, ["Name"]).get("ok"):
            problems.append("%s: AddAccessory did not put it on the body: %s" % (bn, add.get("error") or add.get("value")))
            continue
        rec["accessory"] = worn
        lv = T.rbx_get_props(worn + ".Handle." + V.split_path(wl_path)[-1], ["Order", "AutoSkin", "BindOffset",
                                                                                 "CageMeshContent"]).get("values") or {}
        rec["wrapLayer"] = lv
        for k, v in want.items():
            got = lv.get(k)
            if (k == "BindOffset" and (_cf(got) or [[None]])[0] != _cf(v)[0]) or (k != "BindOffset" and got != v):
                problems.append("%s: WrapLayer.%s reads %r, not %r" % (bn, k, got, v))
        rec["handleSize"] = (T.rbx_get_props(worn + ".Handle", ["Size"]).get("values") or {}).get("Size")
        time.sleep(2.0)  # the wrap is computed over the next frames
        cb = _capture_bounds(T, rig, worn, "%s_%s" % (aname, bn))
        rec["deformed"] = cb
        if not cb.get("ok"):
            problems.append("%s: %s" % (bn, cb.get("error")))
        out["bodies"][bn] = rec
    widths = {k: v["deformed"]["width"] for k, v in out["bodies"].items() if (v.get("deformed") or {}).get("ok")}
    out["deformedWidths"] = widths
    out.update(ok=not problems, verified=not problems)
    if problems:
        out["error"] = "; ".join(problems)
    return out


# ------------------------------------------------------------------------------------------------------------------- head
# The 17 base poses Roblox's FACS reference marks required for a head on the Marketplace
# (create.roblox.com/docs/art/characters/facial-animation/facs-poses-reference, read 2026-11-20).
REQUIRED_FACS = ("LeftEyeClosed", "RightEyeClosed", "EyesLookDown", "JawDrop", "Pucker", "LeftLipCornerPuller",
                 "RightLipCornerPuller", "ChinRaiser", "ChinRaiserUpperLip", "LeftCheekRaiser", "RightCheekRaiser",
                 "LeftInnerBrowRaiser", "RightInnerBrowRaiser", "LeftLipCornerDown", "RightLipCornerDown",
                 "LeftLowerLipDepressor", "RightLowerLipDepressor")
# The five expressions the spec names, as FaceControls values.
EXPRESSIONS = {"leftBlink": {"LeftEyeClosed": 1}, "rightBlink": {"RightEyeClosed": 1}, "mouthOpen": {"JawDrop": 1},
               "happy": {"LeftLipCornerPuller": 1, "RightLipCornerPuller": 1, "LeftCheekRaiser": 1, "RightCheekRaiser": 1},
               "sad": {"LeftLipCornerDown": 1, "RightLipCornerDown": 1, "LeftInnerBrowRaiser": 1, "RightInnerBrowRaiser": 1}}
FACE_MIN_PIXELS = 10    # an expression must move more pixels of the close-up than this, and than 3x the neutral's own noise
FACE_CAM = 2.5          # studs in front of the head the close-up camera stands
# MEASURED ON 0.748: Roblox's Blocky dynamic-head
# template uploaded through Open Cloud as a Model loads as a character - a Humanoid, 15 MeshParts with WrapTargets, a
# FaceControls under Head; the Head's own mesh opens as an EditableMesh and GetFacsPoses lists 50 poses (and
# GetFacsCorrectivePoses 63 combinations); GetFacsPose answers two lists, bone ids and CFrames. SetFacsPose(action, {}, {})
# on an EditableMesh removes that pose: GetFacsPoses no longer lists it.


def pose_motion(cframes):
    """The largest bone translation (studs) and rotation (degrees) in one FACS pose's CFrames. Pure."""
    import math
    move, turn = 0.0, 0.0
    for cf in cframes or []:
        c = _cf(cf)
        if not c:
            continue
        pos, r = c
        move = max(move, math.sqrt(sum(x * x for x in pos)))
        tr = max(-1.0, min(1.0, (r[0] + r[4] + r[8] - 1.0) / 2.0))
        turn = max(turn, math.degrees(math.acos(tr)))
    return round(move, 5), round(turn, 3)


def facs_report(present, required=REQUIRED_FACS):
    """Which required poses a head lacks, and which poses it has beyond them. Pure."""
    have = set(present or [])
    return {"missing": [p for p in required if p not in have], "extra": sorted(have - set(required)), "count": len(have)}


def rbx_head_check(head: Any, without: list = None, expressions: bool = True, at: list = None) -> dict:
    "Roblox Studio: is this a dynamic head Roblox would take? head is a Head MeshPart (or a character holding one) whose mesh is the person's own upload. Its mesh opens as an EditableMesh (`AssetService:CreateEditableMeshAsync`) and `EditableMesh:GetFacsPoses` is held against the 17 poses Roblox's FACS reference marks required: a missing one fails the check, named. without lists poses to clear first on that EditableMesh (`EditableMesh:SetFacsPose` with no bones), to see the check with a pose removed - nothing is written to the head. With expressions, a copy of the head's character is put at `at` and five expressions are set through its `FaceControls` (left and right blink, mouth open, happy, sad): each close-up capture is held against the neutral face and must move enough pixels, the box of the moved pixels reported; the copy is removed after. Roblox's own template head will not open (no permission to load asset, RB57): it must be the person's upload."
    T, V = _T(), _V()
    if without is not None and (not isinstance(without, list) or not all(isinstance(x, str) for x in without)):
        return _no("without is a list of FacsActionUnit names")
    pp, err = T._target(head, "head")
    if err:
        return err
    path = V.join_path(pp)
    g = T.rbx_get_props(path, ["ClassName"])
    cls = g.get("className") or (g.get("values") or {}).get("ClassName")
    if not g.get("ok"):
        return dict(g, ok=False, ran=False)
    if cls != "MeshPart":
        found = [m["path"] for m in (T.rbx_query(root=path, class_name="MeshPart", page_size=200).get("matches") or [])
                 if m["path"].rsplit(".", 1)[-1] == "Head"]
        if not found:
            return _no("%s is a %s holding no MeshPart named Head" % (path, cls), refusedRule="type")
        path = found[0]
    hv = T.rbx_get_props(path, ["MeshContent", "MeshId", "CFrame", "Size"]).get("values") or {}
    mesh = hv.get("MeshContent") if isinstance(hv.get("MeshContent"), str) and hv["MeshContent"] else hv.get("MeshId")
    if not (isinstance(mesh, str) and mesh.startswith("rbxassetid://")):
        return _no("%s's mesh is %r, not an uploaded asset: the FACS poses are read from the uploaded mesh" % (path, mesh),
                   refusedRule="needs-upload")
    em = T.rbx_call("AssetService", "CreateEditableMeshAsync", [mesh], timeout=60)
    h = (em.get("value") or {}).get("handle") if isinstance(em.get("value"), dict) else None
    if not h:
        return {"ok": False, "ran": True, "head": path, "mesh": mesh, "step": "CreateEditableMeshAsync",
                "error": "the head's mesh did not open as an EditableMesh: %s" % (em.get("error") or em.get("value"))}
    out = {"head": path, "mesh": mesh}
    problems = []
    try:
        cleared = []
        for a in without or []:
            r = T.rbx_call({"handle": h}, "SetFacsPose", [a, [], []])
            if not r.get("ok"):
                return dict(out, ok=False, ran=True, step="without", error="%s could not be cleared: %s" % (a, r.get("error")))
            cleared.append(a)
        poses = T.rbx_call({"handle": h}, "GetFacsPoses")
        if not poses.get("ok") or not isinstance(poses.get("value"), list):
            return dict(out, ok=False, ran=True, step="GetFacsPoses", error=poses.get("error") or poses.get("value"))
        # how far each expression's poses move the face's bones (GetFacsPose answers bone ids and CFrames)
        motion = {}
        for a in sorted({k for v in EXPRESSIONS.values() for k in v} & set(poses["value"])):
            fp = T.rbx_call({"handle": h}, "GetFacsPose", [a])
            vals = fp.get("values") if isinstance(fp.get("values"), list) else []
            mv, tn = pose_motion(vals[1] if len(vals) > 1 else [])
            motion[a] = {"bones": len(vals[0]) if vals else 0, "maxMove": mv, "maxTurnDegrees": tn}
        out["poseMotion"] = motion
    finally:
        T.rbx_release_handle(h)
    rep = facs_report(poses["value"])
    out.update(poses=sorted(poses["value"]), facs=rep, cleared=cleared, required=list(REQUIRED_FACS))
    if rep["missing"]:
        problems.append("missing required FACS poses: %s" % ", ".join(rep["missing"]))
    if expressions:
        out["expressions"] = _expressions(T, V, path, at)
        if not out["expressions"].get("ok"):
            problems.append("expressions: %s" % out["expressions"].get("error"))
    out.update(ok=not problems, verified=not problems, passes=not rep["missing"])
    if problems:
        out["error"] = "; ".join(problems)
    return out


def _expressions(T, V, head_path, at):
    """Five expressions on a copy of the head's character, each close-up held against the neutral one."""
    from mif_roblox import png as P
    hp = V.split_path(head_path)
    char = V.join_path(hp[:-1])
    if not (T.rbx_query(root=char, class_name="FaceControls", page_size=10).get("matches") or []):
        return {"ok": False, "error": "%s holds no FaceControls: the expressions need one" % head_path}
    dst = V.split_path(SCRATCH)
    c = T.rbx_clone(char, parent=SCRATCH, name=T._unique_name(dst, "MifHeadCheck"))
    if not c.get("ok"):
        return {"ok": False, "error": "the character did not copy: %s" % c.get("error")}
    copy_ = c["path"]
    out = {"copy": copy_, "each": {}}
    try:
        pv = T.rbx_call(copy_, "PivotTo", [{"position": list(at or [2600, 200, 2000])}])
        if not pv.get("ok"):
            return dict(out, ok=False, error="PivotTo: %s" % pv.get("error"))
        head = copy_ + "." + hp[-1]
        fc = (T.rbx_query(root=head, class_name="FaceControls", page_size=5).get("matches") or [{}])[0].get("path")
        if not fc:
            return dict(out, ok=False, error="the copy's head holds no FaceControls")
        cf = _cf((T.rbx_get_props(head, ["CFrame"]).get("values") or {}).get("CFrame"))
        if not cf:
            return dict(out, ok=False, error="the copy's head CFrame did not read")
        pos, rot = cf
        look = [-rot[2], -rot[5], -rot[8]]   # a part looks along -Z: the third column, negated
        cam = {"position": [pos[i] + look[i] * FACE_CAM for i in range(3)], "look_at": pos}
        names = sorted({k for v in EXPRESSIONS.values() for k in v})
        T.rbx_set_props(fc, {k: 0 for k in names})
        time.sleep(0.5)
        neutral = T.rbx_screenshot(camera=cam, max_width=480, name="head_neutral")
        again = T.rbx_screenshot(camera=cam, max_width=480, name="head_neutral2")
        if not neutral.get("ok") or not again.get("ok"):
            return dict(out, ok=False, error="neutral capture: %s" % (neutral.get("error") or again.get("error")))
        w, hgt, base = P.decode_rgba(open(neutral["file"], "rb").read())
        _w, _h, base2 = P.decode_rgba(open(again["file"], "rb").read())
        noise = silhouette(base2, base, w, hgt, level=24)[1] if (_w, _h) == (w, hgt) else 0
        need = max(FACE_MIN_PIXELS, 3 * noise)
        out.update(noisePixels=noise, needPixels=need)
        problems = []
        for ex, vals in EXPRESSIONS.items():
            sp = T.rbx_set_props(fc, vals)
            rb = T.rbx_get_props(fc, sorted(vals)).get("values") or {}
            time.sleep(0.5)
            shot = T.rbx_screenshot(camera=cam, max_width=480, name="head_" + ex)
            T.rbx_set_props(fc, {k: 0 for k in vals})
            rec = {"values": vals, "readBack": rb, "file": shot.get("file")}
            if not sp.get("ok") or any(abs((rb.get(k) or 0) - v) > 1e-3 for k, v in vals.items()):
                problems.append("%s: FaceControls read %s after setting %s" % (ex, rb, vals))
            if not shot.get("ok"):
                problems.append("%s: capture: %s" % (ex, shot.get("error")))
            else:
                w2, h2, img = P.decode_rgba(open(shot["file"], "rb").read())
                box, n = silhouette(img, base, w, hgt, level=24) if (w2, h2) == (w, hgt) else (None, 0)
                rec.update(movedPixels=n, box=list(box) if box else None)
                if n <= need:
                    problems.append("%s moved %d pixels of the close-up (needs more than %d): the face did not move" % (
                        ex, n, need))
            out["each"][ex] = rec
        out.update(ok=not problems, neutral=neutral.get("file"))
        if problems:
            out["error"] = "; ".join(problems)
        return out
    finally:
        T.rbx_destroy(copy_)
        out["copyRemoved"] = not T.rbx_get_props(copy_, ["Name"]).get("ok")
