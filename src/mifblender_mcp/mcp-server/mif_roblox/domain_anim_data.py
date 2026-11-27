"""Roblox lane 5: animation and rigs (batch 10), audio (11), data and gameplay services (13), editor, packages and
collaboration (14). the internal notes sections 5.7, 5.8, 5.11, 5.13, 5.15-5.17 are the plan; the internal notes's lane 5 section is what
was built.

Every tool here is a composite over the generic layer in tools.py (rbx_call, rbx_get_props, rbx_set_props, rbx_create,
rbx_batch, rbx_query, the undo group): no plugin op of its own, so the plugin file is untouched. Every write is read back
through a member that proves it. Imported by one line at the foot of tools.py, which collects every rbx_* name.

Rules this module adds, each said in the tool's answer when it applies:
- DataStore writes (decision 5, Mifsopo 2026-11-18): refused unless the panel is in full mode AND the call passes
  allow_live_write: true. A Studio write reaches the live game's data. Reads are free.
- Studio preferences (decision 7, 2026-11-20: yes for the snapping): rbx_dragger_settings sets the snap in full mode and
  keeps the person's values to restore; every other preference stays read only (studio-preference).
- Roblox's built-in Studio MCP server: named inside a refusal for a wall it covers (setup offers it: decision 9, yes).
  rbx_place_save is the one tool that calls it (save_place, opt-in per call: use_builtin_mcp true).
- Nothing is ever played aloud (his machine): a Sound's Playing and PlayOnRemove are refused; playback is read from
  properties, never heard.

NEVER RAISE: every failure is {"ok": False, ...}.
"""
import json
import math
import os
import re
import struct
import tempfile
import time
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Any

from mif_roblox import tools as T  # noqa: E402 - tools.py imports this module at its foot, after its own names exist
from mif_roblox import values as V  # noqa: E402

# Members read by name that tools.READ_PREFIXES does not cover yet. List* and AdvanceToNextPage* only read on 0.748
# (every reachable one: DataStoreService:ListDataStoresAsync, DataStore:ListKeysAsync, DataStore:ListVersionsAsync,
# MemoryStoreHashMap:ListItemsAsync, ContentProvider:ListEncryptedAssets, TestService:ListTests, Pages:AdvanceToNextPageAsync).
# Without them a data store read would be judged a write, recorded, and refused in scratch and read modes.
_EXTRA_READ = ("List", "AdvanceToNextPage")
if not set(_EXTRA_READ) <= set(T.READ_PREFIXES):
    T.READ_PREFIXES = tuple(T.READ_PREFIXES) + tuple(p for p in _EXTRA_READ if p not in T.READ_PREFIXES)
# Registering a clip gives a temporary content id and changes nothing in the place.
T.NON_PLACE_CALLS |= {("KeyframeSequenceProvider", "RegisterKeyframeSequence"),
                      ("AnimationClipProvider", "RegisterAnimationClip")}

SCRATCH = "Workspace.MifScratch"
BATCH_MAX = 100   # tools.rbx_batch's limit
BUILTIN_MCP = ("Roblox's built-in Studio MCP server (Studio: Assistant > Manage MCP Servers, docs "
               "https://create.roblox.com/docs/studio/mcp) covers this; MifBridge does not drive it")


def _no(error, **kw):
    return dict({"ok": False, "ran": False, "error": error}, **kw)


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _path(x):
    """A path as a dot string (or the list form when a name has a dot)."""
    parts = V.split_path(x)
    return V.join_path(parts) if parts is not None else None


def _child(parent, name):
    p = V.split_path(parent)
    return V.join_path(p + [name]) if "." not in name else p + [name]


def _class_of(target):
    parts, err = T._target(target) if not isinstance(target, dict) else (target, None)
    if err:
        return None, err
    return T._resolve(parts)


def _exists(target):
    parts = V.split_path(target) if not isinstance(target, dict) else target
    return parts is not None and T._rbx("resolve", target=parts).get("ok") is True


def _props(target, names):
    """{name: decoded} or raises nothing: (values, error)."""
    r = T.rbx_get_props(target, list(names))
    if not r.get("ok"):
        return None, r
    return r.get("values") or {}, None


def _cf_list(cf):
    """A decoded or caller CFrame -> 12 numbers, or None."""
    if isinstance(cf, dict) and "position" in cf:
        rot = cf.get("rotation", [1, 0, 0, 0, 1, 0, 0, 0, 1])
        return list(cf["position"]) + list(rot)
    if isinstance(cf, list) and len(cf) == 12:
        return list(cf)
    if isinstance(cf, list) and len(cf) == 3:
        return list(cf) + [1, 0, 0, 0, 1, 0, 0, 0, 1]
    return None


IDENTITY = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]


def _cf_diff(a, b):
    """(position distance in studs, largest rotation-matrix element difference) between two CFrames, or None."""
    a, b = _cf_list(a), _cf_list(b)
    if a is None or b is None:
        return None
    return math.dist(a[:3], b[:3]), max(abs(x - y) for x, y in zip(a[3:], b[3:]))


def _cf_close(a, b, pos_tol=1e-3, rot_tol=1e-3):
    d = _cf_diff(a, b)
    return d is not None and d[0] <= pos_tol and d[1] <= rot_tol


@T.ends_group_on_raise
def _build(ops, label):
    """Several creates and sets as one undo step: one rbx_batch when it fits, else an undo group of single calls.
    {ok, undoSteps, results: [per op answer]} - each op's own read-back judged by the tool that ran it."""
    if len(ops) <= BATCH_MAX:
        r = T.rbx_batch(ops)
        return dict(r, undoSteps=1 if r.get("ok") else 0, route="batch")
    g = T.begin_group("MifBridge " + label)
    if not g.get("ok"):
        return dict(g, ok=False, route="group")
    results = []
    for i, op in enumerate(ops):
        fn = getattr(T, op["tool"])
        res = fn(**op["args"])
        results.append(res)
        if not res.get("ok"):
            T.rbx_undo_group("cancel")
            return {"ok": False, "ran": True, "failedAt": i, "route": "group", "cancelled": True, "undoSteps": 0,
                    "error": "op %d (%s) failed, the group was cancelled: %s" % (i, op["tool"], res.get("error"))}
    c = T.rbx_undo_group("commit")
    return {"ok": bool(c.get("ok")), "verified": bool(c.get("ok")), "route": "group", "undoSteps": 1, "ops": len(ops),
            "results": [{"path": x.get("path")} for x in results], "error": None if c.get("ok") else c.get("error")}


def _free_name(parent, base, replace=False):
    """(name, error): base when free; with replace, base after destroying what holds it; else base_2, base_3..."""
    path = _child(parent, base)
    if not _exists(path):
        return base, None
    if replace:
        d = T.rbx_destroy(path)
        return (base, None) if d.get("ok") else (None, d)
    return T._unique_name(V.split_path(parent), base), None


def _release(*handles):
    for h in handles:
        if isinstance(h, dict):
            h = h.get("handle")
        if isinstance(h, str):
            T._rbx("release_handle", handle=h)


def _handle(value):
    return value.get("handle") if isinstance(value, dict) and isinstance(value.get("handle"), str) else None


# ================================================================ batch 10: animation and rigs
# The Motor6D names Players:CreateHumanoidModelFromDescriptionAsync puts in a rig. What the tool proves is the set it
# finds; these are what it compares that set with, and the internal notes records the live answer.
RIG_JOINTS = {
    "R15": ("Root", "Waist", "Neck", "LeftShoulder", "LeftElbow", "LeftWrist", "RightShoulder", "RightElbow",
            "RightWrist", "LeftHip", "LeftKnee", "LeftAnkle", "RightHip", "RightKnee", "RightAnkle"),
    "R6": ("RootJoint", "Neck", "Left Shoulder", "Right Shoulder", "Left Hip", "Right Hip"),
}


def rig_joints(rig):
    """The rig's joints: ({part1Name: {joint, path, part0, part1, kind}}, root part name, error). A joint is a Motor6D or,
    on rigs Players:CreateHumanoidModelFromDescriptionAsync makes on 0.748, an AnimationConstraint: both
    carry Transform, which is what an animation drives. AnimationConstraint.Part0/Part1 are Hidden and Deprecated but
    readable; its Attachment0/Attachment1's parents are read when they are not."""
    joints = {}
    for kind in ("Motor6D", "AnimationConstraint"):
        q = T.rbx_query(root=rig, class_name=kind, page_size=500)
        if not q.get("ok"):
            return None, None, q
        for m in q.get("matches") or []:
            vals, err = _props(m["path"], ["Part0", "Part1"] if kind == "Motor6D" else ["Attachment0", "Attachment1"])
            if err:
                return None, None, err
            if kind == "Motor6D":
                p0, p1 = vals.get("Part0"), vals.get("Part1")
            else:
                a0, a1 = vals.get("Attachment0"), vals.get("Attachment1")
                p0 = V.join_path(V.split_path(a0)[:-1]) if a0 else None
                p1 = V.join_path(V.split_path(a1)[:-1]) if a1 else None
            if not p1:
                continue
            name1 = V.split_path(p1)[-1]
            joints[name1] = {"joint": V.split_path(m["path"])[-1], "path": m["path"], "kind": kind,
                             "part0": V.split_path(p0)[-1] if p0 else None, "part1": name1}
    parents = {j["part0"] for j in joints.values()} - set(joints)
    root = sorted(x for x in parents if x)[0] if parents - {None} else None
    return joints, root, None


@T._reports
def rbx_rig_create(rig_type: str = "R15", name: str = "MifRig", parent: Any = SCRATCH, scales: dict = None,
                   position: list = None, build_from_attachments: bool = False) -> dict:
    "Roblox Studio: an R15 or R6 character rig from a HumanoidDescription (Players:CreateHumanoidModelFromDescriptionAsync, through rbx_mannequin), placed under parent and optionally pivoted to position; with build_from_attachments, Humanoid:BuildRigFromAttachments rebuilds its joints. Read back: Humanoid.RigType, and every joint (Motor6D, or AnimationConstraint on the rigs 0.748 makes) by name with the parts it joins, compared with the joint set the rig type should have."
    if rig_type not in RIG_JOINTS:
        return _no("rig_type is R15 or R6")
    mq = T.rbx_mannequin(name=name, parent=parent, rig_type=rig_type, scales=scales, position=position)
    if not mq.get("ok"):
        return dict(mq, step="mannequin")
    rig = mq["path"]
    hum = (T.rbx_find(class_name="Humanoid", root=rig, limit=1).get("matches") or [{}])[0].get("path")
    if not hum:
        return {"ok": False, "ran": True, "path": rig, "error": "the rig has no Humanoid"}
    out = {"path": rig, "humanoid": hum}
    if build_from_attachments:
        b = T.rbx_call(hum, "BuildRigFromAttachments")
        out["buildFromAttachments"] = {"ok": b.get("ok"), "error": b.get("error")}
        if not b.get("ok"):
            return dict(out, ok=False, ran=True, error="BuildRigFromAttachments failed: %s" % b.get("error"))
    vals, err = _props(hum, ["RigType"])
    if err:
        return dict(err, path=rig)
    joints, root, err = rig_joints(rig)
    if err:
        return dict(err, path=rig)
    names = sorted(j["joint"] for j in joints.values())
    want = sorted(RIG_JOINTS[rig_type])
    problems = []
    if vals.get("RigType") != rig_type:
        problems.append("Humanoid.RigType reads %s" % vals.get("RigType"))
    missing, extra = sorted(set(want) - set(names)), sorted(set(names) - set(want))
    if missing:
        problems.append("joints missing: %s" % ", ".join(missing))
    out.update(ok=not problems, verified=not problems, rigType=vals.get("RigType"), jointCount=len(names),
               joints={j["joint"]: {"part0": j["part0"], "part1": j["part1"], "path": j["path"], "kind": j["kind"]}
                       for j in joints.values()}, rootPart=root, extraJoints=extra)
    if problems:
        out["error"] = "; ".join(problems)
    return out


EASING_STYLES = ("Linear", "Constant", "Elastic", "Cubic", "Bounce", "CubicV2")
EASING_DIRECTIONS = ("In", "Out", "InOut")


def _pose_spec(name, spec):
    """A caller's pose ({cframe, easing_style, easing_direction, weight, children} or a bare CFrame) -> (dict, error)."""
    if not isinstance(spec, dict) or "position" in spec or "cframe" not in spec and "children" not in spec:
        spec = {"cframe": spec}
    extra = set(spec) - {"cframe", "easing_style", "easing_direction", "weight", "children"}
    if extra:
        return None, "pose %s: unknown keys %s" % (name, sorted(extra))
    cf = _cf_list(spec.get("cframe", IDENTITY))
    if cf is None or not all(_num(x) for x in cf):
        return None, "pose %s: cframe is 12 numbers, [x,y,z], or {position, rotation}" % name
    style, direction = spec.get("easing_style", "Linear"), spec.get("easing_direction", "In")
    if style not in EASING_STYLES or direction not in EASING_DIRECTIONS:
        return None, "pose %s: easing_style is one of %s, easing_direction one of %s" % (name, EASING_STYLES, EASING_DIRECTIONS)
    w = spec.get("weight", 1)
    if not _num(w) or not 0 <= w <= 1:
        return None, "pose %s: weight is 0-1" % name
    kids = {}
    for k, v in (spec.get("children") or {}).items():
        kids[k], err = _pose_spec(k, v)
        if err:
            return None, err
    return {"cframe": cf, "easing_style": style, "easing_direction": direction, "weight": w, "children": kids}, None


def _nest_by_rig(flat, joints, root):
    """Flat {part: pose} -> nested under the rig's hierarchy from its root part; parts on the way get a weight-0
    identity pose (they hold the nesting, they do not move the joint). (tree, error)."""
    unknown = sorted(p for p in flat if p not in joints and p != root)
    if unknown:
        return None, unknown
    parent_of = {p: j["part0"] for p, j in joints.items()}
    tree = {}

    def node_for(part):
        chain = [part]
        while chain[-1] != root and chain[-1] in parent_of:
            chain.append(parent_of[chain[-1]])
        chain.reverse()
        level, nd = tree, None
        for p in chain:
            if p not in level:
                level[p] = {"cframe": list(IDENTITY), "easing_style": "Linear", "easing_direction": "In", "weight": 0,
                            "children": {}}
            nd = level[p]
            level = nd["children"]
        return nd

    for part, pose in flat.items():
        nd = node_for(part)
        nd.update({k: v for k, v in pose.items() if k != "children"})
    return tree, None


def _seq_ops(kind, name, parent, loop, priority, keyframes):
    """The create ops for a KeyframeSequence tree, parents before children."""
    root = _child(parent, name)
    ops = [{"tool": "rbx_create", "args": {"class_name": kind, "parent": parent, "name": name,
                                           "props": {"Loop": bool(loop), "Priority": priority}}}]

    def poses(parent_path, tree):
        for pname, p in tree.items():
            ops.append({"tool": "rbx_create", "args": {"class_name": "Pose", "parent": parent_path, "name": pname, "props": {
                "CFrame": p["cframe"], "EasingStyle": p["easing_style"], "EasingDirection": p["easing_direction"],
                "Weight": p["weight"]}}})
            poses(_child(parent_path, pname), p["children"])

    for kf in keyframes:
        kpath = _child(root, kf["name"])
        ops.append({"tool": "rbx_create", "args": {"class_name": "Keyframe", "parent": root, "name": kf["name"],
                                                   "props": {"Time": kf["time"]}}})
        poses(kpath, kf["poses"])
        for mname, mval in kf["markers"].items():
            ops.append({"tool": "rbx_create", "args": {"class_name": "KeyframeMarker", "parent": kpath, "name": mname,
                                                       "props": {"Value": mval}}})
    return root, ops


def read_sequence(target):
    """A KeyframeSequence read from Studio: {loop, priority, keyframes: [{name, time, poses (nested), markers}]},
    through KeyframeSequence:GetKeyframes for the keyframe list, the tree for the nesting, and each node's properties."""
    vals, err = _props(target, ["Loop", "Priority", "Name"])
    if err:
        return None, err
    kfs = T.rbx_call(target, "GetKeyframes")
    if not kfs.get("ok"):
        return None, kfs
    listed = sorted(V.split_path(p)[-1] for p in (kfs.get("value") or []) if p)
    tree = T.rbx_tree(root=target, depth=20, limit=10000)
    if not tree.get("ok"):
        return None, tree
    base = V.split_path(target)
    seq = {"name": vals.get("Name"), "loop": vals.get("Loop"), "priority": vals.get("Priority"), "keyframes": [],
           "getKeyframes": listed}

    def read_poses(path, node):
        out = {}
        for c in node.get("children") or []:
            if c.get("className") != "Pose":
                continue
            cp = path + [c["name"]]
            pv, perr = _props(cp, ["CFrame", "Weight", "EasingStyle", "EasingDirection"])
            if perr:
                raise RuntimeError(perr.get("error"))
            out[c["name"]] = {"cframe": _cf_list(pv.get("CFrame")), "weight": pv.get("Weight"),
                              "easing_style": pv.get("EasingStyle"), "easing_direction": pv.get("EasingDirection"),
                              "children": read_poses(cp, c)}
        return out

    try:
        for kn in (tree.get("root") or {}).get("children") or []:
            if kn.get("className") != "Keyframe":
                continue
            kp = base + [kn["name"]]
            kv, kerr = _props(kp, ["Time"])
            if kerr:
                return None, kerr
            markers = {}
            for c in kn.get("children") or []:
                if c.get("className") == "KeyframeMarker":
                    mv, merr = _props(kp + [c["name"]], ["Value"])
                    if merr:
                        return None, merr
                    markers[c["name"]] = mv.get("Value")
            seq["keyframes"].append({"name": kn["name"], "time": kv.get("Time"), "poses": read_poses(kp, kn),
                                     "markers": markers})
    except RuntimeError as exc:
        return None, {"ok": False, "error": "reading a pose failed: %s" % exc}
    seq["keyframes"].sort(key=lambda k: (k["time"], k["name"]))
    return seq, None


def _compare_poses(sent, back, where, problems):
    for name, p in sent.items():
        b = back.get(name)
        if b is None:
            problems.append("%s.%s is missing" % (where, name))
            continue
        if not _cf_close(p["cframe"], b["cframe"], 1e-4, 1e-4):
            problems.append("%s.%s CFrame reads %s, sent %s" % (where, name, b["cframe"], p["cframe"]))
        if not _num(b.get("weight")) or abs(b["weight"] - p["weight"]) > 1e-4:
            problems.append("%s.%s Weight reads %s" % (where, name, b.get("weight")))
        if b.get("easing_style") != p["easing_style"] or b.get("easing_direction") != p["easing_direction"]:
            problems.append("%s.%s easing reads %s %s" % (where, name, b.get("easing_style"), b.get("easing_direction")))
        _compare_poses(p["children"], b.get("children") or {}, where + "." + name, problems)


@T._reports
def rbx_keyframes_set(keyframes: list, name: str = "MifAnimation", parent: Any = SCRATCH, rig: Any = None,
                      loop: bool = False, priority: str = "Action", replace: bool = False) -> dict:
    "Roblox Studio: a KeyframeSequence built from keys, as one undo step. keyframes is a list of {time, name?, poses, markers?}; poses maps a part name to a CFrame (12 numbers, [x,y,z], or {position, rotation}) or {cframe, easing_style, easing_direction, weight, children}. With rig (a character model), poses may be flat: each is checked to be a joint of the rig (a Motor6D's Part1) and nested under its parent part's pose as Studio's Animation Editor nests them; a pose name that is not a joint is refused before anything is made. Read back key for key: KeyframeSequence:GetKeyframes, every Keyframe.Time, every Pose's CFrame, Weight and easing, every KeyframeMarker.Value, Loop and Priority."
    if not isinstance(keyframes, list) or not keyframes:
        return _no("keyframes is a non-empty list of {time, poses, name?, markers?}")
    if priority not in ("Idle", "Movement", "Action", "Action2", "Action3", "Action4", "Core"):
        return _no("priority is an AnimationPriority item: Idle, Movement, Action, Action2-4, Core")
    if not isinstance(name, str) or not name or "." in name:
        return _no("name is a non-empty string without dots")
    joints = root = None
    if rig is not None:
        joints, root, err = rig_joints(rig)
        if err:
            return err
        if not joints:
            return _no("%s has no Motor6D joints to animate" % _path(rig), refusedRule="not-a-rig")
    clean, seen = [], set()
    for i, kf in enumerate(keyframes):
        if not isinstance(kf, dict) or not _num(kf.get("time")) or kf["time"] < 0 or not isinstance(kf.get("poses"), dict):
            return _no("keyframe %d is {time >= 0, poses: {...}}" % i)
        if set(kf) - {"time", "name", "poses", "markers"}:
            return _no("keyframe %d: unknown keys %s" % (i, sorted(set(kf) - {"time", "name", "poses", "markers"})))
        kname = kf.get("name") or "Keyframe_%d" % i
        if not isinstance(kname, str) or "." in kname or kname in seen:
            return _no("keyframe %d: names are unique strings without dots (%r)" % (i, kname))
        seen.add(kname)
        poses = {}
        for pn, ps in kf["poses"].items():
            poses[pn], err = _pose_spec(pn, ps)
            if err:
                return _no("keyframe %d: %s" % (i, err))
        if joints is not None:
            flat = {k: dict(v, children={}) for k, v in poses.items()}
            if any(v["children"] for v in poses.values()):
                return _no("keyframe %d: with rig, give poses flat (part: pose); the rig decides the nesting" % i)
            poses, unknown = _nest_by_rig(flat, joints, root)
            if unknown:
                return _no("keyframe %d: %s is not a joint of %s" % (i, ", ".join(unknown), _path(rig)),
                           refusedRule="not-a-joint", joints=sorted(joints), rootPart=root)
        markers = kf.get("markers") or {}
        if not isinstance(markers, dict) or not all(isinstance(k, str) and "." not in k and isinstance(v, str)
                                                    for k, v in markers.items()):
            return _no("keyframe %d: markers maps a name (no dots) to a string value" % i)
        clean.append({"name": kname, "time": kf["time"], "poses": poses, "markers": markers})
    pparts, err = T._target(parent, "parent")
    if err:
        return err
    sname, err = _free_name(parent, name, replace)
    if err:
        return err
    root_path, ops = _seq_ops("KeyframeSequence", sname, parent, loop, priority, clean)
    built = _build(ops, "keyframes " + sname)
    if not built.get("ok"):
        if built.get("route") == "batch" and built.get("restored") is False:
            return dict(built, ok=False)
        return dict(built, ok=False, path=None)
    back, err = read_sequence(root_path)
    if err:
        return dict(err, ok=False, ran=True, path=_path(root_path), error="built, but reading it back failed: %s" % err.get("error"))
    problems = []
    if back["loop"] is not bool(loop) or back["priority"] != priority:
        problems.append("Loop/Priority read %s/%s" % (back["loop"], back["priority"]))
    if back["getKeyframes"] != sorted(k["name"] for k in clean):
        problems.append("GetKeyframes lists %s" % back["getKeyframes"])
    by_name = {k["name"]: k for k in back["keyframes"]}
    for kf in clean:
        b = by_name.get(kf["name"])
        if b is None:
            problems.append("keyframe %s missing" % kf["name"])
            continue
        if not _num(b["time"]) or abs(b["time"] - kf["time"]) > 1e-4:
            problems.append("%s.Time reads %s, sent %s" % (kf["name"], b["time"], kf["time"]))
        _compare_poses(kf["poses"], b["poses"], kf["name"], problems)
        if b["markers"] != kf["markers"]:
            problems.append("%s markers read %s" % (kf["name"], b["markers"]))
    out = {"ok": not problems, "verified": not problems, "path": _path(root_path), "keyframes": len(clean),
           "instances": len(ops), "undoSteps": built.get("undoSteps"), "length": max(k["time"] for k in clean),
           "readBack": back}
    if problems:
        out["error"] = "; ".join(problems[:20])
        out["problems"] = problems
    return out


CURVE_KINDS = ("rotation", "position", "float")


def _curve_key(kind, k, i):
    if not isinstance(k, dict) or not _num(k.get("time")) or k["time"] < 0:
        return None, "key %d is {time >= 0, value, interpolation?}" % i
    interp = k.get("interpolation", "Linear")
    if interp not in ("Constant", "Linear", "Cubic"):
        return None, "key %d: interpolation is Constant, Linear or Cubic" % i
    v = k.get("value")
    if kind == "rotation":
        cf = _cf_list(v) if not (isinstance(v, list) and len(v) == 9) else [0, 0, 0] + list(v)
        if cf is None:
            return None, "key %d: a rotation value is a CFrame or 9 rotation numbers" % i
        v = cf
    elif kind == "position":
        if not (isinstance(v, list) and len(v) == 3 and all(_num(x) for x in v)):
            return None, "key %d: a position value is [x, y, z]" % i
    elif not _num(v):
        return None, "key %d: a float value is a number" % i
    return {"time": k["time"], "value": v, "interpolation": interp}, None


@T._reports
@T.ends_group_on_raise
def rbx_curve_animation_set(channels: list, name: str = "MifCurveAnimation", parent: Any = SCRATCH, loop: bool = False,
                            priority: str = "Action", replace: bool = False) -> dict:
    "Roblox Studio: a CurveAnimation with per-joint curves, as one undo step. channels is a list of {joint, kind, keys, name?}: joint is the folder path inside the animation, '/'-separated in rig order (e.g. 'LowerTorso/UpperTorso'); kind rotation (a RotationCurve, default name Rotation, values CFrames), position (a Vector3Curve, default name Position, with FloatCurves X, Y, Z; values [x,y,z]) or float (a FloatCurve, values numbers); keys [{time, value, interpolation: Constant|Linear|Cubic}]. Keys go in through FloatCurve:SetKeys, and RotationCurve:InsertKey one by one (its SetKeys fails from a plugin on 0.748). Read back: every curve's GetKeys and its GetValueAtTime at each key time equal to the key."
    if not isinstance(channels, list) or not channels:
        return _no("channels is a non-empty list of {joint, kind, keys}")
    plan, seen = [], set()
    for i, ch in enumerate(channels):
        if not isinstance(ch, dict) or ch.get("kind") not in CURVE_KINDS or not isinstance(ch.get("joint"), str):
            return _no("channel %d is {joint: 'A/B', kind: rotation|position|float, keys: [...]}" % i)
        folders = [f for f in ch["joint"].split("/") if f]
        if not folders or any("." in f for f in folders):
            return _no("channel %d: joint is a '/'-path of folder names without dots" % i)
        cname = ch.get("name") or {"rotation": "Rotation", "position": "Position"}.get(ch["kind"])
        if not cname or "." in cname:
            return _no("channel %d: a float channel needs a name (no dots)" % i)
        if (tuple(folders), cname) in seen:
            return _no("channel %d: %s/%s is given twice" % (i, ch["joint"], cname))
        seen.add((tuple(folders), cname))
        keys = []
        if not isinstance(ch.get("keys"), list) or not ch["keys"]:
            return _no("channel %d: keys is a non-empty list" % i)
        for j, k in enumerate(ch["keys"]):
            ck, err = _curve_key(ch["kind"], k, j)
            if err:
                return _no("channel %d: %s" % (i, err))
            keys.append(ck)
        times = [k["time"] for k in keys]
        if times != sorted(times) or len(set(times)) != len(times):
            return _no("channel %d: key times rise strictly" % i)
        plan.append({"folders": folders, "kind": ch["kind"], "name": cname, "keys": keys})
    sname, err = _free_name(parent, name, replace)
    if err:
        return err
    root = _child(parent, sname)
    ops = [{"tool": "rbx_create", "args": {"class_name": "CurveAnimation", "parent": parent, "name": sname,
                                           "props": {"Loop": bool(loop), "Priority": priority}}}]
    made_folders, curves = set(), []
    for ch in plan:
        cur = root
        for f in ch["folders"]:
            nxt = _child(cur, f)
            if str(nxt) not in made_folders:
                made_folders.add(str(nxt))
                ops.append({"tool": "rbx_create", "args": {"class_name": "Folder", "parent": cur, "name": f}})
            cur = nxt
        cpath = _child(cur, ch["name"])
        cls = {"rotation": "RotationCurve", "position": "Vector3Curve", "float": "FloatCurve"}[ch["kind"]]
        ops.append({"tool": "rbx_create", "args": {"class_name": cls, "parent": cur, "name": ch["name"]}})
        if ch["kind"] == "position":
            for ax_i, ax in enumerate("XYZ"):
                ops.append({"tool": "rbx_create", "args": {"class_name": "FloatCurve", "parent": cpath, "name": ax}})
                curves.append((_child(cpath, ax), "FloatCurveKey",
                               [dict(k, value=k["value"][ax_i]) for k in ch["keys"]], ch, ax))
        else:
            curves.append((cpath, "RotationCurveKey" if ch["kind"] == "rotation" else "FloatCurveKey", ch["keys"], ch, None))
    g = T.begin_group("MifBridge curve animation " + sname)
    if not g.get("ok"):
        return dict(g, ok=False)
    for i, op in enumerate(ops):
        res = T.rbx_create(**op["args"])
        if not res.get("ok"):
            T.rbx_undo_group("cancel")
            return dict(res, ok=False, ran=True, failedAt=i, error="creating %s failed, group cancelled: %s"
                        % (op["args"].get("name"), res.get("error")))
    for cpath, ktype, keys, _ch, _ax in curves:
        wire = [{"$t": ktype, **({"time": k["time"], "value": k["value"] if ktype == "FloatCurveKey" else
                                 {"position": k["value"][:3], "rotation": k["value"][3:]}, "interpolation": k["interpolation"]})}
                for k in keys]
        if ktype == "RotationCurveKey":
            # RotationCurve:SetKeys raises "Variant cast failed" from a plugin on 0.748 while InsertKey with the same
            # key works: rotation keys go in one at a time.
            for w in wire:
                r = T.rbx_call(cpath, "InsertKey", [{k: v for k, v in w.items() if k != "$t"}])
                if not r.get("ok"):
                    break
            r = dict(r, value=len(keys)) if r.get("ok") else r
        else:
            r = T.rbx_call(cpath, "SetKeys", [wire])
        if not r.get("ok") or r.get("value") != len(keys):
            T.rbx_undo_group("cancel")
            return {"ok": False, "ran": True, "curve": _path(cpath), "error": "SetKeys on %s answered %s: %s" % (
                _path(cpath), r.get("value"), r.get("error")), "cancelled": True}
    c = T.rbx_undo_group("commit")
    if not c.get("ok"):
        return dict(c, ok=False, ran=True)
    problems, readback = [], {}
    for cpath, ktype, keys, ch, ax in curves:
        got = T.rbx_call(cpath, "GetKeys")
        back = got.get("value") or []
        row = {"keys": len(back), "atTimes": []}
        if not got.get("ok") or len(back) != len(keys):
            problems.append("%s GetKeys lists %s keys, sent %d" % (_path(cpath), len(back), len(keys)))
        for k in keys:
            v = T.rbx_call(cpath, "GetValueAtTime", [k["time"]])
            val = v.get("value")
            row["atTimes"].append({"time": k["time"], "value": val})
            if ktype == "FloatCurveKey":
                good = _num(val) and abs(val - k["value"]) <= 1e-4 * max(1, abs(k["value"]))
            else:
                d = _cf_diff(val, k["value"]) if val is not None else None
                good = d is not None and d[1] <= 1e-4
            if not good:
                problems.append("%s at %s reads %s, key %s" % (_path(cpath), k["time"], val, k["value"]))
        readback[_path(cpath)] = row
    out = {"ok": not problems, "verified": not problems, "path": _path(root), "curves": len(curves),
           "instances": len(ops), "undoSteps": 1, "readBack": readback}
    if problems:
        out.update(error="; ".join(problems[:20]), problems=problems)
    return out


def _animator_for(rig):
    """(animator path, created by us, error): the rig's Animator, or one made under its Humanoid/AnimationController."""
    a = (T.rbx_find(class_name="Animator", root=rig, limit=1).get("matches") or [{}])[0].get("path")
    if a:
        return a, False, None
    host = ((T.rbx_find(class_name="Humanoid", root=rig, limit=1).get("matches") or [{}])[0].get("path")
            or (T.rbx_find(class_name="AnimationController", root=rig, limit=1).get("matches") or [{}])[0].get("path"))
    if not host:
        return None, False, _no("%s has no Humanoid or AnimationController to hold an Animator" % _path(rig), refusedRule="not-a-rig")
    c = T.rbx_create("Animator", host, name="Animator")
    if not c.get("ok"):
        return None, False, c
    return c["path"], True, None


_ASSET_ID = re.compile(r"^(?:rbxassetid://)?(\d{1,20})$")


def _asset_content(animation):
    """An animation asset given as an int or 'rbxassetid://N' (or its digits) -> the content id; None for anything else."""
    if isinstance(animation, bool):
        return None
    if isinstance(animation, int):
        return "rbxassetid://%d" % animation if animation > 0 else None
    m = _ASSET_ID.match(animation) if isinstance(animation, str) else None
    return "rbxassetid://%s" % m.group(1) if m else None


def _sample_clip(animator, content_id, times, jmap, length_times=8, load_wait=15.0, restore=True):
    """Play one clip (a content id) on the rig's Animator in edit mode and read every joint's Transform at each time.
    times None: length_times points spread over the track's Length. -> ({samples, length, moved, times, startedAtRest},
    error or None). The temporary Animation and the track are removed whatever happens; Stop and a zero step leave the
    last pose. restore (the caller's): joints are put to rest before sampling only with it, and an error after Play
    stops the track and, with it, puts the rig back to rest, so an error never returns with the rig posed."""
    # With restore every joint starts at rest: Stop leaves the last pose, and a joint this clip has no key for yet would
    # otherwise still hold the previous clip's (measured on 0.748: Roblox's run animation keys UpperTorso late). Without
    # it the rig is not written to (one undo step per joint); startedAtRest says the samples began from the rig as found.
    start = {}
    if restore:
        _rest_back(jmap, True, animator)
    else:
        # "moved" is judged against the pose the rig was found in, not identity, so a rig left posed does not read as moved
        for part, j in jmap.items():
            vals, _e = _props(j["path"], ["Transform"])
            start[part] = _cf_list((vals or {}).get("Transform"))
    aname = T._unique_name(V.split_path(SCRATCH), "MifPreviewAnimation")
    anim = T.rbx_create("Animation", SCRATCH, name=aname, props={"AnimationId": content_id})
    track, playing, before = None, False, None
    try:
        if not anim.get("ok"):
            return None, _no("the temporary Animation failed: %s" % anim.get("error"), step="animation")
        ld = T.rbx_call(animator, "LoadAnimation", [anim["path"]])
        track = _handle(ld.get("value"))
        if not track:
            return None, _no("Animator:LoadAnimation answered %s" % (ld.get("error") or ld.get("value")), step="load")
        # An asset's track reads Length 0 until its keyframes arrive (measured on 0.748: the first load of an id read 0
        # straight after LoadAnimation and its real length a moment later), so wait for it, at most load_wait seconds.
        deadline = time.time() + load_wait
        while True:
            ln = T._rbx("get_props", target={"handle": track}, props=["Length"])
            length = V.decode(T._obj(ln.get("values")).get("Length"))
            if (_num(length) and length > 0) or time.time() >= deadline:
                break
            time.sleep(0.25)
        if times is None:
            if not _num(length) or length <= 0:
                return None, _no("the track's Length is %s: the asset did not load as an animation" % length,
                                 step="length")
            times = [round(length * i / length_times, 6) for i in range(length_times)]
        before = _playing(animator)
        playing = True   # from here an error path stops the track (a failed Play may still have started it)
        pl = T.rbx_call({"handle": track}, "Play", [0, 1, 1])
        if not pl.get("ok"):
            return None, _no("AnimationTrack:Play failed: %s" % pl.get("error"), step="play")
        prev, moved, rows = 0.0, False, []
        for t in times:
            st = T.rbx_call(animator, "StepAnimations", [max(0.0, t - prev)])
            prev = t
            if not st.get("ok"):
                return None, _no("Animator:StepAnimations failed: %s" % st.get("error"), step="step")
            tpos = T._rbx("get_props", target={"handle": track}, props=["TimePosition"])
            row = {"time": t, "trackTime": V.decode(T._obj(tpos.get("values")).get("TimePosition")), "joints": {}}
            for part, j in sorted(jmap.items()):
                vals, _e = _props(j["path"], ["Transform"])
                cf = _cf_list((vals or {}).get("Transform"))
                row["joints"][part] = cf
                if cf and not _cf_close(cf, start.get(part) or IDENTITY, 1e-4, 1e-4):
                    moved = True
            rows.append(row)
        playing = False
        T.rbx_call({"handle": track}, "Stop", [0])
        T.rbx_call(animator, "StepAnimations", [0])
        _drain(animator, before)
        return {"samples": rows, "length": length, "moved": moved, "times": times, "startedAtRest": bool(restore)}, None
    finally:
        if playing:
            # an error after Play: stop the track, step it out, and with restore put every joint back to rest
            T.rbx_call({"handle": track}, "Stop", [0])
            T.rbx_call(animator, "StepAnimations", [0])
            _drain(animator, before)
            if restore:
                _rest_back(jmap, True, animator)
        if track:
            _release(track)
        if anim.get("ok"):
            T.rbx_destroy(anim["path"])


def _playing(animator):
    """How many tracks the Animator still holds (Animator:GetPlayingAnimationTracks), or None when unread."""
    r = T.rbx_call(animator, "GetPlayingAnimationTracks")
    v = r.get("value")
    if isinstance(v, dict) and isinstance(v.get("items"), list):
        v = v["items"]
    return len(v) if r.get("ok") and isinstance(v, list) else None


def _drain(animator, before, tries=6):
    """Step a stopped track out of the Animator. Measured on 0.748: after AnimationTrack:Stop(0) and a zero step the
    track is still listed with WeightCurrent 1 and fades over about 0.3 s of StepAnimations, so the NEXT clip played on
    that Animator read a 50/50 blend with this one's last pose at time 0 (a second preview of one sequence on one rig
    failed at its first key). Step 0.5 s at a time until the Animator holds no more tracks than before Play."""
    for _ in range(tries):
        n = _playing(animator)
        if n is None or before is None or n <= before:
            return n
        T.rbx_call(animator, "StepAnimations", [0.5])
    return _playing(animator)


def _rest_back(jmap, restore, animator=None):
    """(every joint at identity, joints put back): the rest pose read after a preview, restored when asked. Then one
    Animator:StepAnimations(0): in edit mode the parts follow a joint's Transform only when the animator steps, so a
    Transform written back to identity alone leaves every part where the last pose put it (measured on 0.748)."""
    rest, reset = {}, []
    for part, j in sorted(jmap.items()):
        vals, _e = _props(j["path"], ["Transform"])
        cf = _cf_list((vals or {}).get("Transform"))
        rest[part] = cf is not None and _cf_close(cf, IDENTITY, 1e-4, 1e-4)
        if restore and not rest[part]:
            r = T.rbx_set_props(j["path"], {"Transform": IDENTITY})
            reset.append(part)
            rest[part] = bool(r.get("ok"))
    if animator:
        T.rbx_call(animator, "StepAnimations", [0])
    return all(rest.values()), reset


def _err_fields(err):
    return {k: v for k, v in (err or {}).items() if k != "ok"}


@T._reports
def rbx_anim_preview(rig: Any, animation: Any, times: list = None, joints: list = None, restore: bool = True,
                     compare_with: Any = None) -> dict:
    "Roblox Studio: play a KeyframeSequence, a CurveAnimation or an animation asset on a rig in edit mode and sample its joints. A sequence or curve is registered (KeyframeSequenceProvider:RegisterKeyframeSequence or AnimationClipProvider:RegisterAnimationClip) into a temporary Animation; an asset (animation as an asset id or 'rbxassetid://N'; read only, nothing is uploaded) plays by its own id. The clip is loaded on the rig's Animator (made if missing), played, and stepped with Animator:StepAnimations to each time (default: each keyframe's time; for an asset 8 times over its Length); every joint's Transform (or those whose Part1 is in joints) is read at each time. A KeyframeSequence is compared at its keyframe times with that keyframe's pose CFrames (weight 1 poses). An asset is compared with the sequence it came from, played the same way at the same times: compare_with (a KeyframeSequence in the place, the one rbx_anim_upload sent) or, without it, the asset's own sequence from KeyframeSequenceProvider:GetKeyframeSequenceAsync. Then the track is stopped and the rest pose read back; with restore, every joint starts at identity, and any joint left posed is put back to identity, an error after Play included; restore false never writes a joint (startedAtRest false: the samples began from the rig as found). Nothing is saved."
    _cls, err = _class_of(rig)
    if err:
        return err
    asset = _asset_content(animation)
    acls = None
    if asset is None:
        acls, err = _class_of(animation)
        if err:
            return err
        if acls not in ("KeyframeSequence", "CurveAnimation"):
            return _no("animation is a KeyframeSequence, a CurveAnimation or an asset id, got %s" % acls, refusedRule="type")
        if compare_with is not None:
            return _no("compare_with goes with an asset id: a KeyframeSequence is compared with its own poses")
    elif compare_with is not None:
        ccls, err = _class_of(compare_with)
        if err:
            return err
        if ccls != "KeyframeSequence":
            return _no("compare_with is a KeyframeSequence, got %s" % ccls, refusedRule="type")
    seq = None
    if acls == "KeyframeSequence":
        seq, err = read_sequence(animation)
        if err:
            return err
    if times is None and asset is None:
        times = sorted({k["time"] for k in seq["keyframes"]}) if seq else [0]
    if times is not None:
        if not isinstance(times, list) or not times or not all(_num(t) and t >= 0 for t in times) or len(times) > 60:
            return _no("times is a list of 1-60 times >= 0")
        times = sorted(set(times))
    jmap, root, err = rig_joints(rig)
    if err:
        return err
    if joints:
        unknown = sorted(set(joints) - set(jmap))
        if unknown:
            return _no("%s are not joints of the rig (each is a Motor6D's Part1 name)" % ", ".join(unknown),
                       refusedRule="not-a-joint", joints=sorted(jmap))
        jmap = {k: v for k, v in jmap.items() if k in joints}
    if asset is not None:
        return _preview_asset(rig, asset, compare_with, times, jmap, restore)
    provider = "KeyframeSequenceProvider" if acls == "KeyframeSequence" else "AnimationClipProvider"
    reg = T.rbx_call(provider, "RegisterKeyframeSequence" if acls == "KeyframeSequence" else "RegisterAnimationClip",
                     [_path(animation)])
    if not reg.get("ok") or not isinstance(reg.get("value"), str):
        return dict(reg, ok=False, step="register")
    animator, made_animator, err = _animator_for(rig)
    if err:
        return dict(err, step="animator")
    out = {"ok": False, "rig": _path(rig), "animation": _path(animation), "contentId": reg["value"], "samples": [],
           "animator": animator, "animatorMade": made_animator}
    try:
        got, err = _sample_clip(animator, reg["value"], times, jmap, restore=restore)
        if err:
            return dict(out, **_err_fields(err))
        out.update(samples=got["samples"], startedAtRest=got["startedAtRest"])
        moved, problems = got["moved"], []
        poses_at = {}
        if seq:
            def flatten(tree, acc):
                for n, p in tree.items():
                    if p.get("weight") == 1:
                        acc[n] = p["cframe"]
                    flatten(p.get("children") or {}, acc)
                return acc
            poses_at = {round(k["time"], 6): flatten(k["poses"], {}) for k in seq["keyframes"]}
        for row in got["samples"]:
            for part, cf in row["joints"].items():
                want = poses_at.get(round(row["time"], 6), {}).get(part)
                if want is not None and cf is not None and not _cf_close(cf, want, 1e-3, 1e-3):
                    problems.append("%s at %s: Transform %s, pose %s" % (part, row["time"], [round(x, 4) for x in cf],
                                                                         [round(x, 4) for x in want]))
        rest_ok, reset = _rest_back(jmap, restore, animator)
        out.update(moved=moved, restPose=rest_ok, resetJoints=reset, poseMismatches=problems)
        if not moved:
            out["error"] = ("no joint moved: Animator:StepAnimations did not pose the rig in edit mode (or the clip holds "
                            "no motion for these joints)")
        elif problems:
            out["error"] = "; ".join(problems[:10])
        elif not out["restPose"]:
            out["error"] = "the rig did not return to its rest pose"
        out["ok"] = moved and not problems and out["restPose"]
        out["verified"] = out["ok"]
        return out
    finally:
        if made_animator:
            T.rbx_destroy(animator)


def _preview_asset(rig, asset, compare_with, times, jmap, restore, tol=1e-3):
    """An animation asset played by its id beside the sequence it came from, at the same times, joint for joint."""
    out = {"ok": False, "rig": _path(rig), "animation": asset, "contentId": asset, "samples": []}
    seq_handle = None
    if compare_with is not None:
        src = _path(compare_with)
    else:
        g = T.rbx_call("KeyframeSequenceProvider", "GetKeyframeSequenceAsync", [asset], timeout=60)
        seq_handle = _handle(g.get("value"))
        if seq_handle:
            src = {"handle": seq_handle}
        elif g.get("ok") and isinstance(g.get("value"), str) and g["value"]:
            # The provider caches: a second call answers the SAME instance, so one moved into the place comes back as its
            # path (measured on 0.748). Use it where it is; it is not ours to release.
            src = g["value"]
        else:
            return dict(out, error="KeyframeSequenceProvider:GetKeyframeSequenceAsync(%s) answered %s" % (
                asset, g.get("error") or g.get("value")), refusedRule=g.get("refusedRule"), step="source")
    animator, made_animator = None, False
    out["sourceHandleKept"] = bool(seq_handle)
    try:
        reg = T.rbx_call("KeyframeSequenceProvider", "RegisterKeyframeSequence", [src])
        if not reg.get("ok") or not isinstance(reg.get("value"), str):
            return dict(out, error="registering the source sequence failed: %s" % (reg.get("error") or reg.get("value")),
                        step="register")
        animator, made_animator, err = _animator_for(rig)
        if err:
            return dict(err, step="animator")
        out.update(animator=animator, animatorMade=made_animator,
                   source=src if isinstance(src, str) else "GetKeyframeSequenceAsync(%s)" % asset)
        a, err = _sample_clip(animator, asset, times, jmap, restore=restore)
        if err:
            return dict(out, **_err_fields(err))
        b, err = _sample_clip(animator, reg["value"], a["times"], jmap, restore=restore)
        if err:
            return dict(out, **dict(_err_fields(err), step="source-%s" % err.get("step")))
        worst, problems = 0.0, []
        for ra, rb in zip(a["samples"], b["samples"]):
            for part, cf in ra["joints"].items():
                d = _cf_diff(cf, rb["joints"].get(part))
                if d is None:
                    problems.append("%s at %s unread" % (part, ra["time"]))
                    continue
                worst = max(worst, d[0], d[1])
                if d[0] > tol or d[1] > tol:
                    problems.append("%s at %s: asset and source differ by %.4f studs, %.4f rotation"
                                    % (part, ra["time"], d[0], d[1]))
        rest_ok, reset = _rest_back(jmap, restore, animator)
        out.update(samples=a["samples"], length=a["length"], sourceLength=b["length"], moved=a["moved"],
                   startedAtRest=a["startedAtRest"],
                   worstDifference=round(worst, 6), tolerance=tol, mismatches=problems, restPose=rest_ok,
                   resetJoints=reset)
        if not a["moved"]:
            out["error"] = "no joint moved: the asset holds no motion for these joints"
        elif problems:
            out["error"] = "; ".join(problems[:10])
        elif not rest_ok:
            out["error"] = "the rig did not return to its rest pose"
        out["ok"] = bool(a["moved"]) and not problems and rest_ok
        out["verified"] = out["ok"]
        return out
    finally:
        # seq_handle is NOT released: the provider caches the instance it answers, and release_handle Destroys an object
        # outside the place, after which every later GetKeyframeSequenceAsync of that id answers the destroyed, empty
        # sequence (measured on 0.748: the second preview of one asset then differed at every joint).
        if made_animator:
            T.rbx_destroy(animator)


# ---- Blender action -> keyframes. The samples arrive in Roblox joint space (what Motor6D.Transform holds). A Blender
# armature's bone basis is not converted here: an imported Roblox rig's bones keep the joint frames only when exported
# with Roblox's settings, and that is measured live before a tolerance goes in the internal notes.
def _zup_to_yup(cf):
    """A CFrame expressed in Blender's Z-up axes -> Roblox's Y-up: x stays, Blender y becomes -z, z becomes y."""
    m = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]   # rows: Roblox axes in Blender coordinates
    p, r = cf[:3], [cf[3:6], cf[6:9], cf[9:12]]
    pos = [sum(m[i][k] * p[k] for k in range(3)) for i in range(3)]
    mr = [[sum(m[i][k] * r[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    rot = [[sum(mr[i][k] * m[j][k] for k in range(3)) for j in range(3)] for i in range(3)]   # m r m^T
    return pos + [x for row in rot for x in row]


@T._reports
def rbx_anim_from_blender(samples: dict, rig: Any, mapping: dict = None, name: str = "MifBlenderAction",
                          parent: Any = SCRATCH, axes: str = "roblox", scale: float = 1.0, tolerance: float = 0.01,
                          loop: bool = False, compare: bool = True) -> dict:
    "Roblox Studio: a Blender action as a KeyframeSequence on a rig, compared sample by sample. samples is {times: [seconds], bones: {bone: [CFrame per time]}} with each CFrame the bone's pose relative to its rest, in joint space (what Motor6D.Transform holds); axes 'blender' converts Z-up to Y-up, scale multiplies positions. Bones map to the rig's joints by mapping {bone: part} or by the same name; bones with no joint are listed, not guessed. The keys go through rbx_keyframes_set, then rbx_anim_preview steps the rig to every sample time and each joint's Transform is compared with the sample within tolerance (studs and rotation-matrix units)."
    if not isinstance(samples, dict) or not isinstance(samples.get("times"), list) or not isinstance(samples.get("bones"), dict):
        return _no("samples is {times: [...], bones: {bone: [CFrame per time]}}")
    times = samples["times"]
    if not times or not all(_num(t) and t >= 0 for t in times) or times != sorted(times) or len(times) > 600:
        return _no("times rise, are >= 0, 1-600 of them")
    if axes not in ("roblox", "blender") or not _num(scale) or scale <= 0 or not _num(tolerance) or tolerance <= 0:
        return _no("axes is roblox or blender, scale and tolerance positive numbers")
    jmap, root, err = rig_joints(rig)
    if err:
        return err
    mapping = mapping or {}
    mapped, unmapped = {}, []
    for bone, cfs in samples["bones"].items():
        part = mapping.get(bone, bone)
        if part not in jmap:
            unmapped.append(bone)
            continue
        if not isinstance(cfs, list) or len(cfs) != len(times):
            return _no("bone %s has %s samples, times has %d" % (bone, len(cfs) if isinstance(cfs, list) else "no", len(times)))
        row = []
        for cf in cfs:
            c = _cf_list(cf)
            if c is None or not all(_num(x) for x in c):
                return _no("bone %s: each sample is a CFrame" % bone)
            if axes == "blender":
                c = _zup_to_yup(c)
            row.append([x * scale for x in c[:3]] + c[3:])
        mapped[part] = row
    if not mapped:
        return _no("no bone maps to a joint of %s" % _path(rig), refusedRule="not-a-joint", unmapped=unmapped,
                   joints=sorted(jmap))
    kfs = [{"time": t, "name": "Sample_%d" % i, "poses": {p: mapped[p][i] for p in mapped}} for i, t in enumerate(times)]
    ks = rbx_keyframes_set(kfs, name=name, parent=parent, rig=rig, loop=loop, replace=True)
    out = {"path": ks.get("path"), "mapped": sorted(mapped), "unmapped": sorted(unmapped), "samples": len(times),
           "keyframes": ks.get("ok")}
    if not ks.get("ok"):
        return dict(out, ok=False, ran=True, error="writing the keyframes failed: %s" % ks.get("error"))
    if not compare:
        return dict(out, ok=True, verified=False, note="compare false: written and read back, not played")
    pv = rbx_anim_preview(rig, ks["path"], times=times, joints=sorted(mapped))
    worst, misses = 0.0, []
    for i, row in enumerate(pv.get("samples") or []):
        for part, cf in (row.get("joints") or {}).items():
            d = _cf_diff(cf, mapped[part][i]) if cf else None
            if d is None:
                misses.append("%s at %s unread" % (part, row["time"]))
                continue
            worst = max(worst, d[0], d[1])
            if d[0] > tolerance or d[1] > tolerance:
                misses.append("%s at %s off by %.4f studs, %.4f rotation" % (part, row["time"], d[0], d[1]))
    good = bool(pv.get("samples")) and not misses
    out.update(ok=good, verified=good, worstDifference=round(worst, 6), tolerance=tolerance,
               preview={k: pv.get(k) for k in ("ok", "moved", "restPose", "error")})
    if not good:
        out["error"] = "; ".join(misses[:10]) or pv.get("error") or "the preview returned no samples"
    return out


# ---- rbxmx: a plugin cannot write files, so the MCP side writes the XML Studio and Open Cloud read.
# Token values are the enum items' Value fields in the 0.748 API dump (the catalog keeps names only); the offline suite
# compares them with the dump when it is on disk.
ENUM_VALUES = {
    "AnimationPriority": {"Idle": 0, "Movement": 1, "Action": 2, "Action2": 3, "Action3": 4, "Action4": 5, "Core": 1000},
    "PoseEasingStyle": {"Linear": 0, "Constant": 1, "Elastic": 2, "Cubic": 3, "Bounce": 4, "CubicV2": 5},
    "PoseEasingDirection": {"In": 0, "Out": 1, "InOut": 2},
}
_CF_TAGS = ("X", "Y", "Z", "R00", "R01", "R02", "R10", "R11", "R12", "R20", "R21", "R22")


def _fmt(x):
    return repr(float(x)) if isinstance(x, float) else str(x)


def rbxmx_from_sequence(seq):
    """A read_sequence() result -> rbxmx text (XML, version 4)."""
    ref = [0]

    def item(cls, props, kids):
        el = ET.Element("Item", {"class": cls, "referent": "RBX%d" % ref[0]})
        ref[0] += 1
        pe = ET.SubElement(el, "Properties")
        for kind, name, val in props:
            if kind == "CoordinateFrame":
                c = ET.SubElement(pe, "CoordinateFrame", {"name": name})
                for tag, x in zip(_CF_TAGS, val):
                    ET.SubElement(c, tag).text = _fmt(x)
            else:
                ET.SubElement(pe, kind, {"name": name}).text = (("true" if val else "false") if kind == "bool" else _fmt(val)
                                                               if kind in ("float", "token") else str(val))
        el.extend(kids)
        return el

    def poses(tree):
        return [item("Pose", [("string", "Name", n), ("CoordinateFrame", "CFrame", p["cframe"]),
                              ("float", "Weight", float(p["weight"])),
                              ("token", "EasingStyle", ENUM_VALUES["PoseEasingStyle"][p["easing_style"]]),
                              ("token", "EasingDirection", ENUM_VALUES["PoseEasingDirection"][p["easing_direction"]])],
                     poses(p.get("children") or {})) for n, p in tree.items()]

    kfs = []
    for k in seq["keyframes"]:
        markers = [item("KeyframeMarker", [("string", "Name", n), ("string", "Value", v)], []) for n, v in k["markers"].items()]
        kfs.append(item("Keyframe", [("string", "Name", k["name"]), ("float", "Time", float(k["time"]))],
                        poses(k["poses"]) + markers))
    top = item("KeyframeSequence", [("string", "Name", seq.get("name") or "KeyframeSequence"), ("bool", "Loop", bool(seq["loop"])),
                                    ("token", "Priority", ENUM_VALUES["AnimationPriority"][seq["priority"]])], kfs)
    root = ET.Element("roblox", {"xmlns:xmime": "http://www.w3.org/2005/05/xmlmime",
                                 "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
                                 "xsi:noNamespaceSchemaLocation": "http://www.roblox.com/roblox.xsd", "version": "4"})
    root.append(top)
    ET.indent(root)
    return ET.tostring(root, encoding="unicode")


def sequence_from_rbxmx(text):
    """rbxmx text -> the read_sequence() shape, for proving a written file holds what was read."""
    inv = {e: {v: k for k, v in m.items()} for e, m in ENUM_VALUES.items()}
    root = ET.fromstring(text)
    top = root.find("Item[@class='KeyframeSequence']")

    def props(el):
        out = {}
        for p in el.find("Properties"):
            n = p.get("name")
            if p.tag == "CoordinateFrame":
                out[n] = [float(p.find(t).text) for t in _CF_TAGS]
            elif p.tag in ("float", "double"):
                out[n] = float(p.text)
            elif p.tag == "token":
                out[n] = int(p.text)
            elif p.tag == "bool":
                out[n] = p.text == "true"
            else:
                out[n] = p.text or ""
        return out

    def poses(el):
        res = {}
        for c in el.findall("Item[@class='Pose']"):
            p = props(c)
            res[p["Name"]] = {"cframe": p["CFrame"], "weight": p["Weight"],
                              "easing_style": inv["PoseEasingStyle"][p["EasingStyle"]],
                              "easing_direction": inv["PoseEasingDirection"][p["EasingDirection"]], "children": poses(c)}
        return res

    tp = props(top)
    seq = {"name": tp.get("Name"), "loop": tp.get("Loop"), "priority": inv["AnimationPriority"][tp["Priority"]], "keyframes": []}
    for k in top.findall("Item[@class='Keyframe']"):
        kp = props(k)
        seq["keyframes"].append({"name": kp["Name"], "time": kp["Time"], "poses": poses(k),
                                 "markers": {props(m)["Name"]: props(m)["Value"] for m in k.findall("Item[@class='KeyframeMarker']")}})
    return seq


def _seq_equal(a, b):
    problems = []
    if a["loop"] != b["loop"] or a["priority"] != b["priority"]:
        problems.append("Loop/Priority differ")
    ka, kb = {k["name"]: k for k in a["keyframes"]}, {k["name"]: k for k in b["keyframes"]}
    if set(ka) != set(kb):
        problems.append("keyframes differ: %s vs %s" % (sorted(ka), sorted(kb)))
    for n in set(ka) & set(kb):
        if abs(ka[n]["time"] - kb[n]["time"]) > 1e-6:
            problems.append("%s.Time differs" % n)
        _compare_poses(ka[n]["poses"], kb[n]["poses"], n, problems)
        if ka[n]["markers"] != kb[n]["markers"]:
            problems.append("%s markers differ" % n)
    return problems


def rbx_anim_rbxmx(target: Any, file: str = None, studio_check: bool = True) -> dict:
    "Roblox Studio: write a KeyframeSequence to an .rbxmx file (Roblox's XML model format) on this machine - a plugin cannot write files, so the MCP side does. The sequence is read from Studio key for key, written with its poses, markers, Loop and Priority (enum tokens from the API dump), then the file is parsed again and compared with what was read. With studio_check, Studio itself reads the file too (SerializationService:DeserializeInstancesAsync on its bytes), the copy is put in Workspace.MifScratch, read key for key, compared, and removed. file defaults to a temp folder."
    seq, err = read_sequence(target)
    if err:
        return err
    if not seq["keyframes"]:
        return _no("%s has no keyframes to write" % _path(target))
    if file is None:
        file = os.path.join(tempfile.gettempdir(), "MifBridge_%s.rbxmx" % re.sub(r"[^A-Za-z0-9_-]", "_", seq["name"] or "anim"))
    if not str(file).lower().endswith(".rbxmx"):
        return _no("file ends in .rbxmx")
    text = rbxmx_from_sequence(seq)
    try:
        with open(file, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        with open(file, encoding="utf-8") as f:
            again = sequence_from_rbxmx(f.read())
    except (OSError, ET.ParseError, KeyError, ValueError) as exc:
        return {"ok": False, "ran": True, "file": file, "error": "writing or re-reading the file failed: %s" % exc}
    problems = _seq_equal(seq, again)
    studio = None
    if studio_check and not problems:
        import base64
        raw = text.encode("utf-8")
        if len(raw) > 1 << 20:
            studio = {"skipped": "the file is over the 1 MB a call buffers cross"}
        else:
            r = T.rbx_call("SerializationService", "DeserializeInstancesAsync", [base64.b64encode(raw).decode()], timeout=60)
            vals = r.get("value") if isinstance(r.get("value"), list) else []
            h = _handle(vals[0]) if vals else None
            if not r.get("ok") or not h or len(vals) != 1:
                problems.append("Studio could not read the file: %s" % (r.get("error") or r.get("value")))
            else:
                path, perr = T._place_handle(h, V.split_path(SCRATCH), (seq.get("name") or "Anim") + "_rbxmx")
                _release(h)   # the deserialized sequence's handle: placed or not, it is the plugin's to drop now
                if perr:
                    problems.append("placing Studio's copy failed: %s" % perr)
                else:
                    copy, cerr = read_sequence(path)
                    diff = _seq_equal(seq, copy) if copy else ["reading Studio's copy failed: %s" % (cerr or {}).get("error")]
                    problems += ["Studio's copy: %s" % x for x in diff]
                    T.rbx_destroy(path)
                    studio = {"read": True, "equal": not diff}
    return {"ok": not problems, "verified": not problems, "file": file, "bytes": len(text.encode("utf-8")),
            "keyframes": len(seq["keyframes"]), "studioCheck": studio, "problems": problems or None,
            "error": "; ".join(problems[:10]) if problems else None}


def rbx_anim_upload(target: Any, display_name: str, confirm: bool = False, rig: Any = None) -> dict:
    "Roblox: upload a KeyframeSequence as an Animation asset through Open Cloud with the person's own key - rbx_anim_rbxmx writes the file, rbx_asset_upload sends it with assetType Animation (permanent and moderated: confirm must be true, the panel in full mode). With rig, the uploaded id is loaded into an Animation and played on the rig with rbx_anim_preview at the sequence's key times, and each joint compared with the keys."
    if confirm is not True:
        return _no("an upload makes a permanent, moderated asset on the person's account: pass confirm true once they agreed",
                   refusedRule="needs-confirm")
    w = rbx_anim_rbxmx(target)
    if not w.get("ok"):
        return dict(w, step="rbxmx")
    up = T.rbx_asset_upload(w["file"], display_name, confirm=True, asset_type="Animation")
    out = {"file": w["file"], "upload": {k: up.get(k) for k in ("ok", "assetId", "moderationState", "error", "refusedRule")}}
    if not up.get("ok"):
        return dict(out, ok=False, ran=up.get("ran"), error="the upload failed: %s" % up.get("error"))
    if rig is None:
        return dict(out, ok=True, verified=False, note="uploaded; pass rig to play it back and compare")
    pv = rbx_anim_preview(rig, up.get("assetId"), compare_with=target)
    good = bool(pv.get("ok"))
    return dict(out, ok=good, verified=good, assetId=up.get("assetId"),
                playback={k: pv.get(k) for k in ("ok", "moved", "worstDifference", "tolerance", "restPose", "error")},
                **({} if good else {"error": "uploaded, but playing it back: %s" % pv.get("error")}))


# ---------------------------------------------------------------- Studio's own pickers, with the person at the screen
# Mifsopo, 2026-11-20 ("yes to all"): the file picker may open on his screen when he is there to answer it. The plugin's
# person_prompt op (0.17.3) opens one only in full mode and only when the call says personPresent; without that every
# picker stays refused needs-person, with the menu that does the same by hand.
PROMPT_WAIT = (5, 580)  # the daemon takes a call timeout of at most 600 s, and the call waits wait_seconds + 15


def _person_prompt(kind, person_present, wait_seconds, steps, **extra):
    if not isinstance(person_present, bool):
        return _no("person_present is true or false")
    if not isinstance(wait_seconds, int) or isinstance(wait_seconds, bool) or not PROMPT_WAIT[0] <= wait_seconds <= PROMPT_WAIT[1]:
        return _no("wait_seconds is %d-%d" % PROMPT_WAIT)
    if not person_present:
        return _no("this opens one of Studio's pickers on the person's screen; call it with person_present: true only when "
                   "the person is there and has agreed to pick the file, or do it by hand", refusedRule="needs-person",
                   steps=steps)
    r = T._rbx("person_prompt", kind=kind, personPresent=True, waitSeconds=wait_seconds, timeout=wait_seconds + 15, **extra)
    if not r.get("ok"):
        return r
    if r.get("path"):
        cls = T.rbx_get_props(r["path"], ["ClassName", "Name"])
        r["readBack"] = cls.get("values") if cls.get("ok") else {"error": cls.get("error")}
        r["verified"] = bool(cls.get("ok"))
        r["ok"] = r["verified"]
    return r


IMPORT_STEPS = {
    "file": ["File > Import 3D (or the Avatar tab > Import 3D) and pick the file"],
    "files": ["File > Import 3D (or the Avatar tab > Import 3D) and pick the files"],
    "fbx_rig": ["Avatar tab > Import 3D, pick the .fbx, and import it as a rig"],
}


def rbx_import_prompt(kind: str = "fbx_rig", person_present: bool = False, file_types: list = None, is_r15: bool = True,
                      wait_seconds: int = 300) -> dict:
    "Roblox Studio: open one of Studio's own importers for the person to pick a file - kind file or files (StudioService:PromptImportFileAsync / PromptImportFilesAsync: File objects with Name and Size, held as handles; the place is not changed, file_types filters the picker), or fbx_rig (Plugin:ImportFbxRigAsync: the rig Studio builds in the Workspace, found by path and read back). The picker opens on the person's screen, so this runs only with person_present: true, said only when the person is there and agreed, and in full mode; otherwise refused needs-person with the menu that does it by hand. A closed picker answers cancelled; no pick within wait_seconds answers outcomeUnknown."
    if kind not in IMPORT_STEPS:
        return _no("kind is file, files or fbx_rig")
    if file_types is not None and not (isinstance(file_types, list) and all(isinstance(x, str) and x for x in file_types)):
        return _no("file_types is a list of extensions such as [\"fbx\", \"png\"]")
    if not isinstance(is_r15, bool):
        return _no("is_r15 is true or false")
    return _person_prompt(kind, person_present, wait_seconds, IMPORT_STEPS[kind], filter=file_types or [], isR15=is_r15)


def rbx_anim_import_fbx(kind: str = "animation", rig: Any = None, person_present: bool = False, is_r15: bool = True,
                        wait_seconds: int = 300) -> dict:
    "Roblox Studio: Studio's own FBX importers - kind animation (Plugin:ImportFbxAnimationAsync onto rig, a Model: the KeyframeSequence Studio makes, found by path and read back) or rig (Plugin:ImportFbxRigAsync). Each opens a file picker on the person's screen, so it runs only with person_present: true, said only when the person is there and agreed, and in full mode; otherwise refused needs-person with the menu. Without a person: rbx_anim_from_blender or rbx_keyframes_set."
    if kind not in ("animation", "rig"):
        return _no("kind is animation or rig")
    if not isinstance(is_r15, bool):
        return _no("is_r15 is true or false")
    steps = (["Avatar tab > Animation Editor, select the rig, then ... > Import > From FBX Animation"] if kind == "animation"
             else ["Avatar tab > Import 3D (or File > Import 3D), pick the .fbx, and import it as a rig"])
    if kind == "rig":
        out = _person_prompt("fbx_rig", person_present, wait_seconds, steps, isR15=is_r15)
    else:
        if person_present is True:
            cls, err = _class_of(rig) if rig is not None else (None, _no("an animation import needs rig, the Model it plays on"))
            if err:
                return err
            if cls != "Model":
                return _no("rig is a Model, got %s" % cls, refusedRule="type")
        out = _person_prompt("fbx_animation", person_present, wait_seconds, steps,
                             rig=V.split_path(rig) if isinstance(rig, str) else rig, isR15=is_r15)
    out.setdefault("member", "Plugin:ImportFbxAnimationAsync" if kind == "animation" else "Plugin:ImportFbxRigAsync")
    if out.get("refusedRule") == "needs-person":
        out["instead"] = "rbx_anim_from_blender (Blender samples to keyframes, compared) or rbx_keyframes_set"
    return out


@T._reports
def rbx_ik_set(host: Any, props: dict, name: str = "MifIK", step: float = 0.1, replace: bool = False,
               run_check: bool = False, settle: float = 2) -> dict:
    "Roblox Studio: an IKControl under host (a Humanoid or AnimationController) with props such as Type (Transform, Position, Rotation, LookAt), ChainRoot, EndEffector, Target, Pole, Weight, Priority, SmoothTime, Enabled. Read back: every property sent, IKControl:GetChainCount and GetChainLength, every chain node's world CFrame (GetNodeWorldCFrame) and GetRawFinalTarget. IKControl does not solve in edit mode, so the solve is proven with run_check: a run-mode play probe (rbx_play_probe, on a copy of the place) samples the EndEffector's world position after settle seconds, and it must lie within 0.05 stud of the Target's. An unanchored rig falls in run mode; anchor its root first."
    hcls, err = _class_of(host)
    if err:
        return err
    if hcls not in ("Humanoid", "AnimationController"):
        return _no("host is a Humanoid or AnimationController, got %s" % hcls, refusedRule="type")
    if not isinstance(props, dict) or not props:
        return _no("props is an object of IKControl properties")
    nm, err = _free_name(_path(host), name, replace)
    if err:
        return err
    c = T.rbx_create("IKControl", host, name=nm, props=props)
    if not c.get("ok"):
        return c
    ik = c["path"]
    out = {"path": ik, "after": c.get("after")}
    if step:
        rig = V.split_path(ik)[:-2]
        animator, _made, aerr = _animator_for(rig)
        if aerr:
            return dict(out, ok=False, ran=True, error="no Animator to step: %s" % aerr.get("error"))
        s = T.rbx_call(animator, "StepAnimations", [step])
        out["stepped"] = s.get("ok")
    cnt = T.rbx_call(ik, "GetChainCount")
    ln = T.rbx_call(ik, "GetChainLength")
    out.update(chainCount=cnt.get("value"), chainLength=ln.get("value"), nodes=[])
    n = cnt.get("value") if isinstance(cnt.get("value"), int) else 0
    for i in range(0, n + 1):
        r = T.rbx_call(ik, "GetNodeWorldCFrame", [i])
        out["nodes"].append({"index": i, "cframe": r.get("value") if r.get("ok") else None,
                             "error": None if r.get("ok") else r.get("error")})
    raw = T.rbx_call(ik, "GetRawFinalTarget")
    out["rawFinalTarget"] = raw.get("value")
    solved = not (isinstance(raw.get("value"), dict) and _cf_close(raw["value"], IDENTITY, 1e-6, 1e-6))
    out["solved"] = solved
    tgt = props.get("Target")
    if tgt is not None:
        tcls, _e = _class_of(tgt)
        prop = "WorldPosition" if tcls in ("Attachment", "Bone") else "Position"
        tv, terr = _props(tgt, [prop])
        ends = [x["cframe"] for x in out["nodes"] if x["cframe"]]
        if not terr and ends and isinstance(tv.get(prop), dict):
            tp = [tv[prop][k] for k in "xyz"]
            d = math.dist(_cf_list(ends[-1])[:3], tp)
            out.update(targetPosition=tp, endDistance=round(d, 5))
    if run_check:
        eff = props.get("EndEffector")
        if tgt is None or eff is None:
            return dict(out, ok=False, ran=True, error="run_check needs Target and EndEffector")
        ecls, _e = _class_of(eff)
        tcls, _e = _class_of(tgt)
        eprop = "WorldPosition" if ecls in ("Attachment", "Bone") else "Position"
        tprop = "WorldPosition" if tcls in ("Attachment", "Bone") else "Position"
        tv, terr = _props(tgt, [tprop])
        if terr:
            return dict(out, ok=False, ran=True, error="reading the target: %s" % terr.get("error"))
        # One run, no axis: the probe reads a Vector3 whole as [X, Y, Z] (the plugin's probe encodes it so when no axis is
        # given), so the point is one sample of one simulation, not three axes joined from three play tests.
        pr = T.rbx_play_probe("sample", target=eff, property=eprop, seconds=0.5, interval=0.25, run_mode=True,
                              settle=settle)
        sm = [x.get("v") for x in pr.get("samples") or [] if isinstance(x, dict) and isinstance(x.get("v"), list)
              and len(x["v"]) == 3 and all(_num(c) for c in x["v"])]
        if not pr.get("ok") or not sm:
            return dict(out, ok=False, ran=True, error="the run-mode probe of %s.%s failed: %s" % (
                eff, eprop, pr.get("error") or "no sample held three numbers"))
        pos = list(sm[-1])
        tp = [tv[tprop][k] for k in "xyz"]
        d = math.dist(pos, tp)
        out["inRun"] = {"endEffector": [round(x, 4) for x in pos], "target": tp, "distance": round(d, 5)}
        good = bool(c.get("ok")) and d <= 0.05
        out.update(ok=good, verified=good, solvedInRun=d <= 0.05)
        if not good:
            out["error"] = "in run mode the end effector is %.4f studs from the target" % d
        return out
    good = bool(c.get("ok")) and (tgt is None or (out.get("endDistance") is not None and out["endDistance"] <= 0.05))
    out.update(ok=good, verified=good)
    if not good:
        out["error"] = ("the end node is %s studs from the target" % out.get("endDistance") if out.get("endDistance") is not None
                        else "the end node or the target could not be read")
        if not solved:
            out["error"] += ("; the solver did not run: GetRawFinalTarget is the identity after Animator:StepAnimations "
                             "(ledger RB505: IKControl solves only in a running experience - pass run_check; the properties "
                             "are set and read back)")
            out["propertiesVerified"] = True
    return out


@T._reports
def rbx_bones(transforms: dict = None, bones: list = None) -> dict:
    "Roblox Studio: skinned-mesh bones posed and read. transforms maps a Bone path to a CFrame set as Bone.Transform (one undo step each); bones lists paths to read. Read back: Bone.Transform equal to what was sent, and Bone.TransformedWorldCFrame for every bone touched."
    if not transforms and not bones:
        return _no("give transforms {bone path: CFrame} and/or bones [paths]")
    out, problems = {"bones": {}}, []
    for path, cf in (transforms or {}).items():
        r = T.rbx_set_props(path, {"Transform": cf})
        if not r.get("ok"):
            problems.append("%s: %s" % (path, r.get("error")))
        out["bones"][path] = {"transform": (r.get("after") or {}).get("Transform"), "verified": r.get("verified")}
    for path in list(transforms or {}) + list(bones or []):
        vals, err = _props(path, ["TransformedWorldCFrame", "Transform"])
        if err:
            problems.append("%s: %s" % (path, err.get("error")))
            continue
        out["bones"].setdefault(path, {}).update(worldCFrame=vals.get("TransformedWorldCFrame"), transform=vals.get("Transform"))
    out.update(ok=not problems, verified=not problems)
    if problems:
        out["error"] = "; ".join(problems)
    return out


def _frame_diff(a, b):
    """Fraction of pixels whose RGB moved by more than 24 between two same-size PNG captures, or None."""
    from mif_roblox import png as PNG
    try:
        wa, ha, ra = PNG.decode_rgba(open(a, "rb").read())
        wb, hb, rb = PNG.decode_rgba(open(b, "rb").read())
    except (OSError, ValueError):
        return None
    if (wa, ha) != (wb, hb):
        return None
    n = wa * ha
    moved = sum(1 for i in range(0, n * 4, 4) if max(abs(ra[i + k] - rb[i + k]) for k in range(3)) > 24)
    return moved / n


@T._reports
def rbx_face_controls(target: Any, values: dict = None, prove: bool = False) -> dict:
    "Roblox Studio: FACS values on a dynamic head. target is a FaceControls, or a head or character holding one (found under it). values maps FaceControls properties (JawDrop, LeftEyeClosed, Pucker... 50 of them, each 0-1) to numbers, set in one undo step and read back; without values, all 50 are read. With prove, the head is captured close up before and after (rbx_screenshot) and the share of pixels that changed is reported: the face moved, not just the number. A target with no FaceControls is refused no-face-controls."
    cat, err = T._catalog()
    if err:
        return err
    cls, err = _class_of(target)
    if err:
        return err
    fc = _path(target) if cls == "FaceControls" else \
        (T.rbx_find(class_name="FaceControls", root=target, limit=1).get("matches") or [{}])[0].get("path")
    if not fc:
        return _no("%s holds no FaceControls (a dynamic head has one)" % _path(target), refusedRule="no-face-controls")
    names = sorted(n for n, (_o, r) in cat.members_of("FaceControls").items()
                   if r[0] == "P" and cat.write_refusal("FaceControls", n) is None and r[1] == "float")
    if values is not None:
        bad = {k: v for k, v in values.items() if k not in names or not _num(v) or not 0 <= v <= 1}
        if bad:
            return _no("values: %s are not FaceControls properties with 0-1 numbers" % sorted(bad), valid=names)
        shots = []
        if prove:
            head = V.join_path(V.split_path(fc)[:-1])
            hv, herr = _props(head, ["CFrame", "Size"])
            if herr:
                return herr
            cf = _cf_list(hv["CFrame"])
            look = [cf[3 + 2], cf[3 + 5], cf[3 + 8]]          # the head's look vector is -Z: the camera sits on -look
            cam = {"position": [cf[0] - look[0] * 3, cf[1] - look[1] * 3, cf[2] - look[2] * 3], "look_at": cf[:3]}
            shots.append(T.rbx_screenshot(camera=cam, max_width=480, name="face_before"))
        r = T.rbx_set_props(fc, values)
        out = dict(r, faceControls=fc)
        if prove:
            shots.append(T.rbx_screenshot(camera=cam, max_width=480, name="face_after"))
            files = [x.get("file") for x in shots]
            diff = _frame_diff(*files) if all(files) else None
            out.update(captures=files, changedPixels=None if diff is None else round(diff, 4))
            if not diff:
                out.update(ok=False, verified=False, error="the values read back but the head's capture did not change "
                           "(%s)" % ("captures failed" if diff is None else "0 pixels moved"))
        return out
    vals, err = _props(fc, names)
    if err:
        return err
    return {"ok": True, "faceControls": fc, "values": vals, "count": len(vals)}


@T._reports
def rbx_humanoid(target: Any, states: dict = None, hip_height: float = None, description: dict = None) -> dict:
    "Roblox Studio: a Humanoid's states, hip height and description. states maps HumanoidStateType names to enabled (Humanoid:SetStateEnabled, read back with GetStateEnabled); hip_height sets Humanoid.HipHeight; description is HumanoidDescription properties (scales, colors, asset ids) put in a temporary HumanoidDescription and applied with Humanoid:ApplyDescriptionAsync, then read back with Humanoid:GetAppliedDescription, field by field. Always answers the current state (GetState), RigType and HipHeight."
    cls, err = _class_of(target)
    if err:
        return err
    if cls != "Humanoid":
        return _no("target is a Humanoid, got %s" % cls, refusedRule="type")
    cat, err = T._catalog()
    if err:
        return err
    out, problems = {"target": _path(target)}, []
    for st, en in (states or {}).items():
        if st not in (cat.enums.get("HumanoidStateType") or []) or not isinstance(en, bool):
            return _no("states maps a HumanoidStateType name to true/false; %r is not one" % st)
    for st, en in (states or {}).items():
        r = T.rbx_call(target, "SetStateEnabled", [st, en])
        g = T.rbx_call(target, "GetStateEnabled", [st])
        if not r.get("ok") or g.get("value") is not en:
            problems.append("state %s reads %s after SetStateEnabled(%s): %s" % (st, g.get("value"), en, r.get("error") or ""))
        out.setdefault("states", {})[st] = g.get("value")
    if description:
        if not isinstance(description, dict):
            return _no("description is an object of HumanoidDescription properties")
        dname = T._unique_name(V.split_path(SCRATCH), "MifDescription")
        d = T.rbx_create("HumanoidDescription", SCRATCH, name=dname, props=description)
        if not d.get("ok"):
            return dict(d, step="description")
        try:
            ap = T.rbx_call(target, "ApplyDescriptionAsync", [d["path"]], timeout=90)
            if not ap.get("ok"):
                problems.append("ApplyDescriptionAsync: %s" % ap.get("error"))
            else:
                got = T.rbx_call(target, "GetAppliedDescription")
                h = _handle(got.get("value"))
                applied = None
                if h:
                    applied, aerr = _props({"handle": h}, list(description))
                    _release(h)
                elif got.get("value"):
                    applied, aerr = _props(got["value"], list(description))
                if applied is None:
                    problems.append("GetAppliedDescription read nothing: %s" % (got.get("error") or got.get("value")))
                else:
                    out["applied"] = applied
                    wire, _e = T._encode_props(cat, "HumanoidDescription", description)
                    for k, w in (wire or {}).items():
                        sent = V.decode(w)
                        if not _loose_same(sent, applied.get(k)):
                            problems.append("applied %s reads %s, sent %s" % (k, applied.get(k), sent))
        finally:
            T.rbx_destroy(d["path"])
    # after the description: ApplyDescriptionAsync recomputes HipHeight from the scales
    if hip_height is not None:
        r = T.rbx_set_props(target, {"HipHeight": hip_height})
        if not r.get("ok"):
            problems.append("HipHeight: %s" % r.get("error"))
    now = T.rbx_call(target, "GetState")
    vals, _e = _props(target, ["RigType", "HipHeight"])
    out.update(state=now.get("value"), **(vals or {}))
    if hip_height is not None and _num((vals or {}).get("HipHeight")) and abs(vals["HipHeight"] - hip_height) > 1e-4:
        problems.append("HipHeight reads %s, set %s" % (vals["HipHeight"], hip_height))
    out.update(ok=not problems, verified=not problems)
    if problems:
        out["error"] = "; ".join(problems)
    return out


def _loose_same(a, b):
    """Equal for a read-back. A whole number on either side compares exactly (a data store's 1,000,000 is not 1,000,099);
    two floats may differ by one float32 rounding (relative 1e-7, so a property Studio stores as float32 still matches)."""
    if _num(a) and _num(b):
        if isinstance(a, int) or isinstance(b, int):
            return a == b
        return abs(a - b) <= max(1e-7 * max(abs(a), abs(b)), 1e-12)
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_loose_same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_loose_same(x, y) for x, y in zip(a, b))
    return a == b


# ================================================================ batch 11: audio
NEVER_SET = {"Sound": ("Playing", "PlayOnRemove"), "AudioPlayer": ("AutoPlay",)}


def _no_playback(cls, props):
    bad = [p for p in NEVER_SET.get(cls, ()) if (props or {}).get(p)]
    if bad:
        return _no("%s would play %s aloud; MifBridge never plays sound on the person's machine (read IsPlaying, "
                   "TimeLength and levels instead)" % (", ".join(bad), cls), refusedRule="audible")
    return None


@T._reports
@T.ends_group_on_raise
def rbx_audio_graph(nodes: dict, wires: list, name: str = "MifAudioGraph", parent: Any = SCRATCH, replace: bool = False) -> dict:
    "Roblox Studio: the wired audio objects in one call and one undo step. nodes maps a name to {class, props} (AudioPlayer, AudioEmitter, AudioListener, AudioDeviceOutput, AudioReverb, AudioEqualizer, AudioAnalyzer... anything with GetInputPins/GetOutputPins); wires is a list of {from, to, from_pin?, to_pin?}. The nodes are made in a Folder, each wire's pins checked against the source's GetOutputPins and the target's GetInputPins (a pin the instance does not list cancels everything; with one pin listed it is used when none is given), then a Wire per pair with SourceInstance, TargetInstance, SourceName, TargetName. Read back: every Wire.Connected true, and each wire listed by the source's GetConnectedWires(pin). Nothing plays."
    cat, err = T._catalog()
    if err:
        return err
    if not isinstance(nodes, dict) or not nodes or not isinstance(wires, list):
        return _no("nodes is {name: {class, props}} and wires a list of {from, to, from_pin?, to_pin?}")
    for n, spec in nodes.items():
        if not isinstance(n, str) or not n or "." in n or not isinstance(spec, dict):
            return _no("node %r is a name without dots mapped to {class, props}" % n)
        cls = spec.get("class")
        if cls not in cat.classes:
            return cat.unknown_class(cls)
        if cat.find_member(cls, "GetOutputPins")[1] is None:
            return _no("%s has no pins (GetInputPins/GetOutputPins): it cannot be wired" % cls, refusedRule="type")
        bad = _no_playback(cls, spec.get("props"))
        if bad:
            return bad
    for i, w in enumerate(wires):
        if not isinstance(w, dict) or w.get("from") not in nodes or w.get("to") not in nodes:
            return _no("wire %d is {from, to} naming nodes" % i)
    gname, err = _free_name(parent, name, replace)
    if err:
        return err
    folder = _child(parent, gname)
    g = T.begin_group("MifBridge audio graph " + gname)
    if not g.get("ok"):
        return dict(g, ok=False)

    def cancel(res):
        T.rbx_undo_group("cancel")
        return dict(res, ok=False, cancelled=True)

    f = T.rbx_create("Folder", parent, name=gname)
    if not f.get("ok"):
        return cancel(f)
    paths = {}
    for n, spec in nodes.items():
        c = T.rbx_create(spec["class"], folder, name=n, props=spec.get("props") or None)
        if not c.get("ok"):
            return cancel(dict(c, node=n))
        paths[n] = c["path"]
    pins = {}

    def pin_list(node, method):
        key = (node, method)
        if key not in pins:
            r = T.rbx_call(paths[node], method)
            pins[key] = [p for p in (r.get("value") or []) if isinstance(p, str)] if r.get("ok") else None
        return pins[key]

    plan = []
    for i, w in enumerate(wires):
        outs, ins = pin_list(w["from"], "GetOutputPins"), pin_list(w["to"], "GetInputPins")
        sp = w.get("from_pin") or (outs[0] if outs and len(outs) == 1 else None)
        tp = w.get("to_pin") or (ins[0] if ins and len(ins) == 1 else None)
        if sp is None or outs is None or sp not in outs or tp is None or ins is None or tp not in ins:
            return cancel(_no("wire %d: %s has output pins %s and %s input pins %s; asked %s -> %s" % (
                i, w["from"], outs, w["to"], ins, w.get("from_pin"), w.get("to_pin")), refusedRule="pin", ran=True))
        plan.append((w, sp, tp))
    wpaths = []
    for i, (w, sp, tp) in enumerate(plan):
        wn = "Wire_%d_%s_%s" % (i, w["from"], w["to"])
        c = T.rbx_create("Wire", folder, name=wn, props={"SourceInstance": paths[w["from"]], "TargetInstance": paths[w["to"]],
                                                         "SourceName": sp, "TargetName": tp})
        if not c.get("ok"):
            return cancel(dict(c, wire=i))
        wpaths.append((c["path"], w, sp))
    cm = T.rbx_undo_group("commit")
    if not cm.get("ok"):
        return dict(cm, ok=False)
    problems, rows = [], []
    for wp, w, sp in wpaths:
        vals, verr = _props(wp, ["Connected"])
        listed = T.rbx_call(paths[w["from"]], "GetConnectedWires", [sp]).get("value") or []
        row = {"wire": wp, "connected": (vals or {}).get("Connected"), "listedBySource": wp in listed}
        rows.append(row)
        if row["connected"] is not True:
            problems.append("%s Connected reads %s" % (wp, row["connected"]))
        if not row["listedBySource"]:
            problems.append("%s:GetConnectedWires(%s) does not list %s" % (w["from"], sp, wp))
    out = {"ok": not problems, "verified": not problems, "path": folder, "nodes": paths, "wires": rows,
           "pins": {"%s.%s" % k: v for k, v in pins.items()}, "undoSteps": 1}
    if problems:
        out["error"] = "; ".join(problems)
    return out


@T._reports
def rbx_sound(name: str = "MifSound", parent: Any = SCRATCH, props: dict = None, sound_group: dict = None,
              effects: list = None, replace: bool = False) -> dict:
    "Roblox Studio: a Sound (SoundId, Volume, Looped, RollOffMode, RollOffMinDistance, RollOffMaxDistance, PlaybackSpeed...) with an optional SoundGroup ({name, Volume, parent}) it is routed to and SoundEffects under it ([{class: ReverbSoundEffect|EqualizerSoundEffect|..., props}]), in one undo step. Read back: each property of each object. Playing and PlayOnRemove are refused: nothing is played aloud."
    cat, err = T._catalog()
    if err:
        return err
    bad = _no_playback("Sound", props)
    if bad:
        return bad
    sname, err = _free_name(parent, name, replace)
    if err:
        return err
    spath = _child(parent, sname)
    ops, reads = [], []
    sprops = dict(props or {})
    if sound_group:
        if not isinstance(sound_group, dict) or not isinstance(sound_group.get("name"), str):
            return _no("sound_group is {name, Volume?, parent?}")
        gparent = sound_group.get("parent", parent)
        gpath = _child(gparent, sound_group["name"])
        if not _exists(gpath):
            gprops = {k: v for k, v in sound_group.items() if k not in ("name", "parent")}
            ops.append({"tool": "rbx_create", "args": {"class_name": "SoundGroup", "parent": gparent,
                                                       "name": sound_group["name"], "props": gprops or None}})
        sprops["SoundGroup"] = gpath
        reads.append((gpath, [k for k in sound_group if k not in ("name", "parent")]))
    ops.append({"tool": "rbx_create", "args": {"class_name": "Sound", "parent": parent, "name": sname, "props": sprops or None}})
    reads.append((spath, list(sprops)))
    for i, e in enumerate(effects or []):
        if not isinstance(e, dict) or not cat.is_a(e.get("class"), "SoundEffect"):
            return _no("effect %d is {class: a SoundEffect class, props}" % i)
        en = e.get("name") or "%s_%d" % (e["class"], i)
        ops.append({"tool": "rbx_create", "args": {"class_name": e["class"], "parent": spath, "name": en,
                                                   "props": e.get("props") or None}})
        reads.append((_child(spath, en), list(e.get("props") or {})))
    for op in ops:
        if not op["args"]["props"]:
            op["args"].pop("props")
    built = _build(ops, "sound " + sname)
    if not built.get("ok"):
        return built
    back = {}
    for path, names in reads:
        if names:
            vals, verr = _props(path, names)
            back[_path(path)] = vals if not verr else {"error": verr.get("error")}
    return {"ok": True, "verified": True, "path": _path(spath), "objects": [_path(r.get("path")) for r in built.get("results") or [] if r.get("path")],
            "readBack": back, "undoSteps": built.get("undoSteps")}


def _curve_wire(curve, lo, hi, what):
    """{distance or angle: volume} from a caller (keys may be strings from JSON) -> (wire, clean, error)."""
    if not isinstance(curve, dict) or not 1 <= len(curve) <= 400:
        return None, None, "%s is an object of 1-400 %s: volume points" % (what, what)
    clean = {}
    for k, v in curve.items():
        try:
            x = float(k)
        except (TypeError, ValueError):
            return None, None, "%s key %r is not a number" % (what, k)
        if not lo <= x <= hi or not _num(v) or not 0 <= v <= 1:
            return None, None, "%s point %r: %s is %s-%s and volume 0-1" % (what, k, what, lo, hi)
        clean[x] = float(v)
    # The plugin's Dictionary keeps string keys; Studio's SetDistanceAttenuation and SetAngleAttenuation read "20.0" as
    # 20, so no plugin change is needed.
    return {"$t": "Dictionary", "v": {_fmt(k): v for k, v in clean.items()}}, clean, None


@T._reports
def rbx_audio_attenuation(target: Any, distance: dict = None, angle: dict = None) -> dict:
    "Roblox Studio: an AudioEmitter's or AudioListener's attenuation curves - distance {studs: volume 0-1} through SetDistanceAttenuation and angle {degrees 0-180: volume} through SetAngleAttenuation, each one undo step. Read back: GetDistanceAttenuation / GetAngleAttenuation equal to the curve sent, point for point."
    cls, err = _class_of(target)
    if err:
        return err
    if cls not in ("AudioEmitter", "AudioListener"):
        return _no("target is an AudioEmitter or AudioListener, got %s" % cls, refusedRule="type")
    if distance is None and angle is None:
        return _no("give distance and/or angle")
    cat, err = T._catalog()
    if err:
        return err
    parts = V.split_path(target)
    out, problems = {"target": _path(target)}, []
    for what, curve, lo, hi, setter, getter in (("distance", distance, 0, 1e7, "SetDistanceAttenuation", "GetDistanceAttenuation"),
                                               ("angle", angle, 0, 180, "SetAngleAttenuation", "GetAngleAttenuation")):
        if curve is None:
            continue
        wire, clean, why = _curve_wire(curve, lo, hi, what)
        if why:
            return _no(why, refusedRule="type")
        bad = cat.read_refusal(cls, setter, kind="F")
        if bad:
            return bad
        r = T._rbx("call", target=parts, method=setter, args=[wire], argc=1, readOnly=False, yields=False, callTimeout=30)
        if not r.get("ok"):
            problems.append("%s: %s" % (setter, r.get("error")))
            continue
        g = T.rbx_call(target, getter)
        got = g.get("value") if isinstance(g.get("value"), dict) else {}
        try:
            back = {float(k): v for k, v in got.items()}
        except (TypeError, ValueError):
            back = {}
        same = set(back) == set(clean) and all(abs(back[k] - clean[k]) <= 1e-4 for k in clean)
        out[what] = {"sent": {_fmt(k): v for k, v in clean.items()}, "readBack": got, "verified": same}
        if not same:
            problems.append("%s reads %s, sent %s" % (getter, got, out[what]["sent"]))
    out.update(ok=not problems, verified=not problems)
    if problems:
        out["error"] = "; ".join(problems)
    return out


# ---- audio file length on this machine, to compare with what Studio says
def _wav_seconds(b):
    if b[:4] != b"RIFF" or b[8:12] != b"WAVE":
        return None
    i, rate, block = 12, None, None
    while i + 8 <= len(b):
        cid, size = b[i:i + 4], struct.unpack("<I", b[i + 4:i + 8])[0]
        if cid == b"fmt ":
            _fmtc, ch, rate, _br, block = struct.unpack("<HHIIH", b[i + 8:i + 22])
        elif cid == b"data" and rate and block:
            return size / block / rate
        i += 8 + size + (size & 1)
    return None


def _ogg_seconds(b):
    if b[:4] != b"OggS":
        return None
    j = b.find(b"\x01vorbis")
    if j < 0:
        return None
    rate = struct.unpack("<I", b[j + 12:j + 16])[0]
    last = b.rfind(b"OggS")
    granule = struct.unpack("<q", b[last + 6:last + 14])[0]
    return granule / rate if rate and granule >= 0 else None


_MP3_RATES = {3: (44100, 48000, 32000), 2: (22050, 24000, 16000), 0: (11025, 12000, 8000)}
_MP3_BITRATES = {(3, 1): (0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320),
                 (2, 1): (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160)}


def _mp3_seconds(b):
    i = 0
    if b[:3] == b"ID3":
        i = 10 + ((b[6] & 0x7f) << 21 | (b[7] & 0x7f) << 14 | (b[8] & 0x7f) << 7 | (b[9] & 0x7f))
    total, frames = 0.0, 0
    while i + 4 <= len(b):
        h = struct.unpack(">I", b[i:i + 4])[0]
        if h >> 21 != 0x7ff:
            i += 1
            continue
        ver, layer, bri, sri, pad = (h >> 19) & 3, (h >> 17) & 3, (h >> 12) & 15, (h >> 10) & 3, (h >> 9) & 1
        if ver == 1 or layer != 1 or bri in (0, 15) or sri == 3:
            i += 1
            continue
        rate = _MP3_RATES[ver][sri]
        br = _MP3_BITRATES[(3 if ver == 3 else 2, 1)][bri] * 1000
        samples = 1152 if ver == 3 else 576
        size = samples // 8 * br // rate + pad
        if size <= 4:
            i += 1
            continue
        total += samples / rate
        frames += 1
        i += size
    return total if frames else None


def audio_file_seconds(path):
    """(seconds, None) or (None, why) for a .wav, .ogg (Vorbis) or .mp3 file."""
    try:
        with open(path, "rb") as f:
            b = f.read()
    except OSError as exc:
        return None, str(exc)
    ext = os.path.splitext(path)[1].lower()
    s = {".wav": _wav_seconds, ".ogg": _ogg_seconds, ".mp3": _mp3_seconds}.get(ext, lambda _b: None)(b)
    return (s, None) if s else (None, "could not read a length from %s" % path)


def _spectrum_summary(value):
    """AudioAnalyzer:GetSpectrum's answer -> {bins, peakBin, peak, sum} (the codec may cut a long list: bins is its total)."""
    items = value.get("items") if isinstance(value, dict) else value
    if not isinstance(items, list):
        return {"error": "GetSpectrum answered %r" % (value,)}
    nums = [x for x in items if _num(x)]
    total = value.get("total", len(items)) if isinstance(value, dict) else len(items)
    if not nums:
        return {"bins": total, "peakBin": None, "peak": None, "sum": 0}
    k = max(range(len(nums)), key=lambda i: nums[i])
    return {"bins": total, "binsRead": len(nums), "peakBin": k, "peak": nums[k], "sum": round(sum(nums), 9)}


def _analyzer_reads(analyzer, seconds, interval=0.25, player=None):
    rows, t0 = [], time.time()
    while True:
        lv, err = _props(analyzer, ["RmsLevel", "PeakLevel"])
        if err:
            return rows, err
        row = {"t": round(time.time() - t0, 2), **lv}
        if player:
            pv, _e = _props(player, ["TimePosition", "IsPlaying"])
            row.update(pv or {})
        sp = T.rbx_call(analyzer, "GetSpectrum", [])
        row["spectrum"] = _spectrum_summary(sp.get("value")) if sp.get("ok") else {"error": sp.get("error")}
        rows.append(row)
        if time.time() - t0 >= seconds:
            return rows, None
        time.sleep(interval)


def _levels(cls, vals, seconds):
    """The clip played silently into an analyzer: a scratch AudioPlayer with the target's asset wired ONLY to an
    AudioAnalyzer (no AudioDeviceOutput, no AudioEmitter, so nothing reaches a speaker), played, read, stopped, removed.
    The target itself is never played. Measured on 0.748: the analyzer reads levels in edit mode, no play session."""
    asset = vals.get("SoundId") if cls == "Sound" else vals.get("Asset")
    if not isinstance(asset, str) or not asset:
        return _no("the target has no asset to read levels from")
    g = rbx_audio_graph({"Player": {"class": "AudioPlayer", "props": {"Asset": asset}},
                         "Analyzer": {"class": "AudioAnalyzer", "props": {"SpectrumEnabled": True}}},
                        [{"from": "Player", "to": "Analyzer"}], name="MifAudioLevels")
    if not g.get("ok"):
        return dict(g, ok=False, step="graph")
    player, analyzer = g["nodes"]["Player"], g["nodes"]["Analyzer"]
    try:
        wires = T.rbx_call(player, "GetConnectedWires", ["Output"])
        n = len(wires.get("value") or []) if wires.get("ok") else None
        if n != 1:
            return _no("the scratch player's Output has %s wires, not only the analyzer's: refused before playing" % n,
                       refusedRule="audible")
        deadline = time.time() + 15
        while True:
            rv, _e = _props(player, ["IsReady"])
            if (rv or {}).get("IsReady") or time.time() >= deadline:
                break
            time.sleep(0.25)
        idle, err = _analyzer_reads(analyzer, 0)
        if err:
            return err
        pl = T.rbx_call(player, "Play", [])
        if not pl.get("ok"):
            return dict(pl, ok=False, step="play")
        rows, err = _analyzer_reads(analyzer, seconds, player=player)
        T.rbx_call(player, "Stop", [])
        if err:
            return err
        rms = [r["RmsLevel"] for r in rows if _num(r.get("RmsLevel"))]
        peak = [r["PeakLevel"] for r in rows if _num(r.get("PeakLevel"))]
        heard = bool(rms) and max(rms) > 0
        return {"ok": heard, "asset": asset, "silentRoute": "AudioPlayer -> AudioAnalyzer only", "idle": idle[0] if idle else None,
                "reads": rows, "rmsMax": max(rms) if rms else None, "rmsMean": round(sum(rms) / len(rms), 6) if rms else None,
                "peakMax": max(peak) if peak else None,
                **({} if heard else {"error": "the analyzer read no level while the clip played"})}
    finally:
        T.rbx_destroy(g["path"])


def rbx_audio_inspect(target: Any, waveform_samples: int = 0, file: str = None, levels_seconds: float = 0) -> dict:
    "Roblox Studio: what a Sound or AudioPlayer holds - AudioPlayer.IsReady and TimeLength (and AudioPlayer:GetWaveformAsync over the whole clip with waveform_samples points), or Sound.IsLoaded and TimeLength. With file, the clip's length on this machine (.wav, .ogg, .mp3) is compared with Studio's within 10 ms. With levels_seconds (0.5-30), AudioAnalyzer RmsLevel, PeakLevel and GetSpectrum are read every 0.25 s while the clip plays SILENTLY: a scratch AudioPlayer with the target's asset is wired only to an AudioAnalyzer (nothing reaches a speaker), played in edit mode, stopped and removed; the target is never played. An AudioAnalyzer target answers its current levels and spectrum, nothing played."
    cls, err = _class_of(target)
    if err:
        return err
    if cls == "AudioAnalyzer":
        rows, err = _analyzer_reads(target, 0)
        return err or {"ok": True, "target": _path(target), "className": cls, **rows[0]}
    if cls not in ("Sound", "AudioPlayer"):
        return _no("target is a Sound, AudioPlayer or AudioAnalyzer, got %s" % cls, refusedRule="type")
    if levels_seconds and (not _num(levels_seconds) or not 0.5 <= levels_seconds <= 30):
        return _no("levels_seconds is 0.5-30")
    names = ["IsLoaded", "TimeLength", "SoundId", "Looped", "Volume"] if cls == "Sound" else \
        ["IsReady", "TimeLength", "Asset", "Looping", "Volume"]
    vals, err = _props(target, names)
    if err:
        return err
    out = {"ok": True, "target": _path(target), "className": cls, "values": vals}
    if waveform_samples:
        if cls != "AudioPlayer" or not isinstance(waveform_samples, int) or not 1 <= waveform_samples <= 2000:
            return _no("waveform_samples (1-2000) reads an AudioPlayer")
        tl = vals.get("TimeLength") or 0
        wv = T.rbx_call(target, "GetWaveformAsync", [[0, tl], waveform_samples], timeout=60)
        out["waveform"] = wv.get("value") if wv.get("ok") else {"error": wv.get("error")}
    if file:
        secs, why = audio_file_seconds(file)
        out["fileSeconds"] = secs
        if secs is None:
            return dict(out, ok=False, error=why)
        studio = vals.get("TimeLength")
        out["differenceMs"] = round(abs(secs - studio) * 1000, 2) if _num(studio) else None
        if out["differenceMs"] is None or out["differenceMs"] > 10:
            out.update(ok=False, error="Studio's TimeLength %s and the file's %.4f s differ by more than 10 ms" % (studio, secs))
    if levels_seconds:
        lv = _levels(cls, vals, levels_seconds)
        out["levels"] = lv
        if not lv.get("ok"):
            out.update(ok=False, error=out.get("error") or "levels: %s" % lv.get("error"))
    return out


def rbx_audio_upload(file: str, display_name: str, confirm: bool = False, wait_seconds: float = 30) -> dict:
    "Roblox: upload an audio file (.mp3, .ogg, .wav, .flac) as an Audio asset through Open Cloud (rbx_asset_upload: permanent and moderated, confirm true, the panel in full mode), then load it in a scratch Sound (Workspace.MifScratch) and wait for Sound.IsLoaded. Read back: IsLoaded true and TimeLength within 10 ms of the file's length. Nothing is played. Once the upload is done every answer, a failed load check too, says uploaded true with the assetId and a loadCheck block: the asset exists, so re-check it with rbx_sound and rbx_audio_inspect rather than upload it again."
    if confirm is not True:
        return _no("an upload makes a permanent, moderated asset: pass confirm true once the person agreed", refusedRule="needs-confirm")
    secs, why = audio_file_seconds(file) if isinstance(file, str) else (None, "file is a path")
    up = T.rbx_asset_upload(file, display_name, confirm=True, asset_type="Audio")
    if not up.get("ok"):
        return dict(up, step="upload")
    # From here the asset exists for good (permanent, moderated, counted against the audio quota): every answer below says
    # uploaded true with its id, so a caller re-checks it instead of uploading it again.
    aid = up.get("assetId")
    out = {"uploaded": True, "assetId": aid, "moderationState": up.get("moderationState"), "fileSeconds": secs}
    recheck = ("the audio asset %s exists (uploaded, permanent): do NOT upload it again. Re-check it later with "
               "rbx_sound(props={\"SoundId\": \"rbxassetid://%s\"}) then rbx_audio_inspect on that Sound (a new asset is "
               "often still in moderation)" % (aid, aid))
    s = rbx_sound(name="MifUploadedSound", props={"SoundId": "rbxassetid://%s" % aid}, replace=True)
    if not s.get("ok"):
        out.update(ok=False, verified=False, step="sound",
                   loadCheck={"ran": False, "error": s.get("error"), "refusedRule": s.get("refusedRule")},
                   error="the load check did not run (the scratch Sound failed: %s); %s" % (s.get("error"), recheck))
        return out
    t0, vals = time.time(), {}
    while time.time() - t0 < wait_seconds:
        vals, _e = _props(s["path"], ["IsLoaded", "TimeLength"])
        if (vals or {}).get("IsLoaded"):
            break
        time.sleep(1)
    check = {"ran": True, "sound": s["path"], "isLoaded": (vals or {}).get("IsLoaded"),
             "timeLength": (vals or {}).get("TimeLength"), "waitedSeconds": round(time.time() - t0, 2)}
    good = check["isLoaded"] is True and (secs is None or (_num(check["timeLength"]) and abs(check["timeLength"] - secs) <= 0.01))
    check["ok"] = good
    # the load check's fields stay at the top level too (sound, isLoaded, timeLength), as before
    out.update(sound=s["path"], isLoaded=check["isLoaded"], timeLength=check["timeLength"], loadCheck=check,
               ok=good, verified=good)
    if not good:
        out["error"] = "%s; %s" % (why if secs is None and why else
                                   "the uploaded clip did not load, or its length differs from the file's", recheck)
    return out


# ---------------------------------------------------------------- licensed sound effects
# FOUND ON A GAME (2026-11-21): weather loops and a thunder crack were found by calling the toolbox API by hand - plain
# keywords on the audio category bury the licensed library under user uploads, so the creator filter is what finds it.
# Routes, all public reads with no key and no cookie (measured 2026-11-21, RB1920):
#   users.roblox.com POST /v1/usernames/users {usernames}                  a creator's name -> user id
#   apis.roblox.com GET /toolbox-service/v1/marketplace/3?keyword=&limit=&includeOnlySoundEffects=true
#                   &creatorType=1&creatorTargetId=<id>&cursor=           category 3 is audio; the creator filter is
#                                                                          applied by the server; ids only, a cursor
#   apis.roblox.com GET /toolbox-service/v1/items/details?assetIds=<up to 50>   title, duration (whole seconds),
#                                                                          audioType, creator, free
SOUND_SEARCH = "https://apis.roblox.com/toolbox-service/v1/marketplace/3"
SOUND_DETAILS = "https://apis.roblox.com/toolbox-service/v1/items/details"
USERS_BY_NAME = "https://users.roblox.com/v1/usernames/users"
LICENSED_SFX = "ProSoundEffects"          # the licensed sound-effect library on the Creator Store (L89)


def _public(method, url, body=None):
    """One public Roblox read (no key is ever sent): (status, json or None, error or None)."""
    from mif_roblox import opencloud as OC
    headers = {"Accept": "application/json"}
    data = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(body).encode("utf-8")
    try:
        st, _h, raw = OC.REQUEST(method, url, headers, data, 30)
    except OSError as exc:
        return 0, None, "%s %s did not answer: %s" % (method, url.split("?")[0], exc)
    out = OC._json(raw)
    if st != 200:
        return st, out, "%s %s answered %s: %s" % (method, url.split("?")[0], st, json.dumps(out)[:200])
    return st, out, None


def _creator_id(creator):
    """A user name or id -> (user id, name, None) or (None, None, error)."""
    if isinstance(creator, int) and not isinstance(creator, bool) and creator > 0:
        return creator, None, None
    if not isinstance(creator, str) or not creator.strip():
        return None, None, "creator is a Roblox user name or a positive user id"
    if creator.strip().isdigit():
        return int(creator.strip()), None, None
    _st, out, err = _public("POST", USERS_BY_NAME, {"usernames": [creator.strip()], "excludeBannedUsers": True})
    if err:
        return None, None, err
    hit = next((u for u in (out or {}).get("data") or [] if str(u.get("name", "")).lower() == creator.strip().lower()), None)
    if not hit or not isinstance(hit.get("id"), int):
        return None, None, "no Roblox user is named %r (the toolbox filter takes a user; a group's id goes in as a number " \
                           "with creator_type group)" % creator
    return hit["id"], hit.get("name"), None


def _sound_rows(ids):
    """Toolbox details for ids, 50 a call -> (rows in the order of ids, error or None)."""
    by_id = {}
    for i in range(0, len(ids), 50):
        chunk = ids[i:i + 50]
        _st, out, err = _public("GET", "%s?assetIds=%s" % (SOUND_DETAILS, ",".join(str(x) for x in chunk)))
        if err:
            return None, err
        for d in (out or {}).get("data") or []:
            a, c = d.get("asset") or {}, d.get("creator") or {}
            fiat = d.get("fiatProduct") or {}
            if not isinstance(a.get("id"), int):
                continue
            by_id[a["id"]] = {"id": a["id"], "title": a.get("name"),
                              "audioTitle": (a.get("audioDetails") or {}).get("title"),
                              "durationSeconds": a.get("duration"),
                              "audioType": (a.get("audioDetails") or {}).get("audioType"), "typeId": a.get("typeId"),
                              "creator": {"id": c.get("id"), "name": c.get("name"), "type": c.get("type")},
                              "free": fiat.get("isFree")}
    return [by_id[x] for x in ids if x in by_id], None


def sound_preload_snippet(rows):
    """Luau for a client self-test: preload every id, then require IsLoaded and TimeLength > 0 of each (L89)."""
    lines = ["-- MifBridge sound self-test: every sound id the game uses must load.",
             "-- Put it in a LocalScript (StarterPlayerScripts) for a test run; it prints one PASS or names each failure.",
             "local ContentProvider = game:GetService(\"ContentProvider\")",
             "local SOUNDS = {"]
    for r in rows:
        # a raw CR or LF inside a quoted Luau string is a malformed string: both become a space. Cut to 80 before
        # escaping, so the cut never splits an escape and leaves a backslash before the closing quote.
        title = str(r.get("title") or "")[:80].replace("\\", "\\\\").replace("\"", "\\\"").replace("\r", " ").replace("\n", " ")
        lines.append("\t{id = %d, title = \"%s\"}," % (int(r["id"]), title))
    lines += ["}",
              "local made, failed = {}, {}",
              "for _, s in ipairs(SOUNDS) do",
              "\tlocal snd = Instance.new(\"Sound\")",
              "\tsnd.SoundId = \"rbxassetid://\" .. s.id",
              "\tsnd.Name = \"SelfTest_\" .. s.id",
              "\tsnd.Parent = script",
              "\ttable.insert(made, snd)",
              "end",
              "ContentProvider:PreloadAsync(made, function(contentId, status)",
              "\tif status ~= Enum.AssetFetchStatus.Success then failed[contentId] = status.Name end",
              "end)",
              "local bad = 0",
              "for i, snd in ipairs(made) do",
              "\tlocal s = SOUNDS[i]",
              "\tif failed[snd.SoundId] or not snd.IsLoaded or snd.TimeLength <= 0 then",
              "\t\tbad += 1",
              "\t\twarn((\"SOUND FAIL %d %s: fetch %s, IsLoaded %s, TimeLength %.2f\"):format(s.id, s.title,",
              "\t\t\ttostring(failed[snd.SoundId] or \"Success\"), tostring(snd.IsLoaded), snd.TimeLength))",
              "\tend",
              "\tsnd:Destroy()",
              "end",
              "print((\"SOUND SELF-TEST %s: %d of %d loaded\"):format(bad == 0 and \"PASS\" or \"FAIL\", #made - bad, #made))"]
    return "\n".join(lines) + "\n"


def _check_loaded(rows, wait_seconds):
    """Each id in a scratch Sound under Workspace.MifScratch in this Studio: IsLoaded and TimeLength, then removed.
    Nothing is played (rbx_sound refuses Playing)."""
    made, res = [], []
    for r in rows:
        s = rbx_sound(name="MifSfxCheck_%d" % r["id"], props={"SoundId": "rbxassetid://%d" % r["id"]}, replace=True)
        if not s.get("ok"):
            for p in made:
                T.rbx_destroy(p)
            return None, dict(s, step="scratch Sound for %d" % r["id"])
        made.append(s["path"])
    t0 = time.time()
    vals = {}
    while True:
        for p in made:
            if not (vals.get(p) or {}).get("IsLoaded"):
                vals[p] = _props(p, ["IsLoaded", "TimeLength"])[0] or {}
        if all(v.get("IsLoaded") for v in vals.values()) or time.time() - t0 >= wait_seconds:
            break
        time.sleep(0.5)
    for r, p in zip(rows, made):
        v = vals.get(p) or {}
        tl = v.get("TimeLength")
        res.append({"id": r["id"], "isLoaded": v.get("IsLoaded") is True, "timeLength": tl,
                    "loads": v.get("IsLoaded") is True and _num(tl) and tl > 0})
        T.rbx_destroy(p)
    return {"seconds": round(time.time() - t0, 2), "rows": res,
            "notLoaded": [x["id"] for x in res if not x["loads"]]}, None


def rbx_sound_search(keyword: str = None, creator: Any = LICENSED_SFX, limit: int = 30, ids: list = None,
                     creator_type: str = "user", check_loaded: bool = False, wait_seconds: float = 15) -> dict:
    "Roblox Creator Store: find sound effects by keyword from one creator - by default ProSoundEffects, the licensed sound-effect library - through Roblox's public toolbox API (no key, no cookie, read only: nothing is bought, inserted or uploaded). The search is audio (category 3) with includeOnlySoundEffects and the creator filter applied by Roblox (creatorTargetId; a user name is turned into its id by users.roblox.com), then each id's details: id, title, durationSeconds (whole seconds, as the toolbox gives it), audioType, creator, free. Every row is checked to be the creator's; creator None searches everyone (user uploads bury the library). ids [..] instead of keyword reads those ids' details. limit 1-200. Each answer carries selfTest: a client LocalScript that preloads every id with ContentProvider:PreloadAsync and prints PASS, or warns each id whose fetch failed, IsLoaded is false or TimeLength is 0 - put it in the game's test run. check_loaded true also loads each id (up to 50) in a scratch Sound in Workspace.MifScratch of this Studio and waits up to wait_seconds for Sound.IsLoaded and TimeLength above 0, then removes them; nothing is played. Nothing found fails: an empty answer is not a search that found nothing to worry about."
    if keyword is None and ids is None:
        return _no("give keyword (words to search) or ids (asset ids to read)")
    if keyword is not None and ids is not None:
        return _no("keyword searches; ids reads given ids - pass one")
    if not (isinstance(limit, int) and not isinstance(limit, bool) and 1 <= limit <= 200):
        return _no("limit is a whole number 1-200")
    if creator_type not in ("user", "group"):
        return _no("creator_type is user or group")
    if not isinstance(check_loaded, bool) or not (_num(wait_seconds) and 1 <= wait_seconds <= 120):
        return _no("check_loaded is true or false; wait_seconds 1-120")
    out = {"route": "public toolbox API, no key or cookie (apis.roblox.com/toolbox-service/v1)", "readOnly": True}
    cid, cname = None, None
    if creator is not None:
        if creator_type == "group":
            if not (isinstance(creator, int) and not isinstance(creator, bool) and creator > 0):
                return _no("a group creator is its numeric group id")
            cid = creator
        else:
            cid, cname, err = _creator_id(creator)
            if err:
                return _no(err)
        out["creator"] = {"id": cid, "name": cname or (creator if isinstance(creator, str) else None), "type": creator_type}
    if ids is not None:
        if not isinstance(ids, list) or not ids or not all(isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in ids):
            return _no("ids is a non-empty list of positive asset ids")
        found = list(dict.fromkeys(ids))[:limit]
        out["asked"] = len(found)
    else:
        if not isinstance(keyword, str) or not keyword.strip():
            return _no("keyword is the words to search")
        found, cursor, total, pages = [], None, None, 0
        while len(found) < limit and pages < 10:
            q = {"keyword": keyword.strip(), "limit": min(100, max(10, limit)), "includeOnlySoundEffects": "true"}
            if cid is not None:
                q.update(creatorType=1 if creator_type == "user" else 2, creatorTargetId=cid)
            if cursor:
                q["cursor"] = cursor
            _st, page, err = _public("GET", SOUND_SEARCH + "?" + urllib.parse.urlencode(q))
            pages += 1
            if err:
                return _no("the toolbox search failed: %s" % err, **out)
            total = page.get("totalResults") if total is None else total
            for d in page.get("data") or []:
                if isinstance(d, dict) and isinstance(d.get("id"), int) and d["id"] not in found:
                    found.append(d["id"])
            cursor = page.get("nextPageCursor")
            if not cursor or not page.get("data"):
                break
        found = found[:limit]
        out.update(keyword=keyword.strip(), totalResults=total, pages=pages)
    rows, err = _sound_rows(found) if found else ([], None)
    if err:
        return _no("the toolbox details read failed: %s" % err, **out)
    if cid is not None:
        other = [r for r in rows if r["creator"]["id"] != cid]
        if other:
            out["droppedNotCreator"] = [{"id": r["id"], "creator": r["creator"]} for r in other][:20]
        rows = [r for r in rows if r["creator"]["id"] == cid]
    missing = [x for x in found if x not in {r["id"] for r in rows}]
    if ids is not None and missing:
        out["notFound"] = missing
    out.update(count=len(rows), sounds=rows)
    if not rows:
        return dict(out, ok=False, error="no sound matched%s: nothing found is not an answer to build on - try other "
                                         "words%s" % (" %r" % keyword if keyword else "", " or creator None" if cid else ""))
    out["selfTest"] = sound_preload_snippet(rows)
    out["ok"] = True
    if check_loaded:
        if len(rows) > 50:
            return dict(out, ok=False, error="check_loaded loads at most 50 sounds in Studio; pass a smaller limit")
        chk, err = _check_loaded(rows, wait_seconds)
        if err:
            return dict(out, ok=False, loaded=err, error="the Studio load check failed: %s" % err.get("error"))
        out["loaded"] = chk
        if chk["notLoaded"]:
            out.update(ok=False, error="%d of %d sounds did not load in Studio within %g s: %s"
                       % (len(chk["notLoaded"]), len(rows), wait_seconds, chk["notLoaded"][:10]))
    return out


# ================================================================ batch 13: data and gameplay services (5.11, 5.17)
# Studio's own words when the place cannot reach its data: measured live and recorded in the internal notes; until then these
# patterns name the switch, and any other failure is passed through as Studio said it.
SETTING_OFF = re.compile(r"StudioAccessToApis|API Services|access to API|publish(ed)? (this )?(game|place)|"
                         r"must publish|not published|universeId", re.I)
SETTING_STEPS = ["Publish the place (data stores belong to a published experience).",
                 "Game Settings > Security > turn on 'Enable Studio Access to API Services', then Save."]


def _ds_error(r, step):
    msg = str(r.get("error") or "")
    if SETTING_OFF.search(msg):
        return dict(r, ok=False, refusedRule="setting-off", steps=SETTING_STEPS, step=step,
                    error="the place cannot reach its data stores from Studio: %s" % msg)
    return dict(r, ok=False, step=step)


def _store(store, scope="global", ordered=False):
    """(DataStore handle, error)."""
    if not isinstance(store, str) or not store or not isinstance(scope, str):
        return None, _no("store is a data store name and scope a string")
    r = T.rbx_call("DataStoreService", "GetOrderedDataStore" if ordered else "GetDataStore", [store, scope], timeout=30)
    h = _handle(r.get("value"))
    if not h:
        return None, _ds_error(r if not r.get("ok") else dict(r, error="GetDataStore answered %r" % (r.get("value"),)), "GetDataStore")
    return h, None


def _pages(pages_value, item_props, limit):
    """Walk a Pages handle: each item read (an Instance handle's props, or a plain value) until limit or IsFinished."""
    ph = _handle(pages_value)
    if not ph:
        return None, _no("expected a Pages object, got %r" % (pages_value,))
    items, more = [], True
    try:
        while len(items) < limit:
            cur = T.rbx_call({"handle": ph}, "GetCurrentPage")
            if not cur.get("ok"):
                return None, cur
            page = cur.get("value")
            if isinstance(page, dict) and isinstance(page.get("items"), list):
                page = page["items"]
            page = page if isinstance(page, list) else []
            left = False
            for i, it in enumerate(page):
                if len(items) >= limit:
                    # limit reached inside this page: the rest are items there are more of, and their handles are
                    # released here, since nothing else will read them
                    left = True
                    _release(*[_handle(x) for x in page[i:]])
                    break
                h = _handle(it)
                if h:
                    vals, err = _props({"handle": h}, item_props)
                    _release(h)
                    items.append(vals if not err else {"error": err.get("error")})
                else:
                    items.append(it)
            fin = T._rbx("get_props", target={"handle": ph}, props=["IsFinished"])
            more = left or not V.decode(T._obj(fin.get("values")).get("IsFinished"))
            if not more or len(items) >= limit:
                break
            adv = T.rbx_call({"handle": ph}, "AdvanceToNextPageAsync", timeout=30)
            if not adv.get("ok"):
                return None, adv
        return {"items": items, "more": more}, None
    finally:
        _release(ph)


def _no_handles(value):
    """A returned value with each handle in it released and named by its class (IncrementAsync answers its new value and a
    DataStoreKeyInfo handle; live 2026-11-21): a write's answer never leaves a handle behind in the plugin."""
    if isinstance(value, list):
        return [_no_handles(v) for v in value]
    h = _handle(value)
    if h:
        _release(h)
        return {"released": value.get("className") or "Object"}
    return value


def _key_info(h):
    vals, err = _props({"handle": h}, ["Version", "CreatedTime", "UpdatedTime"])
    md = T.rbx_call({"handle": h}, "GetMetadata")
    uids = T.rbx_call({"handle": h}, "GetUserIds")
    _release(h)
    if err:
        return {"error": err.get("error")}
    return dict(vals, metadata=md.get("value"), userIds=uids.get("value"))


def _get(store_h, key):
    r = T.rbx_call({"handle": store_h}, "GetAsync", [key], timeout=30)
    if not r.get("ok"):
        return None, _ds_error(r, "GetAsync")
    vals = r.get("values") or [r.get("value")]
    info = _key_info(_handle(vals[1])) if len(vals) > 1 and _handle(vals[1]) else None
    return {"value": vals[0] if vals else None, "keyInfo": info}, None


def rbx_datastore_read(store: str = None, key: str = None, scope: str = "global", prefix: str = "", versions: bool = False,
                       version: str = None, ordered: bool = False, limit: int = 50) -> dict:
    "Roblox Studio: read the place's data stores - no store: the stores (DataStoreService:ListDataStoresAsync); a store and no key: its keys (DataStore:ListKeysAsync, or for ordered an OrderedDataStore:GetSortedAsync page); a key: its value and DataStoreKeyInfo (GlobalDataStore:GetAsync: Version, CreatedTime, UpdatedTime, metadata, user ids); with versions, its versions (DataStore:ListVersionsAsync); with version, that version's value (DataStore:GetVersionAsync). Paged up to limit. Refused setting-off, with the steps, when the place cannot reach its data from Studio. Read only: reads are never gated."
    if not isinstance(limit, int) or not 1 <= limit <= 500:
        return _no("limit is 1-500")
    if store is None:
        r = T.rbx_call("DataStoreService", "ListDataStoresAsync", [prefix or "", min(limit, 100)], timeout=60)
        if not r.get("ok"):
            return _ds_error(r, "ListDataStoresAsync")
        pg, err = _pages(r.get("value"), ["DataStoreName", "CreatedTime", "UpdatedTime"], limit)
        return err or {"ok": True, "stores": pg["items"], "more": pg["more"]}
    h, err = _store(store, scope, ordered)
    if err:
        return err
    try:
        if key is None:
            if ordered:
                r = T.rbx_call({"handle": h}, "GetSortedAsync", [False, min(limit, 100), None, None], timeout=60)
                if not r.get("ok"):
                    return _ds_error(r, "GetSortedAsync")
                pg, err = _pages(r.get("value"), [], limit)
                return err or {"ok": True, "store": store, "ordered": True, "entries": pg["items"], "more": pg["more"]}
            r = T.rbx_call({"handle": h}, "ListKeysAsync", [prefix or "", min(limit, 100)], timeout=60)
            if not r.get("ok"):
                return _ds_error(r, "ListKeysAsync")
            pg, err = _pages(r.get("value"), ["KeyName"], limit)
            return err or {"ok": True, "store": store, "scope": scope, "keys": [x.get("KeyName") for x in pg["items"]],
                           "more": pg["more"]}
        if not isinstance(key, str) or not key:
            return _no("key is a string")
        if version is not None:
            r = T.rbx_call({"handle": h}, "GetVersionAsync", [key, version], timeout=30)
            if not r.get("ok"):
                return _ds_error(r, "GetVersionAsync")
            vals = r.get("values") or [r.get("value")]
            info = _key_info(_handle(vals[1])) if len(vals) > 1 and _handle(vals[1]) else None
            return {"ok": True, "store": store, "key": key, "version": version, "value": vals[0], "keyInfo": info}
        got, err = _get(h, key)
        if err:
            return err
        out = {"ok": True, "store": store, "scope": scope, "key": key, **got}
        if versions:
            r = T.rbx_call({"handle": h}, "ListVersionsAsync", [key, "Descending", 0, 0, min(limit, 100)], timeout=60)
            if not r.get("ok"):
                return _ds_error(r, "ListVersionsAsync")
            pg, err = _pages(r.get("value"), ["Version", "CreatedTime", "IsDeleted"], limit)
            if err:
                return err
            out.update(versions=pg["items"], moreVersions=pg["more"])
        return out
    finally:
        _release(h)


def _live_gate(allow_live_write):
    if allow_live_write is not True:
        return _no("a data store write reaches the live game's data: pass allow_live_write true for this call "
                   "(decision 5)", refusedRule="live-data")
    mode, _st = T._plugin_mode()
    if mode != "full":
        return _no("a data store write needs the MifBridge panel in full mode (it is %s): the person raises it in the panel, or rbx_set_mode where their policy allows"
                   % mode, refusedRule="full-mode", mode=mode)
    return None


DS_CALLS = {"set": "SetAsync", "increment": "IncrementAsync", "remove": "RemoveAsync"}


def ds_proved(action, before, after):
    """rbx_datastore_write's proved list (COVERAGE_PLAN 7.2 #6, a gated tool lists what it proved): the GlobalDataStore
    write, read back by GetAsync, only when the key's value changed - a set of the value already there, or a remove of
    a missing key, proves nothing. The op is the rbx_call that made the write."""
    cat, _err = T._catalog()
    if cat is None or action not in DS_CALLS or _loose_same(before, after):
        return []
    return T._proved(cat, "GlobalDataStore", [DS_CALLS[action]], "write", "call")


@T._reports
def rbx_datastore_write(store: str, key: str, action: str, value: Any = None, delta: int = 1, scope: str = "global",
                        allow_live_write: bool = False, allow_ids: list = None) -> dict:
    "Roblox Studio: write one data store key - action set (GlobalDataStore:SetAsync with value), increment (IncrementAsync by delta, a whole number) or remove (RemoveAsync). Studio's writes reach the live game's data, so this is refused unless the call passes allow_live_write: true AND the MifBridge panel is in full mode (decision 5), AND the open place's experience (game.GameId) is on the person's write-target list (roblox_write_targets.json) or named in allow_ids (refusedRule write-target). Read back: GlobalDataStore:GetAsync after the write - the value sent, the old value plus delta, or nothing."
    if action not in ("set", "increment", "remove"):
        return _no("action is set, increment or remove")
    if not isinstance(key, str) or not key or not isinstance(store, str) or not store:
        return _no("store and key are non-empty strings")
    if action == "increment" and (not isinstance(delta, int) or isinstance(delta, bool)):
        return _no("delta is a whole number")
    if action == "set":
        _w, why = V.encode("Variant", value)
        if why:
            return _no("value: %s" % why, refusedRule="type")
    gate = _live_gate(allow_live_write)
    if gate:
        return gate
    from mif_roblox import opencloud as OC
    g = T._rbx("get_props", target=[], props=["GameId"])
    gid = (g.get("values") or {}).get("GameId") if g.get("ok") else None
    gate = OC.write_target_check(gid if gid else None, False, allow_ids)
    if gate:
        return dict(gate, gameId=gid)
    h, err = _store(store, scope)
    if err:
        return err
    try:
        before, err = _get(h, key)
        if err:
            return err
        if action == "set":
            r = T.rbx_call({"handle": h}, "SetAsync", [key, value], timeout=30, allow_ids=allow_ids)
            want = value
        elif action == "increment":
            r = T.rbx_call({"handle": h}, "IncrementAsync", [key, delta], timeout=30, allow_ids=allow_ids)
            old = before.get("value")
            want = (old if _num(old) else 0) + delta
        else:
            r = T.rbx_call({"handle": h}, "RemoveAsync", [key], timeout=30, allow_ids=allow_ids)
            want = None
        if not r.get("ok"):
            return _ds_error(r, action)
        after, err = _get(h, key)
        if err:
            return dict(err, ran=True, outcomeUnknown=True, error="written, then the read-back failed: %s" % err.get("error"))
        good = _loose_same(want, after.get("value"))
        out = {"ok": good, "verified": good, "store": store, "key": key, "action": action, "before": before.get("value"),
               "after": after.get("value"), "keyInfo": after.get("keyInfo"), "returned": _no_handles(r.get("value", r.get("values")))}
        if not good:
            out["error"] = "GetAsync reads %r after the %s, expected %r" % (after.get("value"), action, want)
        else:
            out["proved"] = ds_proved(action, before.get("value"), after.get("value"))
        return out
    finally:
        _release(h)


# ---- Open Cloud data stores (v2), outside Studio, with the person's own key (opencloud.py reads and redacts it)
def _cloud(method, path, body=None, query=None):
    from mif_roblox import opencloud as OC
    key, err = OC.read_key()
    if err:
        return None, None, err
    if query:
        path += "?" + urllib.parse.urlencode({k: v for k, v in query.items() if v is not None})
    data = json.dumps(body).encode("utf-8") if body is not None else None
    st, out, _w = OC._call(method, path, key, data, "application/json" if data is not None else None)
    return st, OC.redact(out, key), None


EMPTY_PAGES = 10   # empty pages with a token followed by a key listing before it answers what it has


def _entry_path(universe_id, store, key=None, scope=None):
    q = urllib.parse.quote
    p = "/cloud/v2/universes/%d/data-stores/%s" % (universe_id, q(store, safe=""))
    if scope:
        p += "/scopes/%s" % q(scope, safe="")
    p += "/entries"
    if key is not None:
        p += "/%s" % q(key, safe="")
    return p


@T._reports
def rbx_datastore_cloud(action: str, universe_id: int, store: str = None, key: str = None, value: Any = None,
                        amount: int = None, scope: str = None, page_size: int = 50, page_token: str = None,
                        allow_live_write: bool = False, allow_ids: list = None) -> dict:
    "Roblox Open Cloud: the experience's data stores without Studio, with the person's own key (read from their key file, never shown). action list_stores, list_keys, get, revisions (reads, free), or set, increment, remove (writes: refused unless allow_live_write is true and the MifBridge panel is in full mode, decision 5, and universe_id is on the person's write-target list, roblox_write_targets.json, or named in allow_ids: the key's scopes usually cover every experience, refusedRule write-target). Endpoints are Roblox's v2 ones: /cloud/v2/universes/{universe}/data-stores[/{store}[/scopes/{scope}]/entries[/{key}]], :increment, :listRevisions. Read back: every write is followed by a get of the entry and compared; an increment reads the entry first (a missing key is 0) and must read back exactly that plus amount. It is not idempotent, so it is sent once (never retried after no answer or a 5xx); when the send got no answer the reply says landed yes, no or unknown (outcomeUnknown) from a read of the entry."
    if not isinstance(universe_id, int) or isinstance(universe_id, bool) or universe_id <= 0:
        return _no("universe_id is the experience's universe (game) id, a positive whole number")
    reads, writes = ("list_stores", "list_keys", "get", "revisions"), ("set", "increment", "remove")
    if action not in reads + writes:
        return _no("action is one of %s" % ", ".join(reads + writes))
    if action != "list_stores" and not (isinstance(store, str) and store):
        return _no("%s needs store" % action)
    if action in ("get", "revisions") + writes and not (isinstance(key, str) and key):
        return _no("%s needs key" % action)
    if not isinstance(page_size, int) or not 1 <= page_size <= 256:
        return _no("page_size is 1-256")
    if action in writes:
        gate = _live_gate(allow_live_write)
        if gate:
            return gate
        from mif_roblox import opencloud as OC
        gate = OC.write_target_check(universe_id, False, allow_ids)
        if gate:
            return gate
    page = {"maxPageSize": page_size, "pageToken": page_token}
    if action == "list_stores":
        st, out, err = _cloud("GET", "/cloud/v2/universes/%d/data-stores" % universe_id, query=page)
    elif action == "list_keys":
        st, out, err = _cloud("GET", _entry_path(universe_id, store, scope=scope), query=page)
        # Live 2026-11-21 (two stores): the first page answered no entries and only nextPageToken "8#"; the keys came on
        # the next page. An empty page with a token is followed (up to EMPTY_PAGES), never answered as "no keys".
        followed = 0
        while (not err and st == 200 and isinstance(out, dict) and not out.get("dataStoreEntries")
               and out.get("nextPageToken") and followed < EMPTY_PAGES):
            followed += 1
            st, out, err = _cloud("GET", _entry_path(universe_id, store, scope=scope),
                                  query=dict(page, pageToken=out["nextPageToken"]))
        if followed and isinstance(out, dict):
            out = dict(out, emptyPagesFollowed=followed)
    elif action == "get":
        st, out, err = _cloud("GET", _entry_path(universe_id, store, key, scope))
    elif action == "revisions":
        st, out, err = _cloud("GET", _entry_path(universe_id, store, key, scope) + ":listRevisions", query=page)
    elif action == "set":
        st, out, err = _cloud("PATCH", _entry_path(universe_id, store, key, scope), {"value": value},
                              query={"allowMissing": "true"})
    elif action == "increment":
        if not isinstance(amount, int) or isinstance(amount, bool):
            return _no("increment needs amount, a whole number")
        # The value before, so the read-back can prove the entry moved by exactly amount (a doubled increment, or one
        # that did nothing, still reads as a number). A key that does not exist yet (404) starts at 0.
        bst, bgot, berr = _cloud("GET", _entry_path(universe_id, store, key, scope))
        if berr:
            return berr
        if bst == 404:
            inc_before = 0
        elif bst == 200 and _num((bgot or {}).get("value") if isinstance(bgot, dict) else None):
            inc_before = bgot["value"]
        else:
            return _no("the entry's value before the increment could not be read as a number (%s: %s), so the increment "
                       "could not be proven; nothing was sent" % (bst, json.dumps(bgot)[:200]), readBefore={
                           "status": bst, "entry": bgot})
        st, out, err = _cloud("POST", _entry_path(universe_id, store, key, scope) + ":increment", {"amount": amount})
    else:
        st, out, err = _cloud("DELETE", _entry_path(universe_id, store, key, scope))
    if err:
        return err
    if action == "increment":
        from mif_roblox import opencloud as OC
        if OC.outcome_unknown(st):
            gst, got, _e = _cloud("GET", _entry_path(universe_id, store, key, scope))
            now = (got or {}).get("value") if gst == 200 and isinstance(got, dict) else (0 if gst == 404 else None)
            want = inc_before + amount
            landed = ("unknown" if not _num(now) else "yes" if _loose_same(want, now)
                      else "no" if _loose_same(inc_before, now) else "unknown")
            return {"ok": False, "ran": None, "outcomeUnknown": landed == "unknown", "landed": landed, "status": st,
                    "action": action, "before": inc_before, "after": now, "expected": want,
                    "readBack": {"status": gst, "entry": got},
                    "error": "the increment answered %s (no answer, or a server error) and was sent once, not again; the "
                             "entry reads %r against %r before (%r expected after one increment): %s" % (
                                 st or "nothing", now, inc_before, want,
                                 {"yes": "it landed once", "no": "it did not land",
                                  "unknown": "read the entry before sending it again"}[landed])}
    res = {"ok": 200 <= st < 300, "status": st, "action": action, "answer": out}
    if not res["ok"]:
        res["error"] = "Open Cloud answered %s: %s" % (st, json.dumps(out)[:400])
        if st in (401, 403):
            # Measured 2026-11-20: a key without data-store permissions answers 403 "Scope not authorized."; say what
            # the key holds so the person knows to add the experience's Data Stores permissions to it.
            from mif_roblox import opencloud as OC
            res.update(refusedRule="key-scope", keyScopes=OC.scope_summary(),
                       steps=["Creator Hub > Open Cloud > API Keys > this key > Access Permissions: add Data Stores for "
                              "this experience with the operations this action needs, then Save."])
        return res
    if action in reads:
        return res
    gst, got, _e = _cloud("GET", _entry_path(universe_id, store, key, scope))
    res["readBack"] = {"status": gst, "entry": got}
    if action == "remove":
        good = gst == 404
    elif action == "set":
        good = gst == 200 and _loose_same(value, (got or {}).get("value"))
    else:
        # after == before + amount, from the read-back and, when the increment's answer carries the new value, from it too
        expect = inc_before + amount
        answered = out.get("value") if isinstance(out, dict) else None
        back = got.get("value") if isinstance(got, dict) else None
        res.update(before=inc_before, expected=expect, after=back)
        good = (gst == 200 and _num(back) and _loose_same(expect, back)
                and (answered is None or (_num(answered) and _loose_same(expect, answered))))
    res.update(ok=good, verified=good)
    if not good:
        res["error"] = "the entry read back after the %s does not hold what was written (%s)" % (action, gst)
        if action == "increment":
            res["error"] += ": before %s + amount %s should read %s; the read-back holds %s%s" % (
                inc_before, amount, inc_before + amount, back,
                "" if answered is None else ", the increment answered %s" % (answered,))
    return res


@T._reports
def rbx_game_setup(teams: list = None, spawns: list = None, starter_player: dict = None, text_chat: dict = None,
                   tools: list = None) -> dict:
    "Roblox Studio: an experience's starting setup in one undo step - teams [{name, TeamColor, AutoAssignable}] under the Teams service; spawns [{name, parent?, team?, Neutral?, Position?, Size?, Duration?...}] as SpawnLocations (a spawn's team names one of the teams and takes its TeamColor); starter_player StarterPlayer properties (CharacterWalkSpeed, CameraMaxZoomDistance...); text_chat TextChatService properties; tools [{name, parent?, RequiresHandle?, CanBeDropped?, ToolTip?...}] as Tools (parent default StarterPack). Every property is checked against the catalog first (TextChatService.ChatVersion's write is RobloxScriptSecurity, so it is refused). Read back: each property of each object."
    ops, colors = [], {}
    for i, t in enumerate(teams or []):
        if not isinstance(t, dict) or not isinstance(t.get("name"), str):
            return _no("team %d is {name, TeamColor, AutoAssignable?}" % i)
        if not _exists("Teams"):
            return _no("this place has no Teams service; the person adds it (Model tab > Service > Teams), or plugin "
                       "support for services not yet created is needed (requested)", refusedRule="not-found")
        p = {k: v for k, v in t.items() if k != "name"}
        colors[t["name"]] = p.get("TeamColor")
        ops.append({"tool": "rbx_create", "args": {"class_name": "Team", "parent": "Teams", "name": t["name"], "props": p or None}})
    for i, s in enumerate(spawns or []):
        if not isinstance(s, dict) or not isinstance(s.get("name"), str):
            return _no("spawn %d is {name, parent?, team?, ...SpawnLocation properties}" % i)
        p = {k: v for k, v in s.items() if k not in ("name", "parent", "team")}
        if s.get("team") is not None:
            if s["team"] not in colors or colors[s["team"]] is None:
                return _no("spawn %d: team %r is not one of this call's teams with a TeamColor" % (i, s["team"]))
            p.update(TeamColor=colors[s["team"]], Neutral=False)
        ops.append({"tool": "rbx_create", "args": {"class_name": "SpawnLocation", "parent": s.get("parent", SCRATCH),
                                                   "name": s["name"], "props": p or None}})
    for i, t in enumerate(tools or []):
        if not isinstance(t, dict) or not isinstance(t.get("name"), str):
            return _no("tool %d is {name, parent?, ...Tool properties}" % i)
        p = {k: v for k, v in t.items() if k not in ("name", "parent")}
        ops.append({"tool": "rbx_create", "args": {"class_name": "Tool", "parent": t.get("parent", "StarterPack"),
                                                   "name": t["name"], "props": p or None}})
    if starter_player:
        ops.append({"tool": "rbx_set_props", "args": {"target": "StarterPlayer", "props": starter_player}})
    if text_chat:
        ops.append({"tool": "rbx_set_props", "args": {"target": "TextChatService", "props": text_chat}})
    if not ops:
        return _no("give teams, spawns, starter_player, text_chat and/or tools")
    for op in ops:
        if op["tool"] == "rbx_create" and not op["args"]["props"]:
            op["args"].pop("props")
    built = _build(ops, "game setup")
    if not built.get("ok"):
        return built
    back = {}
    for op in ops:
        a = op["args"]
        path = a.get("target") or _path(_child(a["parent"], a["name"]))
        names = list(a.get("props") or {})
        if names:
            vals, err = _props(path, names)
            back[path] = vals if not err else {"error": err.get("error")}
        else:
            back[path] = {"exists": _exists(path)}
    return {"ok": True, "verified": True, "objects": sorted(back), "readBack": back, "undoSteps": built.get("undoSteps")}


INFO_TYPES = ("Asset", "Product", "GamePass", "Subscription", "Bundle")


def rbx_product_info(asset_id: int = None, info_type: str = "Asset", items: list = None, roots: list = None,
                     file: str = None, folder: str = None) -> dict:
    "Roblox Studio: a Marketplace record (MarketplaceService:GetProductInfoAsync) for an asset, developer product, game pass or bundle id: the whole record, and its Name, Creator and price picked out. items checks a game's config instead: a list of {key, id, kind: GamePass or Product, price} (price optional), each id read as the kind named and answered with whether it exists as that kind, is for sale and has the price given (the list price, UserBasePriceInRobux: PriceInRobux is after the reading account's own discounts); when the record is missing as that kind, the other kind is asked so the answer can say a pass id was filed as a product (or the reverse). A null id is reported unset, never skipped, and fails the check like any mismatch. TYPED PRICES: items with roots (paths in the connected Studio, 'game' for the whole place), or with file (a saved .rbxlx or .rbxl) or folder (.lua/.luau files) - the two that need no Studio and read no Marketplace record - also searches the Text of every TextLabel and TextButton and every script string literal there for each item's list price (the price given, and with roots the list price read) typed beside a Robux mark (R$, Robux, the Robux glyph) or the item's name: a typed price is the list price for every player, while the price a player pays is set per region, and Roblox's price management will not finish its opt-in while it detects one. Each finding under typedPrices has its path, line, text, the price, the items it can be, what it stood beside (by: robux-mark, item-name, or item-name-in-path for a label named for the item) and the fix (read the price at run time); a price read from GetProductInfoAsync is not one. A root, file or folder with no UI text and no scripts fails rather than answering clean, and cannotSee says what a static search misses. Read only; nothing is created, bought or changed."
    if roots is not None or file is not None or folder is not None:
        if items is None:
            return _no("roots, file and folder search for the prices of items: give items too")
        return _typed_prices(items, roots, file, folder)
    if items is not None:
        return _product_items(items)
    if not isinstance(asset_id, int) or isinstance(asset_id, bool) or asset_id <= 0:
        return _no("asset_id is a positive whole number")
    if info_type not in INFO_TYPES:
        return _no("info_type is one of %s" % ", ".join(INFO_TYPES))
    r = T.rbx_call("MarketplaceService", "GetProductInfoAsync", [asset_id, info_type], timeout=60)
    if not r.get("ok"):
        return r
    rec = r.get("value") if isinstance(r.get("value"), dict) else {}
    creator = rec.get("Creator") if isinstance(rec.get("Creator"), dict) else {}
    return {"ok": bool(rec), "assetId": asset_id, "infoType": info_type, "name": rec.get("Name"),
            "creator": {k: creator.get(k) for k in ("Name", "Id", "CreatorType", "CreatorTargetId") if k in creator},
            "priceInRobux": rec.get("PriceInRobux"), "isForSale": rec.get("IsForSale"), "record": rec,
            "error": None if rec else "GetProductInfoAsync returned no record"}


PRODUCT_KINDS = {"GamePass": "GamePass", "Product": "Product", "pass": "GamePass", "product": "Product",
                 "gamepass": "GamePass", "developer product": "Product"}


def _product_record(pid, kind):
    r = T.rbx_call("MarketplaceService", "GetProductInfoAsync", [pid, kind], timeout=60)
    rec = r.get("value") if r.get("ok") and isinstance(r.get("value"), dict) else None
    return rec or None, (None if rec else (r.get("error") or "no record"))


def _items_refusal(items):
    if not isinstance(items, list) or not 1 <= len(items) <= 100:
        return _no("items is a list of 1-100 {key, id, kind, price}")
    for i, it in enumerate(items):
        if not isinstance(it, dict) or set(it) - {"key", "id", "kind", "price"} or not isinstance(it.get("key"), str):
            return _no("items[%d] is {key: its config name, id, kind: GamePass or Product, price (optional)}" % i)
        if it.get("kind") not in PRODUCT_KINDS:
            return _no("items[%d].kind is GamePass or Product" % i)
        pid = it.get("id")
        if pid is not None and (not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0):
            return _no("items[%d].id is a positive whole number, or null while it is unset" % i)
        pr = it.get("price")
        if pr is not None and (not isinstance(pr, int) or isinstance(pr, bool) or pr < 0):
            return _no("items[%d].price is whole Robux" % i)
    return None


def _product_items(items):
    bad = _items_refusal(items)
    if bad:
        return bad
    rows, problems = [], []
    for it in items:
        kind = PRODUCT_KINDS[it["kind"]]
        row = {"key": it["key"], "id": it.get("id"), "kind": kind, "price": it.get("price")}
        if it.get("id") is None:
            row.update(status="unset", ok=False)
            problems.append("%s is unset" % it["key"])
            rows.append(row)
            continue
        rec, err = _product_record(it["id"], kind)
        if rec is None:
            other = "Product" if kind == "GamePass" else "GamePass"
            rec2, _e = _product_record(it["id"], other)
            if rec2 is not None:
                row.update(status="wrong-kind", ok=False, foundAs=other, name=rec2.get("Name"))
                problems.append("%s: %s is a %s, not a %s" % (it["key"], it["id"], other, kind))
            else:
                row.update(status="missing", ok=False, error=err)
                problems.append("%s: %s has no %s record (%s)" % (it["key"], it["id"], kind, err))
            rows.append(row)
            continue
        # PriceInRobux is the price after the reading account's own discounts (a Roblox Plus subscription took 10% off,
        # the internal notes); the price a creator sets is UserBasePriceInRobux.
        base = rec.get("UserBasePriceInRobux")
        got_price, for_sale = (base if isinstance(base, int) and base > 0 else rec.get("PriceInRobux")), rec.get("IsForSale")
        row.update(name=rec.get("Name"), listPrice=got_price, priceInRobux=rec.get("PriceInRobux"), isForSale=for_sale,
                   productType=rec.get("ProductType"), record=rec)
        bad = []
        if for_sale is not True:
            bad.append("not for sale")
        if it.get("price") is not None and got_price != it["price"]:
            bad.append("price %s, config says %s" % (got_price, it["price"]))
        row.update(status="ok" if not bad else "mismatch", ok=not bad, problems=bad)
        if bad:
            problems.append("%s: %s" % (it["key"], ", ".join(bad)))
        rows.append(row)
    out = {"ok": not problems, "items": rows, "checked": len(rows),
           "unset": [r["key"] for r in rows if r["status"] == "unset"]}
    if problems:
        out["error"] = "; ".join(problems)
    return out


MAX_PRICE_TEXTS = 2000   # UI texts read from a Studio in one call (one property read each); more is truncated


def _typed_prices(items, roots, file, folder):
    """rbx_product_info's search for typed prices: save_audit.typed_prices over a saved place, a folder, or the scripts
    and UI text under roots in the connected Studio (where the items' config check runs too)."""
    from mif_roblox import save_audit as SA
    bad = _items_refusal(items)
    if bad:
        return bad
    if sum(x is not None for x in (roots, file, folder)) != 1:
        return _no("give one of roots, file or folder")
    want = [{"key": it["key"], "prices": [it["price"]] if it.get("price") is not None else [], "names": [it["key"]]}
            for it in items]
    if roots is None:
        if not any(w["prices"] for w in want):
            return _no("a file or folder is searched for each item's price as given, with no Studio to read it from: "
                       "give the items their price")
        if file is not None:
            if not isinstance(file, str) or not os.path.isfile(file):
                return _no("file is an existing place file (.rbxlx or .rbxl): %r" % (file,))
            ext = os.path.splitext(file)[1].lower()
            if ext not in (".rbxlx", ".rbxl"):
                return _no("file is a .rbxlx or .rbxl place (got %s)" % (ext or "no extension"))
            try:
                sources, origin = SA.sources_from_file(file, text_classes=SA.TEXT_CLASSES)
            except Exception as exc:  # noqa: BLE001 - an unreadable place is named with its reason
                return _no("%s could not be read as a place: %s: %s" % (os.path.basename(file), type(exc).__name__, exc),
                           ran=True)
            scan = SA.typed_prices(want, sources, origin)
            scan["file"] = file
        else:
            if not isinstance(folder, str) or not os.path.isdir(folder):
                return _no("folder is an existing folder of .lua/.luau files: %r" % (folder,))
            scan = SA.typed_prices(want, SA.sources_from_folder(folder), "folder")
            scan["folder"] = folder
        out = {"ok": scan["ok"], "ran": True, "checked": 0, "typedPrices": scan,
               "marketplace": "not read: a file or folder is searched with no Studio, for each item's price as given"}
        if not scan["ok"]:
            out["error"] = scan.get("error")
        return out
    if not isinstance(roots, list) or not 1 <= len(roots) <= 20:
        return _no("roots is a list of 1-20 paths in the place ('game' for all of it)")
    parts_of = []
    for i, root in enumerate(roots):
        parts = [] if root == "game" else V.split_path(root) if isinstance(root, (str, list)) else None
        if parts is None:
            return _no("roots[%d] is a path in the place: %r" % (i, root))
        parts_of.append(parts)
    res = _product_items(items)
    if "items" not in res:
        return res
    for w, row in zip(want, res["items"]):
        lp = row.get("listPrice")
        if isinstance(lp, int) and not isinstance(lp, bool) and lp > 0:
            w["prices"].append(lp)
        if isinstance(row.get("name"), str):
            w["names"].append(row["name"])
    sources, truncated, empty, texts = [], False, [], 0
    for parts in parts_of:
        name = V.join_path(parts) or "game"
        r = T._rbx("script_scan", root=parts)
        if not r.get("ok"):
            return r
        got = [x for x in r.get("sources") or [] if isinstance(x, dict) and isinstance(x.get("source"), str)]
        truncated = truncated or bool(r.get("truncated"))
        for cls in SA.TEXT_CLASSES:
            page = 0
            while True:
                q = T.rbx_query(root=parts or "game", class_name=cls, page=page, page_size=500)
                if not q.get("ok"):
                    return q
                for m in q.get("matches") or []:
                    if texts >= MAX_PRICE_TEXTS:
                        truncated = True
                        break
                    vals, err = _props(m["handle"] if m.get("ambiguous") and m.get("handle") else m["path"], ["Text"])
                    if err:
                        return dict(err, ok=False, error="the Text of %s could not be read: %s" % (m.get("path"), err.get("error")))
                    texts += 1
                    if isinstance(vals.get("Text"), str):
                        got.append({"path": m["path"], "className": m.get("className") or cls, "text": vals["Text"]})
                if not q.get("more") or texts >= MAX_PRICE_TEXTS:
                    break
                page += 1
        if not got:
            empty.append(name)
        sources += got
    scan = SA.typed_prices(want, sources, "studio", truncated=truncated)
    scan["roots"] = [V.join_path(p) or "game" for p in parts_of]
    if empty:
        scan.update(ok=False, emptyRoots=empty)
        scan["error"] = "no UI text and no scripts under %s: nothing was checked there, which is not clean%s" % (
            ", ".join(empty), ("; " + scan["error"]) if sources and scan.get("error") else "")
    out = dict(res, ok=bool(res.get("ok")) and scan["ok"], typedPrices=scan)
    errors = [e for e in (res.get("error"), scan.get("error")) if e]
    if errors:
        out["error"] = "; ".join(errors)
    return out


# ================================================================ batch 14: editor, packages, collaboration (5.13, 5.15, 5.16)
# Studio's preferences live on settings().Studio, not under game. The plugin reaches them when a path starts with
# "settings()" (plugin request filed by lane 5); until then a read answers the plugin's not-found.
STUDIO_SETTINGS = "settings().Studio"


def _preference_refusal(names):
    return _no("%s would change the person's Studio, not the place. Decision 7 (which preferences the bridge may set) is "
               "open, so every preference is read only" % ", ".join(sorted(names)), refusedRule="studio-preference")


def rbx_studio_settings(names: list = None, set: dict = None) -> dict:  # noqa: A002 - the verb a caller writes
    "Roblox Studio: Studio's own preferences (settings().Studio) - every property a plugin can read, or names; the theme by name and Studio:GetAvailableThemes. Read only: set is refused studio-preference, since these change the person's Studio and which ones the bridge may set is not decided."
    if set:
        return _preference_refusal(set)
    cat, err = T._catalog()
    if err:
        return err
    readable = sorted(n for n, (_o, r) in cat.members_of("Studio").items()
                      if r[0] == "P" and cat.read_refusal("Studio", n, kind="P") is None and r[1] not in ("QFont", "QDir")
                      and "Deprecated" not in r[4])
    want = names or readable
    unknown = [n for n in want if n not in readable]
    if unknown:
        return _no("not readable Studio properties: %s" % unknown, readable=len(readable))
    values, errors = {}, {}
    for i in range(0, len(want), 40):
        chunk = [n for n in want[i:i + 40] if n != "Theme"]
        if not chunk:
            continue
        r = T._rbx("get_props", target=V.split_path(STUDIO_SETTINGS), props=chunk)
        if not r.get("ok"):
            return dict(r, ok=False, note="the plugin reaches settings().Studio only with the settings() path root (requested)")
        for k, v in T._obj(r.get("values")).items():
            values[k] = V.decode(v)
    if "Theme" in want:
        th = T._rbx("get_props", target=V.split_path(STUDIO_SETTINGS), props=["Theme"])
        tv = V.decode(T._obj(th.get("values")).get("Theme"))
        h = _handle(tv)
        if h:
            nm, _e = _props({"handle": h}, ["Name"])
            _release(h)
            values["Theme"] = (nm or {}).get("Name")
        else:
            values["Theme"] = tv
    themes = T._rbx("call", target=V.split_path(STUDIO_SETTINGS), method="GetAvailableThemes", args=[], argc=0,
                    readOnly=True, yields=False, callTimeout=10)
    return {"ok": True, "values": values, "count": len(values), "errors": errors or None,
            "availableThemes": (V.decode((themes.get("returns") or [None])[0]) if themes.get("ok")
                                else {"error": themes.get("error")}),
            "writable": False, "decision": "7 open: read only"}


DRAGGER = ("LinearSnapEnabled", "LinearSnapIncrement", "AngleSnapEnabled", "AngleSnapIncrement")
# Decision 7 (Mifsopo, 2026-11-20: yes): the bridge may set the snapping and must put the person's values back. The values
# it replaced are kept on disk the first time it changes them, so restore works after a crash or another session.
# DraggerService's snapping outlives the place and the Studio (RB1400), and the plugin refuses a DraggerService write
# outside full mode (studio-preference), the scratch place included.
DRAGGER_ORIGINAL = "dragger_original.json"


def _dragger_file():
    from mif_roblox import daemon as D
    return os.path.join(D.home(), DRAGGER_ORIGINAL)


def _dragger_read():
    dv, err = _props("DraggerService", list(DRAGGER))
    if err:
        return None, None, err
    sv, err = _props("StudioService", ["GridSize", "RotateIncrement"])
    if err:
        return None, None, err
    return dv, sv, None


def _dragger_check(values):
    if not isinstance(values, dict) or not values:
        return "set is a dict of %s" % ", ".join(DRAGGER)
    bad = [k for k in values if k not in DRAGGER]
    if bad:
        return "not a snapping setting: %s (these are %s)" % (bad, ", ".join(DRAGGER))
    for k, v in values.items():
        if k.endswith("Enabled"):
            if not isinstance(v, bool):
                return "%s is true or false" % k
        elif not _num(v) or v <= 0 or (k == "AngleSnapIncrement" and v > 360):
            return "%s is a number above 0%s" % (k, " and at most 360" if k == "AngleSnapIncrement" else "")
    return None


def _dragger_write(values, dv_before):
    """(answer, error). An error carries setRan: False only when the set itself was refused before it changed anything
    (the one case where the person's saved values may be forgotten); True when the set ran and the read-back failed;
    None when the set answered outcomeUnknown (it may have landed: nothing more is sent)."""
    r = T.rbx_set_props("DraggerService", dict(values))
    if r.get("outcomeUnknown"):
        return None, dict(r, ok=False, setRan=None, step="set",
                          error="the snap set answered outcomeUnknown (%s): it may have changed the snapping, so nothing "
                                "more was sent; read it with rbx_dragger_settings() before anything else" % r.get("error"))
    if not r.get("ok") and (r.get("refusedRule") or r.get("ran") is False):
        return None, dict(r, setRan=False)
    dv, sv, err = _dragger_read()
    if err:
        return None, dict(err, ok=False, setRan=True, step="read-back", setProps=r,
                          error="the snap set ran but reading it back failed: %s" % err.get("error"))
    wrong = {k: {"sent": v, "readBack": dv.get(k)} for k, v in values.items()
             if not (dv.get(k) == v or (_num(v) and _num(dv.get(k)) and abs(dv[k] - v) < 1e-4))}
    return {"dragger": dv, "studio": sv, "before": dv_before, "mismatch": wrong or None, "setProps": r}, None


def rbx_dragger_settings(set: dict = None, restore: bool = False) -> dict:  # noqa: A002
    "Roblox Studio: the editor's snapping - DraggerService LinearSnapEnabled, LinearSnapIncrement, AngleSnapEnabled, AngleSnapIncrement, with StudioService GridSize and RotateIncrement (read only mirrors). set changes them and reads them back; the person's values from before the bridge's first change are kept on disk, and restore: true puts exactly those back and forgets them. These are the person's Studio preferences, not the place (they outlive the place and Studio), so set and restore run only in full mode."
    if not isinstance(restore, bool):
        return _no("restore is true or false")
    if set and restore:
        return _no("set or restore, not both")
    path = _dragger_file()
    if not set and not restore:
        dv, sv, err = _dragger_read()
        if err:
            return err
        saved = None
        if os.path.isfile(path):
            try:
                saved = json.load(open(path, encoding="utf-8")).get("values")
            except (OSError, ValueError):
                saved = {"error": "unreadable: %s" % path}
        return {"ok": True, "dragger": dv, "studio": sv, "writable": "full mode", "pendingRestore": saved}
    if restore:
        try:
            rec = json.load(open(path, encoding="utf-8"))
        except FileNotFoundError:
            return _no("nothing to restore: the bridge has not changed the snapping since the last restore", file=path)
        except (OSError, ValueError) as exc:
            return _no("the saved snapping at %s is unreadable: %s" % (path, exc))
        values = rec.get("values") if isinstance(rec, dict) else None
        msg = _dragger_check(values)
        if msg:
            return _no("the saved snapping at %s is not usable: %s" % (path, msg))
        dv0, _sv, err = _dragger_read()
        if err:
            return err
        out, err = _dragger_write(values, dv0)
        if err:
            return err
        ok = not out["mismatch"]
        if ok:
            os.remove(path)
        return dict(out, ok=ok, verified=ok, restored=values, savedAt=rec.get("savedAt"),
                    error=None if ok else "Studio did not hold every restored value; the saved values stay in %s" % path)
    msg = _dragger_check(set)
    if msg:
        return _no(msg)
    dv0, _sv, err = _dragger_read()
    if err:
        return err
    if os.path.isfile(path):
        kept = "kept from an earlier change"
    else:
        kept = "saved now"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        from mif_roblox import durable
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"values": dv0, "savedAt": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, f)
            durable.sync(f)
        os.replace(tmp, path)
    out, err = _dragger_write(set, dv0)
    if err:
        if kept == "saved now" and err.get("setRan") is False:
            os.remove(path)  # the set was refused: nothing was changed, so there is nothing to restore
        elif err.get("setRan") is not False:
            # the set ran (or may have): the person's values stay on disk for restore
            err = dict(err, original=kept, restoreWith="rbx_dragger_settings(restore=true)",
                       error="%s; the person's snapping from before is kept in %s for restore" % (err.get("error"), path))
        return err
    ok = not out["mismatch"]
    return dict(out, ok=ok, verified=ok, original=kept, restoreWith="rbx_dragger_settings(restore=true)",
                error=None if ok else "Studio did not hold every value sent")


PACKAGE_STEPS = {
    "make": "Right-click the model in the Explorer > Convert to Package.",
    "update": "Right-click the package > Publish Changes to Package (or Package > Update All).",
    "revert": "Right-click the package > Revert to Package / Discard Local Changes.",
}


def rbx_packages_read(root: Any = "game", contents: bool = False, action: str = None) -> dict:
    "Roblox Studio: every package in the place - each PackageLink under root with its package (the link's parent), PackageLink.PackageId and VersionNumber; with contents, the asset ids inside each package (AssetService:GetAssetIdsForPackageAsync). action make, update or revert is refused needs-person with the menu that does it: PackageService is RobloxScriptSecurity. Read only."
    if action is not None:
        if action not in PACKAGE_STEPS:
            return _no("action is make, update or revert")
        return _no("packages are made, published and reverted by the person in Studio (PackageService is "
                   "RobloxScriptSecurity)", refusedRule="needs-person", steps=[PACKAGE_STEPS[action]])
    q = T.rbx_query(root=root, class_name="PackageLink", page_size=500)
    if not q.get("ok"):
        return q
    out = []
    for m in q.get("matches") or []:
        vals, err = _props(m["path"], ["PackageId", "VersionNumber"])
        row = {"link": m["path"], "package": _path(V.split_path(m["path"])[:-1]), **(vals or {"error": (err or {}).get("error")})}
        if contents and vals:
            num = re.search(r"(\d+)", str(vals.get("PackageId") or ""))
            if num:
                ids = T.rbx_call("AssetService", "GetAssetIdsForPackageAsync", [int(num.group(1))], timeout=60)
                row["assetIds"] = ids.get("value") if ids.get("ok") else {"error": ids.get("error")}
        out.append(row)
    return {"ok": True, "root": _path(root) if root not in ("game", None) else "game", "packages": out, "count": len(out),
            "total": q.get("total"), "note": None if out else "the query ran over the root and found no PackageLink"}


def rbx_file_sync_status(file_path: str = None, limit: int = 200) -> dict:
    "Roblox Studio: which instances sync to files on disk - InstanceFileSyncService:GetAllInstances with InstanceFileSyncService:GetStatus for each, LiveSyncService.HasSyncedInstances and LiveSyncService:GetSyncState per instance; with file_path, the instance synced to that file (GetSyncedInstance). Read only."
    out = {"ok": True}
    if file_path is not None:
        r = T.rbx_call("InstanceFileSyncService", "GetSyncedInstance", [file_path])
        if not r.get("ok"):
            return r
        out["instance"] = r.get("value")
        if not r.get("value"):
            out.update(ok=False, error="no instance is synced to %s" % file_path)
            return out
    allr = T.rbx_call("InstanceFileSyncService", "GetAllInstances")
    if not allr.get("ok"):
        return allr
    insts = [p for p in (allr.get("value") or []) if isinstance(p, str)][:limit]
    rows = []
    for p in insts:
        st = T.rbx_call("InstanceFileSyncService", "GetStatus", [p])
        ls = T.rbx_call("LiveSyncService", "GetSyncState", [p])
        rows.append({"instance": p, "status": st.get("value"), "liveSync": ls.get("values", ls.get("value"))})
    has, herr = _props("LiveSyncService", ["HasSyncedInstances"])
    out.update(instances=rows, count=len(rows), hasSyncedInstances=(has or {}).get("HasSyncedInstances"))
    if herr:
        # On 0.748 LiveSyncService is not in a fresh place until something creates it, and the plugin finds services
        # with FindService.
        out["liveSyncService"] = {"error": herr.get("error"), "refusedRule": herr.get("refusedRule")}
    return out


def rbx_collab_guard(target: Any) -> dict:
    "Roblox Studio: would a script write to target be refused, and why - its open editor tab's ScriptDocument:IsReadOnly and GetReadOnlyReasons (a collaborator's lock, a package, a running test), and whether it sits inside a package (a PackageLink on an ancestor). rbx_script_write already refuses a read-only tab with Studio's reasons, and a script inside a package whether its tab is open or not; this answers before trying. Team Create drafts (DraftsService) are RobloxScriptSecurity and not read. Read only."
    cls, err = _class_of(target)
    if err:
        return err
    cat, err = T._catalog()
    if err:
        return err
    if not cat.is_a(cls, "LuaSourceContainer"):
        return _no("target is a script, got %s" % cls, refusedRule="type")
    reasons = []
    doc = T.rbx_script_documents(target)
    open_tab = bool(doc.get("ok"))
    if doc.get("ok") and doc.get("readOnly"):
        reasons += ["editor: %s" % r for r in (doc.get("readOnlyReasons") or ["read only"])]
    # The same check rbx_script_write and rbx_batch refuse a write by (tools._package_of): every page of PackageLinks.
    root, err = T._package_of(V.split_path(target))
    if err:
        return err
    package = _path(root) if root else None
    if package:
        reasons.append("package: inside %s; edits belong to the package and are published by the person" % package)
    return {"ok": True, "target": _path(target), "writable": not reasons, "reasons": reasons, "openTab": open_tab,
            "package": package, "refusedRule": "read-only-document" if reasons else None,
            "teamCreate": "drafts are not readable by a plugin (DraftsService is RobloxScriptSecurity)"}


@T._reports
def rbx_place_publish(file: str = None, place_id: int = None, confirm: bool = False, preflight: bool = False,
                      selftest_flag: str = None, config_module: str = None, id_keys: list = None,
                      odds_products: list = None, accept: list = None, allow_ids: list = None,
                      dry_run: bool = False, test_switches: list = None) -> dict:
    "Roblox Open Cloud: publish a place file the person saved (.rbxl or .rbxlx) as a new version of place_id with their own key (dry_run true runs every gate below and answers what it would send, sending nothing) - Roblox's Place Publishing API, POST /universes/v1/{universeId}/places/{placeId}/versions?versionType=Published with the file as the body (scope universe-places:write), sent once and never retried. A place a Studio has open from Roblox is held by its edit session (PlaceOngoingTeamCreateSession), answered place-session-open with the step; closing that Studio commits its unsaved edits as a version of their own about a minute later (live 2026-11-21). A publish goes live to players, so confirm must be true and the MifBridge panel in full mode, and the place and its experience must be on the person's write-target list (roblox_write_targets.json) or named in allow_ids (refusedRule write-target); MifBridge's own scratch place is refused; a copy rbx_launch made of the person's place (places/<port>/copies) is published like their own file. Read back: the versionNumber answered, above the asset's revision before (ok; verified only when exactly one higher, revisionNote says when another save landed in between), and the asset's revision after; an upload that gets no answer (a dropped connection, a 5xx) is outcomeUnknown with revisionMoved - false when the revision is unchanged right after it, so nothing has landed so far and it is read again before a second send. PREFLIGHT, run alone with preflight true (read only, no confirm) and always before a publish: from the place open in the connected Studio and from Roblox, each check answered pass, fail, warn, unread or ask with what it read, never set - placeOpen (Studio holds the file being published), selfTestOff (selftest_flag: a ValueBase path, 'Path@Attribute', or 'Module:key' read through rbx_module_values, off), idsSet (each of id_keys in config_module set to an id, never nil; a key is a dotted path and a * segment takes every entry of a list there, so 'Store.Passes.*.Id' checks the id inside each entry of a store list, each named by its path; a * over an empty list is unset), noSecretsInScripts (every script - read from the saved file itself, whole, with no cut, and from Studio only when the file cannot be read - scanned for the person's own Open Cloud key, an x-api-key header, a JWT, a Roblox cookie or a long secret-shaped token, reported by script and line, never the text), secretsPresent (every HttpService:GetSecret with a literal name, by script and line, against the names of the experience's secrets listed with the person's key - Roblox's Secrets Store API, GET /cloud/v2/universes/{universeId}/secrets, scope universe.secret:read, names only and never a secret's content: a name the experience does not hold fails with its script and line, a key without the scope is unread with the step, a computed name is listed unresolved and warned, never guessed; a place that reads no secret lists nothing), saveLayer (rbx_save_audit's check: no stand-in or table-backed store where RunService:IsStudio() can be false, and no failed DataStore call that carries on in memory, by script and line), passGrant (rbx_save_audit's pass check: no game pass granted on PromptGamePassPurchaseFinished or restored from the save on join without a UserOwnsGamePassAsync read, by script and line), paidRandomItems (odds_products, the Robux items that change odds, or [] for none: the questionnaire answer, and a script must read PolicyService's ArePaidRandomItemsRestricted), and with place_id the experience's visibility (private first), vrOff, consoleReady (rbx_controller_audit when Console is on), the 16+ age gate and gamePageThumbnails (its own page's media, a public read); noBridgeObjects (with file, read from the saved file itself, else the open Studio: the place still holds Workspace.MifScratch, MifBridge's working folder, which must never ship: fail with the step to delete it, save and run the preflight again; ServerStorage's MifBridgePort and MifBridgeLaunched markers, inert there, are warned in bridgeMarkers); fileCurrent (with file, when the connected Studio holds it: every script in the saved file against the same script in the open place, line endings set aside; a script that differs or is in one only fails, because the checks above that read Studio read the open place while the file is what goes live - save first; Studio's scan stops at 4 MB of source a call, so it is paged by subtree, and only a part still not read whole (one script over 4 MB, two children of one name) is unread, named in cutAt; properties outside scripts are not compared); testSwitchesOff (with file: every attribute a test script in the saved file sets - a script whose path names a self-test or a Test - by <receiver>:SetAttribute with a literal name, plus test_switches the caller names as 'Path@Attribute' or an attribute name, read on the file's instances: one still holding a value other than false, 0, '' or nil fails, named with the script line that sets it, because a self-test run in the edit place leaves its switches there and a save keeps them; a receiver that is a local variable is matched by name on every instance; an instance whose attributes do not decode is unread); noStackedCopies (with file: top-level names under Workspace, ReplicatedStorage and ServerStorage held more than once in the saved file, by name and count - a server's start-up folders stack one copy per Run-mode run, since Stop keeps what a run made: a name a script in the place names as a string fails, naming that script; any other repeated name is warned); teamCreateOff is the person's check (not readable by a plugin). A publish whose preflight has a fail, unread or ask is refused refusedRule preflight, unless every such check is named in accept by the person. TIME: a publish, and preflight true, run a play test for consoleReady (when Console is on) after waiting their turn for the machine's play-test lock, so one can take minutes - about 6 on a game on 2026-11-26. Never put a short timeout around this call: a caller ended mid play test leaves the audit's probe LocalScript in StarterPlayerScripts (the next rbx_controller_audit removes one that is MifBridge's own); dry_run true starts no play test. With no Studio connected and a file given, the preflight reads the file alone (fromFile): the scripts, Workspace.MifScratch and a self-test flag that is a ValueBase path (its Value from the file, .rbxlx or binary .rbxl); a check only an open Studio answers - an attribute or module flag, idsSet, consoleReady - is unread and named in notFromFile, never passed; placeOpen and fileCurrent are not run. The publish itself still needs the panel in full mode, so a connected Studio."
    from mif_roblox import opencloud as OC
    from mif_roblox import launch as L
    from mif_roblox import domain_store as DS
    for name, v in (("id_keys", id_keys), ("odds_products", odds_products), ("accept", accept),
                    ("test_switches", test_switches)):
        if v is not None and not (isinstance(v, list) and all(isinstance(x, (str, int)) and not isinstance(x, bool)
                                                              for x in v)):
            return _no("%s is a list of names" % name)
    for name, v in (("selftest_flag", selftest_flag), ("config_module", config_module)):
        if v is not None and not (isinstance(v, str) and v):
            return _no("%s is a path" % name)
    if place_id is not None and (not isinstance(place_id, int) or isinstance(place_id, bool) or place_id <= 0):
        return _no("place_id is a positive whole number")
    pre_args = dict(file=file, place_id=place_id, selftest_flag=selftest_flag, config_module=config_module,
                    id_keys=id_keys, odds_products=odds_products, test_switches=test_switches)
    if preflight is True:
        return DS.preflight(**pre_args)
    if confirm is not True:
        return _no("publishing makes a new live version of the place: pass confirm true once the person agreed",
                   refusedRule="needs-confirm")
    if not isinstance(place_id, int) or isinstance(place_id, bool) or place_id <= 0:
        return _no("place_id is a positive whole number")
    if not isinstance(file, str) or not os.path.isfile(file) or os.path.splitext(file)[1].lower() not in (".rbxl", ".rbxlx"):
        return _no("file is an existing .rbxl or .rbxlx the person saved")
    try:
        # MifBridge's own scratch place holds nothing of the person's; a copy rbx_launch made of their place
        # (places/<port>/copies) holds their game and is what rbx_place_save saves into, so it is publishable (found
        # 2026-11-22: refusing it broke the no-click route at its last step). The preflight fails a place that still
        # carries Workspace.MifScratch (noBridgeObjects).
        if L.in_places_folder(file) and not L.is_copy_place(file):
            return _no("that is MifBridge's own scratch place: nothing of the person's is in it, so it is not published",
                       refusedRule="scratch-place")
    except Exception:  # noqa: BLE001 - an unreadable check is not a pass: the file is still the person's choice
        pass
    mode, _st = T._plugin_mode()
    if mode != "full":
        return _no("publishing writes to the person's account; the MifBridge panel must be in full mode (it is %s)" % mode,
                   refusedRule="full-mode", mode=mode)
    uid, uerr = DS.universe_of_place(place_id)
    gate = OC.write_target_check(uid, place_id, allow_ids)
    if gate:
        return dict(gate, universeError=uerr) if uerr else gate
    pre = DS.preflight(play=dry_run is not True, **pre_args)   # a dry run starts no play test: such checks read unknown
    blocking = sorted(set(pre.get("failed") or []) | set(pre.get("toAsk") or []))
    if not pre.get("ran") or [b for b in blocking if b not in (accept or [])]:
        return _no("the preflight did not pass: %s. Fix them, or the person names each in accept" % (
            ", ".join(b for b in blocking if b not in (accept or [])) or pre.get("error")), refusedRule="preflight",
            preflight=pre)
    key, err = OC.read_key()
    if err:
        return err
    if dry_run is True:
        # every gate above ran (confirm, the file, full mode, the write-target list, the preflight, the key): say what
        # would be sent and send nothing - no revision read, no POST
        return {"ok": True, "ran": False, "dryRun": True, "placeId": place_id, "universeId": uid, "file": file,
                "bytes": os.path.getsize(file), "wouldPost": "/universes/v1/%d/places/%d/versions?versionType=Published"
                % (uid, place_id), "preflight": pre,
                "note": "nothing was sent; the same call with dry_run false publishes a new live version"}
    try:
        st0, before, _w = OC._call("GET", "/assets/v1/assets/%d" % place_id, key)
        rev0 = (before or {}).get("revisionId") if st0 == 200 and isinstance(before, dict) else None
        with open(file, "rb") as f:
            data = f.read()
        ctype = "application/xml" if file.lower().endswith(".rbxlx") else "application/octet-stream"
        # Roblox's Place Publishing API (scope universe-places:write, Creator Hub "Place Publishing"): the file is the
        # body. Sent once (tries=1): a retried POST after a lost answer could publish twice. Live 2026-11-21: the
        # Assets API PATCH of the same place answered "DataLoss: Place Content Corrupted" for an .rbxlx Studio opens.
        path = "/universes/v1/%d/places/%d/versions?versionType=Published" % (uid, place_id)
        st, ans, _w = OC._call("POST", path, key, data, ctype, timeout=300, tries=1)
        ver = ans.get("versionNumber") if isinstance(ans, dict) else None
        st1, after, _w = OC._call("GET", "/assets/v1/assets/%d" % place_id, key)
        rev1 = (after or {}).get("revisionId") if st1 == 200 and isinstance(after, dict) else None
        out = {"placeId": place_id, "universeId": uid, "file": file, "bytes": len(data), "status": st,
               "versionNumber": ver, "revisionBefore": rev0, "revisionAfter": rev1}
        if OC.outcome_unknown(st):
            # a connection that drops mid-upload (found 2026-11-26: a 131 MB place, an SSL write error) is no answer: the
            # revisions read on either side say whether it landed, so the caller is told which, not left to compare them
            try:
                moved = None if rev0 is None or rev1 is None else int(rev1) > int(rev0)
            except (TypeError, ValueError):
                moved = None
            say = ("the place's revision is still %s right after it: nothing has landed so far - read it again (a large "
                   "place is processed after the upload) and publish again only if it is still %s" % (rev0, rev0)
                   if moved is False else
                   "the revision moved from %s to %s: a version landed (this publish, or another save in between) - "
                   "do not publish again until its content is checked" % (rev0, rev1) if moved else
                   "the revision could not be read on both sides (before %s, after %s): read it before publishing "
                   "again" % (rev0, rev1))
            return OC.redact(dict(out, ok=False, ran=None, outcomeUnknown=True, revisionMoved=moved, answer=ans,
                                  error="the publish answered %s (no answer, or a server error): %s" % (
                                      st or "nothing", say)), key)
        try:
            # ok: a version above the revision before. A Team Create save between the read and the POST makes it +2,
            # still a good publish (ok false there would invite a second one); verified needs exactly one higher.
            good = st == 200 and ver is not None and (rev0 is None or int(ver) > int(rev0))
            one = rev0 is not None and ver is not None and int(ver) == int(rev0) + 1
            seen = rev1 is not None and ver is not None and int(rev1) == int(ver)
        except (TypeError, ValueError):
            good, one, seen = False, False, False
        out.update(ok=good, ran=True, verified=good and seen and one)
        if good and rev0 is not None and not one:
            out["revisionNote"] = ("published as version %s, more than one above revision %s: another save (Team Create) "
                                   "landed between the read and the publish; do not publish again" % (ver, rev0))
        if not good:
            out.update(answer=ans, error="POST %s answered %s, version %s (revision before: %s): %s" % (
                path.split("?")[0], st, ver, rev0, json.dumps(ans)[:300]))
            if OC.session_held(ans):
                out.update(refusedRule="place-session-open", steps=[OC.SESSION_STEP])
            elif st in (401, 403):
                out.update(refusedRule="key-scope", steps=["Creator Hub > Open Cloud > API Keys > this key > Access "
                                                           "Permissions: add Place Publishing (universe-places:write) for "
                                                           "this experience, then Save."])
        return OC.redact(out, key)
    except Exception as exc:  # noqa: BLE001 - never raise, never let the key out
        return OC.redact({"ok": False, "ran": None, "outcomeUnknown": True, "error": "publish failed: %s" % OC.exc_text(exc, key)}, key)


def _file_state(path):
    """{bytes, mtime, sha256} of a file, or None when it cannot be read."""
    import hashlib
    try:
        with open(path, "rb") as f:
            data = f.read()
        return {"bytes": len(data), "mtime": os.path.getmtime(path), "sha256": hashlib.sha256(data).hexdigest()}
    except OSError:
        return None


def place_file_shape(path):
    """What a place file holds, read from disk: {format, services, externals, instances, marker} or {error}. services
    and externals are an .rbxlx's top-level Items and <External> tags (a hand-written file without them was refused by
    Roblox, RB1969); a binary .rbxl gives its instance count. marker is ServerStorage.MifBridgePort or None."""
    from mif_roblox import launch as L
    from mif_roblox import rbxl as RX
    try:
        with open(path, "rb") as f:
            blob = f.read()
    except OSError as exc:
        return {"error": "the file cannot be read: %s" % exc}
    try:
        if blob.startswith(RX.MAGIC):
            _ncls, ninst, _chunks, _t = RX.parse(blob)
            return {"format": "rbxl", "instances": ninst, "marker": RX.read_marker(blob)}
        text = blob.decode("utf-8")
        root = ET.fromstring(text)
        _ref, marker = L._port_marker(text)
        return {"format": "rbxlx", "services": len(root.findall("Item")), "externals": len(root.findall("External")),
                "marker": marker}
    except (ValueError, ET.ParseError, UnicodeDecodeError) as exc:
        return {"error": "not a readable place file: %s" % exc}


@T._reports
def rbx_place_save(confirm: bool = False, use_builtin_mcp: bool = False, wait_seconds: int = 90,
                   place_file: str = None) -> dict:
    "Roblox Studio: save the open place to its own local file (.rbxl or .rbxlx, the one this Studio opened with -localPlaceFile) with no dialog, through Roblox's built-in Studio MCP server's save_place (Ctrl+S; RB1422) - the only dialog-free local save on 0.748 (a plugin's ExportPlace opens a dialog, RB43). OPT-IN: use_builtin_mcp true, because MifBridge otherwise never calls that server; the person's 'Enable Studio as MCP server' switch (Studio: Assistant > Manage MCP Servers) must be on. Needs confirm true and full mode. The Studio is proved to be MifBridge's own before the save: of the Studios the server lists, exactly one may have this file name open, and a Folder MifBridge makes in its ServerStorage must show there through the server's inspect_instance and be gone again before save_place runs; two Studios with the same file name are refused ambiguous and neither is read. Refused for MifBridge's scratch place and for a place opened from Roblox. A copy rbx_launch made of a person's place (places/<port>/copies) is MifBridge's own file and saves with confirm alone; any other file is the person's own, opened with rbx_launch open_original true, and saves only when place_file names that exact path (refused path-not-named otherwise), so a save never lands on a file the caller did not mean. Answers the file before and after (bytes, mtime, sha256), its shape read back from disk (format, services, External tags, instances, ServerStorage.MifBridgePort - which must still match), the server's reply and the proxy PIDs it started and ended. Then rbx_place_publish(file, place_id, confirm) sends the saved file."
    from mif_roblox import builtin_save as BS
    from mif_roblox import domain_release as DR
    from mif_roblox import launch as L
    if not isinstance(confirm, bool) or not isinstance(use_builtin_mcp, bool):
        return _no("confirm and use_builtin_mcp are true or false")
    if place_file is not None and not isinstance(place_file, str):
        return _no("place_file is the full path of the place file to save over, or omitted")
    if not isinstance(wait_seconds, int) or isinstance(wait_seconds, bool) or not 10 <= wait_seconds <= 300:
        return _no("wait_seconds is 10-300 (the Studios join the built-in server within about 40 s, RB1422)")
    if use_builtin_mcp is not True:
        return _no("the only dialog-free local save on Studio 0.748 is Roblox's built-in Studio MCP server's save_place "
                   "(RB1422); MifBridge calls that server only when asked: pass use_builtin_mcp true",
                   refusedRule="opt-in", builtInMcp=DR.builtin_mcp())
    if confirm is not True:
        return _no("saving writes the open place over its local file: pass confirm true once the person agreed",
                   refusedRule="needs-confirm")
    mode, st = T._plugin_mode()
    if mode != "full":
        return _no("saving writes the person's place file; the MifBridge panel must be in full mode (it is %s)" % mode,
                   refusedRule="full-mode", mode=mode)
    pid = ((st or {}).get("plugin") or {}).get("studioPid")
    args = L.CMD_ARGS(pid) if pid else None
    if not args:
        return _no("the connected Studio's command line could not be read (pid %s), so its place file is not known" % pid)
    joined = " ".join(args)
    if "-task" in args and "EditPlace" in args:
        return _no("this Studio opened the place from Roblox (-task EditPlace): its save is a cloud version, not a file. "
                   "Open a local copy (rbx_launch place_file) to save a file", refusedRule="not-local-file",
                   studioPid=pid)
    place = L._place_arg(args)
    if not place or not os.path.isfile(place):
        return _no("the connected Studio has no local place file open (%s)" % (place or joined[:200]),
                   refusedRule="not-local-file", studioPid=pid)
    if L.is_scratch_place(place) and not L.is_copy_place(place):
        return _no("that is MifBridge's own scratch place: nothing of the person's is in it, and every lane's scratch "
                   "file has the same name", refusedRule="scratch-place", file=place)
    # A person's own file is written only when the caller names it (found 2026-11-22: a launch on the default port
    # opened the person's file itself, and a save from that Studio would have written over it).
    if not L.is_copy_place(place) and not (place_file and os.path.normcase(os.path.abspath(place_file))
                                           == os.path.normcase(os.path.abspath(place))):
        return _no("this Studio has the person's own file %s open, not a copy MifBridge made: saving writes over it, so "
                   "place_file must name that exact path%s" % (place, " (it names %s)" % place_file if place_file else ""),
                   refusedRule="path-not-named", file=place)
    enabled = DR._builtin_enabled()
    # A settings file with no mcp-server key at all is the switch a reboot dropped (RB2889). The plugin copy read
    # first can still say on, so the key is read separately and named as the cause of a refusal; it is not refused on
    # by itself, because the plugin copy has been right while the settings file was stale (RB1422).
    dropped = [w for w in DR.builtin_warnings() if w.get("code") == "mcp-server-key-missing"]
    if enabled is False:
        return _no("Roblox's built-in Studio MCP server is switched off on this machine" + (
                   " (Studio's AssistantSettings file holds no mcp-server key: a reboot dropped the switch, RB2889)"
                   if dropped else ""), refusedRule="builtin-off", switchDropped=bool(dropped),
                   steps=["the person: %s (one switch for every Studio of the user, RB1004)" % DR.BUILTIN_ENABLE])
    before = _file_state(place)
    shape0 = place_file_shape(place)

    def make_nonce():
        import uuid
        name = "MifBridgeSaveCheck_" + uuid.uuid4().hex[:12]
        r = T._rbx("create", parent=["ServerStorage"], className="Folder", name=name, props={}, tags=[])
        return (name, None) if r.get("ok") else (None, r.get("error") or "create failed")

    def drop_nonce(name):
        r = T._rbx("destroy", target=["ServerStorage", name])
        return None if r.get("ok") else (r.get("error") or "destroy failed")

    r = BS.save(place, make_nonce, drop_nonce, wait_seconds=wait_seconds)
    after = _file_state(place)
    shape1 = place_file_shape(place)
    r.update(studioPid=pid, before=before, after=after, shapeBefore=shape0, shapeAfter=shape1, builtInEnabled=enabled)
    if r.get("refusedRule") == "not-listed" and dropped:
        r.update(switchDropped=True, error=r.get("error", "") + ". Cause found: Studio's AssistantSettings file holds "
                 "no mcp-server key, so the switch is off (a reboot dropped it, RB2889)",
                 steps=["the person: %s, then call again" % DR.BUILTIN_ENABLE])
    if r.get("refusedRule") == "not-listed" and not dropped:
        start = next((st for p, st in L.running() if p == pid), None)
        on_at = DR.switch_on_after(start)
        if on_at is not None:
            r.update(studioStartedBeforeSwitch=True, error=r.get("error", "") + ". This Studio (pid %s) started at %s, "
                     "before the MCP switch was turned on at %s: a Studio started before it did not join the built-in "
                     "server until it was relaunched, so relaunch it (RB3064)" % (pid, time.strftime("%H:%M:%S", time.localtime(start)),
                                                         time.strftime("%H:%M:%S", time.localtime(on_at))),
                     steps=["save anything the person changed by hand, close this Studio and open the place again, "
                            "then call again"])
    if r.get("ok"):
        wrote = bool(before and after and (after["mtime"] > before["mtime"] or after["sha256"] != before["sha256"]))
        same_marker = "error" not in shape1 and shape1.get("marker") == shape0.get("marker")
        r["verified"] = wrote and "error" not in shape1 and same_marker
        if not wrote:
            r.update(ok=False, error="save_place answered Saved, but the file on disk did not change")
        elif "error" in shape1:
            r.update(ok=False, error="the saved file does not read back: %s" % shape1["error"])
        elif not same_marker:
            r.update(ok=False, error="the saved file's ServerStorage.MifBridgePort reads %s, not %s" % (
                shape1.get("marker"), shape0.get("marker")))
    return r


__all__ = sorted(n for n in list(globals()) if n.startswith("rbx_"))
