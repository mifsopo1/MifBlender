"""Roblox lane R3: a Blender action sampled in Roblox joint space, for rbx_anim_from_blender.

A Motor6D (or an AnimationConstraint) poses its child part by Transform, in the joint's own frame:
    Part1 = Part0 * C0 * Transform * C1:Inverse()          (Attachment0.CFrame / Attachment1.CFrame for a constraint)
so the joint frame on the child's side, F = Part0 * C0 * Transform, moves with the child. A Blender pose bone composes
the same way: bone = parent * (parent_rest^-1 * rest) * basis. If every mapped bone is given a fixed offset Q so that its
rest frame, brought into the rig's space, IS the joint's rest frame, the two chains are the same algebra and

    Transform(t) = (J_p^-1 * J_c)^-1 * G_p(t)^-1 * G_c(t),     G(t) = conv(bone(t)) * Q,   Q = conv(bone_rest)^-1 * J_rest

where J is a joint's rest frame in the rig (chained from the root part through C0/C1 or the attachments' CFrames, so a
rig left posed does not disturb it), conv brings Blender armature space to the rig's: Roblox x = Blender x, Roblox y =
Blender z, Roblox z = -Blender y, positions divided by the Blender units per stud and offset. A joint whose Part0 has no
bone (the root's HumanoidRootPart) keeps G_p at rest. The scale and offset are fitted from the bone heads to the joint
pivots (least squares, the rotation fixed); each bone's distance from its pivot is reported, since a rest pose that does
not match the rig's (a T-pose against an A-pose) cannot be retargeted by an offset.

Blender is read with typed ops only: render_info (frame rate, range, current frame), list_bones (rest head, tail and
roll in armature space; the rest rotation rebuilt as Bone.MatrixFromAxisRoll does, within 3e-5 of Blender's on 44
cases) and, at each frame, set_frame_range current then list_bones posed (each pose bone's armature-space matrix through
the depsgraph: constraints, drivers and actions applied). evaluate_at_frame cannot carry it: path_resolve does not
reach PoseBone.x_axis, and a Matrix path comes back as text. The scene's frame is put back after, and checked.

NEVER RAISE: every failure is {"ok": False, ...}.
"""
import math

# ---------------------------------------------------------------- 4x4 rigid frames as (R 3x3 rows, p 3)


def frame_from_cf(cf):
    """A CFrame (12 numbers: position, then the rotation's rows) -> (R, p)."""
    return [list(cf[3:6]), list(cf[6:9]), list(cf[9:12])], list(cf[:3])


def cf_from_frame(f):
    r, p = f
    return [p[0], p[1], p[2]] + r[0] + r[1] + r[2]


def mul(a, b):
    ra, pa = a
    rb, pb = b
    r = [[sum(ra[i][k] * rb[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    p = [sum(ra[i][k] * pb[k] for k in range(3)) + pa[i] for i in range(3)]
    return r, p


def inv(a):
    r, p = a
    rt = [[r[j][i] for j in range(3)] for i in range(3)]
    return rt, [-sum(rt[i][k] * p[k] for k in range(3)) for i in range(3)]


def frame_diff(a, b):
    """(position distance, largest rotation element difference) between two frames."""
    return (math.dist(a[1], b[1]), max(abs(a[0][i][j] - b[0][i][j]) for i in range(3) for j in range(3)))


def _norm(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n > 1e-12 else None


def mat_from_axis_roll(axis, roll):
    """A bone's rest rotation (columns x, y, z in armature space) from its head-to-tail axis and roll - Blender's
    Bone.MatrixFromAxisRoll: the smallest rotation taking +Y to the axis, then the roll about the axis. Rows returned."""
    n = _norm(axis)
    if n is None:
        return None
    x, y, z = n
    if 1 + y > 1e-6:
        # Rodrigues for Y -> n about Y x n = (z, 0, -x): R = I + K + K^2 / (1 + y)
        k = [[0, x, 0], [-x, 0, -z], [0, z, 0]]
        k2 = [[sum(k[i][m] * k[m][j] for m in range(3)) for j in range(3)] for i in range(3)]
        b = [[(1 if i == j else 0) + k[i][j] + k2[i][j] / (1 + y) for j in range(3)] for i in range(3)]
    else:
        b = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
    c, s = math.cos(roll), math.sin(roll)
    t = 1 - c
    rr = [[t * x * x + c, t * x * y - s * z, t * x * z + s * y],
          [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
          [t * x * z - s * y, t * y * z + s * x, t * z * z + c]]
    return [[sum(rr[i][m] * b[m][j] for m in range(3)) for j in range(3)] for i in range(3)]


# Blender armature axes -> Roblox: x stays, Blender z is Roblox y, Blender y is Roblox -z.
TO_RBX = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]


def to_rig(rot_rows, pos, k, offset):
    """A Blender armature-space frame -> the rig's space: rotation TO_RBX * R, position TO_RBX * p * k + offset."""
    r = [[sum(TO_RBX[i][m] * rot_rows[m][j] for m in range(3)) for j in range(3)] for i in range(3)]
    q = [sum(TO_RBX[i][m] * pos[m] for m in range(3)) for i in range(3)]
    return r, [q[i] * k + offset[i] for i in range(3)]


def fit_scale_offset(heads, pivots, scale=None):
    """Least squares p = k * TO_RBX(h) + o (k = studs per Blender unit). -> (k, o, residuals)."""
    q = [[sum(TO_RBX[i][m] * h[m] for m in range(3)) for i in range(3)] for h in heads]
    n = len(q)
    qm = [sum(v[i] for v in q) / n for i in range(3)]
    pm = [sum(v[i] for v in pivots) / n for i in range(3)]
    if scale:
        k = 1.0 / scale
    else:
        den = sum((v[i] - qm[i]) ** 2 for v in q for i in range(3))
        k = sum((q[j][i] - qm[i]) * (pivots[j][i] - pm[i]) for j in range(n) for i in range(3)) / den if den > 1e-12 else 1.0
    o = [pm[i] - k * qm[i] for i in range(3)]
    res = [math.dist([k * q[j][i] + o[i] for i in range(3)], pivots[j]) for j in range(n)]
    return k, o, res


# ---------------------------------------------------------------- the rig's rest joint frames, chained from the root


def _cf(v):
    if isinstance(v, dict) and "position" in v:
        return list(v["position"]) + list(v.get("rotation", [1, 0, 0, 0, 1, 0, 0, 0, 1]))
    return list(v) if isinstance(v, list) and len(v) == 12 else None


def rest_joint_frames(rig):
    """({part1: {joint, part0, rest (world frame of the joint at rest), childOffset}}, root part, error). Each joint's
    local frames are read (Motor6D C0 and C1, or an AnimationConstraint's Attachment0.CFrame and Attachment1.CFrame) and
    chained from the root part's CFrame, so the answer does not depend on any pose the rig was left in."""
    from mif_roblox import domain_anim_data as A
    jmap, root, err = A.rig_joints(rig)
    if err:
        return None, None, err
    if not jmap or not root:
        return None, None, {"ok": False, "ran": True, "error": "%s has no joints with a root part" % A._path(rig)}
    local = {}
    for part, j in jmap.items():
        if j["kind"] == "Motor6D":
            vals, err = A._props(j["path"], ["C0", "C1"])
            if err:
                return None, None, err
            c0, c1 = _cf(vals.get("C0")), _cf(vals.get("C1"))
        else:
            vals, err = A._props(j["path"], ["Attachment0", "Attachment1"])
            if err:
                return None, None, err
            a0, err0 = A._props(vals.get("Attachment0"), ["CFrame"])
            a1, err1 = A._props(vals.get("Attachment1"), ["CFrame"])
            if err0 or err1:
                return None, None, err0 or err1
            c0, c1 = _cf(a0.get("CFrame")), _cf(a1.get("CFrame"))
        if c0 is None or c1 is None:
            return None, None, {"ok": False, "ran": True, "error": "joint %s has no readable local frames" % j["path"]}
        local[part] = (frame_from_cf(c0), frame_from_cf(c1))
    rv, err = A._props(A._child(rig, root), ["CFrame"])
    if err:
        return None, None, err
    parts_world = {root: frame_from_cf(_cf(rv.get("CFrame")))}
    out = {}
    pending = dict(jmap)
    while pending:
        done = [p for p, j in pending.items() if j["part0"] in parts_world]
        if not done:
            return None, None, {"ok": False, "ran": True,
                                "error": "joints not reachable from the root part %s: %s" % (root, sorted(pending))}
        for p in done:
            j = pending.pop(p)
            c0, c1 = local[p]
            jw = mul(parts_world[j["part0"]], c0)
            parts_world[p] = mul(jw, inv(c1))
            out[p] = {"joint": j["joint"], "jointPath": j["path"], "part0": j["part0"], "rest": jw, "childOffset": c1}
    return out, root, None


# ---------------------------------------------------------------- the sampler


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def action_samples(ops, armature, rig, frames=None, frame_start=None, frame_end=None, step=1, mapping=None,
                   scale=None, max_pivot_offset=None):
    """The work behind bl_roblox_action_samples. ops(op, **kw) sends one MifBlender op."""
    if not isinstance(armature, str) or not armature:
        return {"ok": False, "ran": False, "error": "armature is the Blender armature object's name"}
    if mapping is not None and not (isinstance(mapping, dict) and all(isinstance(k, str) and isinstance(v, str)
                                                                      for k, v in mapping.items())):
        return {"ok": False, "ran": False, "error": "mapping is {bone: part}"}
    if scale is not None and (not _num(scale) or scale <= 0):
        return {"ok": False, "ran": False, "error": "scale (Blender units per stud) is a positive number"}
    if not isinstance(step, int) or isinstance(step, bool) or not 1 <= step <= 100:
        return {"ok": False, "ran": False, "error": "step is 1-100 frames"}
    ri = ops("render_info")
    if not ri.get("ok"):
        return dict(ri, ok=False, step="render_info")
    fps = (ri.get("fps") or 24) / (ri.get("fpsBase") or 1.0)
    if frames is None:
        fs = ri.get("frameStart") if frame_start is None else frame_start
        fe = ri.get("frameEnd") if frame_end is None else frame_end
        if not all(isinstance(x, int) and not isinstance(x, bool) for x in (fs, fe)) or fe < fs:
            return {"ok": False, "ran": False, "error": "frame_start and frame_end are whole frames, start <= end"}
        frames = list(range(fs, fe + 1, step))
        if frames[-1] != fe:
            frames.append(fe)
    if not isinstance(frames, list) or not frames or not all(isinstance(f, int) and not isinstance(f, bool) for f in frames) \
            or frames != sorted(set(frames)) or len(frames) > 600:
        return {"ok": False, "ran": False, "error": "frames rise, whole numbers, 1-600 of them"}
    joints, root, err = rest_joint_frames(rig)
    if err:
        return dict(err, step="rig")
    lb = ops("list_bones", object=armature, limit=10000)
    if not lb.get("ok"):
        return dict(lb, ok=False, step="list_bones")
    mapping = mapping or {}
    bones = {}
    unmapped = []
    for b in lb.get("bones") or []:
        part = mapping.get(b["name"], b["name"])
        if part in joints:
            if part in bones.values():
                return {"ok": False, "ran": False, "error": "two bones map to the joint of %s" % part}
            bones[b["name"]] = part
        else:
            unmapped.append(b["name"])
    if not bones:
        return {"ok": False, "ran": False, "refusedRule": "not-a-joint", "error": "no bone of %s maps to a joint of the rig"
                % armature, "bones": [b["name"] for b in lb.get("bones") or []][:100], "joints": sorted(joints)}
    rows = {b["name"]: b for b in lb["bones"]}
    rest_rot, heads, pivots = {}, [], []
    for bn, part in bones.items():
        h, t = rows[bn].get("headArmatureSpaceBU"), rows[bn].get("tailArmatureSpaceBU")
        r = mat_from_axis_roll([t[i] - h[i] for i in range(3)], rows[bn].get("roll") or 0.0)
        if r is None:
            return {"ok": False, "ran": False, "error": "bone %s has no length" % bn}
        rest_rot[bn] = r
        heads.append(h)
        pivots.append(joints[part]["rest"][1])
    k, off, res = fit_scale_offset(heads, pivots, scale)
    pivot_res = {bn: round(res[i], 6) for i, bn in enumerate(bones)}
    q = {}
    for bn, part in bones.items():
        g0 = to_rig(rest_rot[bn], rows[bn]["headArmatureSpaceBU"], k, off)
        q[bn] = mul(inv(g0), joints[part]["rest"])
    by_part = {part: bn for bn, part in bones.items()}
    out_bones = {part: [] for part in bones.values()}
    names = sorted(bones)
    started = ri.get("frameCurrent")
    try:
        for f in frames:
            sf = ops("set_frame_range", current=f)
            if not sf.get("ok"):
                return dict(sf, ok=False, step="set_frame_range", frame=f)
            pb = ops("list_bones", object=armature, posed=True, limit=10000)
            if not pb.get("ok"):
                return dict(pb, ok=False, step="list_bones posed", frame=f)
            posed = {b["name"]: b.get("posed") for b in pb.get("bones") or [] if b["name"] in bones}
            g = {}
            for bn in names:
                m = (posed.get(bn) or {}).get("matrixArmature")
                if not (isinstance(m, list) and len(m) == 4):
                    return {"ok": False, "ran": True, "error": "frame %s: bone %s has no posed matrix" % (f, bn),
                            "step": "list_bones posed"}
                rot = [[m[i][0], m[i][1], m[i][2]] for i in range(3)]
                g[bn] = mul(to_rig(rot, [m[0][3], m[1][3], m[2][3]], k, off), q[bn])
            for part, bn in by_part.items():
                j = joints[part]
                pbn = by_part.get(j["part0"])
                if pbn is not None:
                    rel_rest = mul(inv(joints[j["part0"]]["rest"]), j["rest"])
                    tcur = mul(inv(rel_rest), mul(inv(g[pbn]), g[bn]))
                else:
                    tcur = mul(inv(j["rest"]), g[bn])          # Part0 has no bone: it holds its rest frame
                out_bones[part].append([round(x, 7) for x in cf_from_frame(tcur)])
    finally:
        back = ops("set_frame_range", current=started) if isinstance(started, int) else {"ok": False}
    if not back.get("ok") or (back.get("after") or {}).get("current") != started:
        return {"ok": False, "ran": True, "step": "restore frame",
                "error": "the samples were read but the scene's frame was not put back to %s: %s" % (started, back.get("error"))}
    times = [round((f - frames[0]) / fps, 6) for f in frames]
    fixed = sorted({joints[p]["part0"] for p in by_part if joints[p]["part0"] not in by_part})
    worst = max(pivot_res.values()) if pivot_res else 0
    out = {"ok": True, "samples": {"times": times, "bones": out_bones}, "frames": frames, "fps": fps,
           "mapped": {bn: part for bn, part in bones.items()}, "unmappedBones": unmapped,
           "jointsWithoutBone": sorted(set(joints) - set(by_part)), "fixedParents": fixed,
           "studsPerBlenderUnit": round(k, 6), "offset": [round(x, 6) for x in off],
           "pivotResidual": pivot_res, "worstPivotResidual": round(worst, 6), "rootPart": root,
           "space": "joint (Motor6D.Transform / AnimationConstraint.Transform), Roblox axes, studs"}
    if max_pivot_offset is not None and worst > max_pivot_offset:
        out.update(ok=False, error="a bone head is %.4f studs from its joint pivot (more than %s): the armature's rest pose "
                                   "does not match the rig's" % (worst, max_pivot_offset))
    return out


def send(samples_out, rig, name="MifBlenderAction", parent="Workspace.MifScratch", tolerance=0.01, loop=False):
    """The samples through rbx_anim_from_blender (keys written, played and compared)."""
    from mif_roblox import domain_anim_data as A
    return A.rbx_anim_from_blender(samples_out["samples"], rig, name=name, parent=parent, axes="roblox",
                                   tolerance=tolerance, loop=loop)

