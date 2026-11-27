"""A Blender action baked into per-bone world deltas and played on a skinned Bone rig in Roblox.

WHY. A Model upload brings no animation (RB1026), and rbx_anim_from_blender drives Motor6D joints, not the Bones of a
skinned mesh. What worked on a game: each clip baked in Blender as per-bone world deltas from rest, posed = D * rest,
written into a ModuleScript as packed text, and every frame each Bone set to

    Bone.Transform = rest^-1 * Dp^-1 * D * rest

with rest the Bone's WorldCFrame in the imported rig read while every Bone is at rest: a Bone's WorldCFrame carries its
ancestors' Transforms and leaves out only its own (RB1701), so rests are read before any Bone is posed and Dp the parent bone's delta (identity for a root). Proof of the formula: a bone's posed world frame is
parent_posed * Bone.CFrame * Transform; with parent_posed = Dp * parent_rest and rest = parent_rest * Bone.CFrame, the
Transform above makes it D * rest.

FROM BLENDER TO THE RIG. D is a rigid delta in Blender world space. The rig's space is Roblox (X, Y, Z) = k * Blender
(-x, z, y) + o: the measured import mapping (RB1156, RB1472; k = studs per Blender unit, 100 for Blender's default FBX
export, 1 for profile roblox). So D_rbx = (M R M^T, k M t + o - M R M^T o) with M the axis map. k and o are fitted by
least squares from the bones' rest heads in Blender to the Bones' rest positions in Studio (rotation fixed), each bone's
residual reported: a rig whose bones left the meshes' space (the import shrink, RB1472) shows it there and is refused.

THE PACKED TEXT, one clip:  line 1 "MIFCLIP1,<fps>,<frames>,<bones>", line 2 the bone names joined by "," (a name's ",", "%",
"]" and line breaks written %XX), then one line
per frame: for each bone qx,qy,qz,qw,px,py,pz of D_rbx (6 decimals for the quaternion, 4 for studs). The module parses
it once and slerps between frames.

RETARGET (retarget()): a clip made on one armature, played on another with the same bone names. Each bone's turn
relative to its parent is copied (the world delta's rotation, so the two skeletons' bone rolls do not matter), the
target keeps its own bone offsets, and only a head that moves against its parent's carry (the hip) moves, by one ratio
of the two hips' rest heights - what a game did to share one skeleton's clips across meshes.

NEVER RAISE from the public helpers; they answer (value, error) or dicts.
"""
import math

M = [[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]     # Blender (x, y, z) -> Roblox (-x, z, y); proper (det +1)


def _mm(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _mv(a, v):
    return [sum(a[i][k] * v[k] for k in range(3)) for i in range(3)]


def _t(a):
    return [[a[j][i] for j in range(3)] for i in range(3)]


def rigid(m4):
    """A 4x4 row-major matrix -> (R, p) with R orthonormalized (Blender matrices carry float noise and bone scale 1)."""
    R = [[float(m4[i][j]) for j in range(3)] for i in range(3)]
    p = [float(m4[i][3]) for i in range(3)]
    # Gram-Schmidt on the columns
    cols = [[R[0][j], R[1][j], R[2][j]] for j in range(3)]
    out = []
    for c in cols:
        for o in out:
            d = sum(c[k] * o[k] for k in range(3))
            c = [c[k] - d * o[k] for k in range(3)]
        n = math.sqrt(sum(v * v for v in c)) or 1.0
        out.append([v / n for v in c])
    R = [[out[j][i] for j in range(3)] for i in range(3)]
    return R, p


def mul(a, b):
    (Ra, pa), (Rb, pb) = a, b
    return _mm(Ra, Rb), [x + y for x, y in zip(_mv(Ra, pb), pa)]


def inv(a):
    R, p = a
    Rt = _t(R)
    return Rt, [-v for v in _mv(Rt, p)]


def quat(R):
    """(x, y, z, w) of a rotation matrix (rows), w >= 0."""
    tr = R[0][0] + R[1][1] + R[2][2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [(R[2][1] - R[1][2]) / s, (R[0][2] - R[2][0]) / s, (R[1][0] - R[0][1]) / s, 0.25 * s]
    elif R[0][0] > R[1][1] and R[0][0] > R[2][2]:
        s = math.sqrt(1.0 + R[0][0] - R[1][1] - R[2][2]) * 2
        q = [0.25 * s, (R[0][1] + R[1][0]) / s, (R[0][2] + R[2][0]) / s, (R[2][1] - R[1][2]) / s]
    elif R[1][1] > R[2][2]:
        s = math.sqrt(1.0 + R[1][1] - R[0][0] - R[2][2]) * 2
        q = [(R[0][1] + R[1][0]) / s, 0.25 * s, (R[1][2] + R[2][1]) / s, (R[0][2] - R[2][0]) / s]
    else:
        s = math.sqrt(1.0 + R[2][2] - R[0][0] - R[1][1]) * 2
        q = [(R[0][2] + R[2][0]) / s, (R[1][2] + R[2][1]) / s, 0.25 * s, (R[1][0] - R[0][1]) / s]
    if q[3] < 0:
        q = [-v for v in q]
    n = math.sqrt(sum(v * v for v in q)) or 1.0
    return [v / n for v in q]


def quat_mat(q):
    x, y, z, w = q
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def cf12(f):
    R, p = f
    return [p[0], p[1], p[2]] + [R[i][j] for i in range(3) for j in range(3)]


def from12(c):
    c = [float(v) for v in c]
    return [c[3:6], c[6:9], c[9:12]], c[0:3]


def rest_matrix(head, tail, roll, armature_world):
    """A bone's rest frame in Blender world: armature_world * (MatrixFromAxisRoll(tail - head, roll), head)."""
    from mif_roblox.domain_anim_blender import mat_from_axis_roll
    r = mat_from_axis_roll([tail[i] - head[i] for i in range(3)], roll or 0.0)
    if r is None:
        return None
    return mul(armature_world, (r, list(head)))


def to_rbx(D, k, o):
    """A Blender-world rigid delta -> the rig's space: (M R M^T, k M t + o - M R M^T o)."""
    R, t = D
    Rr = _mm(_mm(M, R), _t(M))
    kt = [k * v for v in _mv(M, t)]
    return Rr, [kt[i] + o[i] - _mv(Rr, o)[i] for i in range(3)]


def point_to_rbx(p, k, o):
    return [k * v + o[i] for i, v in enumerate(_mv(M, p))]


def fit(heads, targets, scale=None):
    """Least squares target = k * M h + o, M fixed. -> (k, o, residuals)."""
    q = [_mv(M, h) for h in heads]
    n = len(q)
    qm = [sum(v[i] for v in q) / n for i in range(3)]
    tm = [sum(v[i] for v in targets) / n for i in range(3)]
    if scale:
        k = float(scale)
    else:
        den = sum((v[i] - qm[i]) ** 2 for v in q for i in range(3))
        k = sum((q[j][i] - qm[i]) * (targets[j][i] - tm[i]) for j in range(n) for i in range(3)) / den if den > 1e-12 else 1.0
    o = [tm[i] - k * qm[i] for i in range(3)]
    res = [math.dist([k * q[j][i] + o[i] for i in range(3)], targets[j]) for j in range(n)]
    return k, o, res


def transform(rest, D, Dp=None):
    """Bone.Transform = rest^-1 * Dp^-1 * D * rest (frames as (R, p)); Dp None for a root bone."""
    inner = D if Dp is None else mul(inv(Dp), D)
    return mul(mul(inv(rest), inner), rest)


# ------------------------------------------------------------------------------------------------- retarget
def _axis_deg(a, b):
    """Degrees between two directions (a bone's head-to-tail in two skeletons' rests)."""
    na, nb = math.sqrt(sum(v * v for v in a)), math.sqrt(sum(v * v for v in b))
    if na < 1e-12 or nb < 1e-12:
        return None
    return math.degrees(math.acos(max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b)) / (na * nb)))))


def _order(parents):
    """Bone names parents first."""
    out, seen = [], set()

    def visit(n, trail=()):
        if n in seen or n in trail:
            return
        p = parents.get(n)
        if p is not None and p in parents:
            visit(p, trail + (n,))
        seen.add(n)
        out.append(n)
    for n in sorted(parents):
        visit(n)
    return out


def height(rest, tails, bone, up=2):
    """A bone's rest head above the skeleton's lowest head or tail, along world up (Blender Z)."""
    low = min([f[1][up] for f in rest.values()] + [t[up] for t in tails.values()])
    return rest[bone][1][up] - low


def retarget(src_rest, src_parents, src_posed, tgt_rest, tgt_parents, src_tails=None, tgt_tails=None, hip=None,
             ratio=None):
    """A source armature's clip carried onto a target armature with the same bone names.

    Frames are (R, p) in one space per skeleton (its world, rotation only: what bl_roblox_bone_clip calls Blender world).
    src_rest / tgt_rest: each bone's rest frame (head at p); src_posed: a list of frames, each {bone: posed frame}.

    Each bone's rotation relative to its parent is copied: the source's world delta D = posed * rest^-1 has rotation Rd,
    and Rd_parent^-1 Rd is the parent-relative turn, so the target bone's world delta rotation is the same Rd (a target
    bone the source lacks follows its parent). That is independent of how each skeleton rolled its bones. The target's
    own bone offsets are kept: a child's head is carried by its parent's delta from the TARGET's rest. A head that moves
    relative to where its parent carries it (the hip's translation, in practice) moves by ratio times the source's move;
    ratio is target hip height / source hip height (rest head above each skeleton's lowest point), one number for every
    clip, as the game did, or given. hip defaults to the bone whose head moves the most relative to its parent's carry.
    -> (frames: [{bone: target posed frame}], report) or (None, error)."""
    names = [n for n in _order(tgt_parents) if n in tgt_rest]
    shared = [n for n in names if n in src_rest]
    if not shared:
        return None, "no bone name is in both armatures"
    if not src_posed:
        return None, "the source has no frame"

    def sdelta(fr, n):
        return mul(fr[n], inv(src_rest[n]))

    def scarried(fr, n):
        p = src_parents.get(n)
        h = src_rest[n][1]
        if p is None or p not in src_rest or p not in fr:
            return h
        d = sdelta(fr, p)
        return [x + y for x, y in zip(_mv(d[0], h), d[1])]
    moves = {}
    for fr in src_posed:
        for n in shared:
            if n not in fr:
                continue
            c = scarried(fr, n)
            moves[n] = max(moves.get(n, 0.0), math.dist(fr[n][1], c))
    if hip is None:
        hip = max(shared, key=lambda n: (moves.get(n, 0.0), -names.index(n)))
    elif hip not in shared:
        return None, "hip %s is not a bone of both armatures" % hip
    hs = height(src_rest, src_tails or {}, hip)
    ht = height(tgt_rest, tgt_tails or {}, hip)
    if ratio is None:
        if hs <= 1e-9:
            return None, "the source's hip %s stands %.4g above its lowest point: no height ratio" % (hip, hs)
        ratio = ht / hs
    frames = []
    for fr in src_posed:
        D, out = {}, {}
        for n in names:
            p = tgt_parents.get(n)
            p = p if p in D else None
            if n in fr and n in src_rest:
                Rd = sdelta(fr, n)[0]
            else:
                Rd = D[p][0] if p else [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]
            h = tgt_rest[n][1]
            head = [x + y for x, y in zip(_mv(D[p][0], h), D[p][1])] if p else list(h)
            if n in fr and n in src_rest:
                u = [a - b for a, b in zip(fr[n][1], scarried(fr, n))]
                head = [head[i] + ratio * u[i] for i in range(3)]
            out[n] = (_mm(Rd, tgt_rest[n][0]), head)
            D[n] = mul(out[n], inv(tgt_rest[n]))
        frames.append(out)
    # the two rests' bone directions, bone by bone: a retarget copies turns, so a target posed differently at rest (an
    # A-pose under a T-pose clip) keeps that difference in every frame
    rest_deg = {}
    for n in shared:
        a = _mv(src_rest[n][0], [0, 1, 0])
        b = _mv(tgt_rest[n][0], [0, 1, 0])
        d = _axis_deg(a, b)
        if d is not None:
            rest_deg[n] = round(d, 3)
    report = {"hip": hip, "hipRatio": round(ratio, 6), "hipHeightSource": round(hs, 6), "hipHeightTarget": round(ht, 6),
              "hipMoveSource": round(moves.get(hip, 0.0), 6), "bonesCopied": len(shared),
              "targetOnly": [n for n in names if n not in src_rest][:50],
              "sourceOnly": sorted(set(src_rest) - set(names))[:50],
              "restAxisDegrees": {n: d for n, d in sorted(rest_deg.items(), key=lambda kv: -kv[1])[:10] if d > 1.0}}
    return frames, report


# ------------------------------------------------------------------------------------------------- packed text
# A bone name may hold the text's own separators (a comma, a line break) or the module's long-bracket close: those
# characters and "%" itself are written %XX, so every name round-trips; a name without them is written as it is.
NAME_ESCAPES = {c: "%%%02X" % ord(c) for c in "%,\r\n]"}


def pack_name(name):
    return "".join(NAME_ESCAPES.get(c, c) for c in str(name))


def unpack_name(text):
    import re  # noqa: PLC0415
    return re.sub(r"%([0-9A-Fa-f]{2})", lambda m: chr(int(m.group(1), 16)), text)


def pack(fps, bones, deltas):
    """deltas[frame][bone index] = (R, p) in the rig's space -> the packed text."""
    lines = ["MIFCLIP1,%s,%d,%d" % (repr(round(float(fps), 6)), len(deltas), len(bones)),
             ",".join(pack_name(b) for b in bones)]
    for fr in deltas:
        vals = []
        for R, p in fr:
            q = quat(R)
            vals += ["%.6f" % v for v in q] + ["%.4f" % v for v in p]
        lines.append(",".join(v.rstrip("0").rstrip(".") if "." in v else v for v in vals))
    return "\n".join(lines)


def unpack(text):
    """The packed text -> (fps, bones, deltas) or raises ValueError. CRLF or CR line ends read as LF."""
    rows = text.strip().splitlines()
    head = rows[0].split(",")
    if head[0] != "MIFCLIP1" or len(head) != 4 or len(rows) < 2:
        raise ValueError("not a MIFCLIP1 clip")
    fps, nf, nb = float(head[1]), int(head[2]), int(head[3])
    bones = [unpack_name(b) for b in rows[1].split(",")]
    if len(bones) != nb or len(rows) != nf + 2:
        raise ValueError("clip header says %d frames and %d bones, the text holds %d and %d" % (nf, nb, len(rows) - 2,
                                                                                              len(bones)))
    deltas = []
    for r in rows[2:]:
        v = [float(x) for x in r.split(",")]
        if len(v) != 7 * nb:
            raise ValueError("a frame holds %d numbers, not %d" % (len(v), 7 * nb))
        deltas.append([(quat_mat(v[i * 7:i * 7 + 4]), v[i * 7 + 4:i * 7 + 7]) for i in range(nb)])
    return fps, bones, deltas


LUAU = r'''--!strict
-- A baked bone clip, generated by MifBridge bl_roblox_bone_clip. Do not edit: rebake.
-- Each frame holds every bone's world delta from rest in the rig's space (posed = D * rest). Each Bone is set to
-- Transform = rest^-1 * Dp^-1 * D * rest, rest being the Bone's WorldCFrame read once before any pose.
local Clip = {}
Clip.name = %(name)s
Clip.parents = %(parents)s
Clip.text = [==[
%(text)s
]==]

local parsed: {fps: number, frames: number, bones: {string}, data: {{CFrame}}}? = nil

local function parse()
	if parsed then
		return parsed
	end
	local lines = string.split(Clip.text, "\n")
	local first = 1
	while lines[first] == "" do
		first += 1
	end
	local head = string.split(lines[first], ",")
	local fps, frames = tonumber(head[2]) :: number, tonumber(head[3]) :: number
	local bones = string.split(lines[first + 1], ",")
	for b = 1, #bones do
		-- a name's comma, line break, "]" and "%%" are packed as %%XX
		bones[b] = (string.gsub(bones[b], "%%%%(%%x%%x)", function(h: string): string
			return string.char(tonumber(h, 16) :: number)
		end))
	end
	local data = {}
	for f = 1, frames do
		local v = string.split(lines[first + 1 + f], ",")
		local row = {}
		for b = 1, #bones do
			local i = (b - 1) * 7
			local n = function(k: number): number
				return tonumber(v[i + k]) :: number
			end
			row[b] = CFrame.new(n(5), n(6), n(7), n(1), n(2), n(3), n(4))
		end
		data[f] = row
	end
	parsed = {fps = fps, frames = frames, bones = bones, data = data}
	return parsed
end

-- Every bone's delta at time t (seconds), slerped between baked frames; looped when loop is true.
function Clip.sample(t: number, loop: boolean?): {[string]: CFrame}
	local c = parse()
	local x = t * c.fps
	if loop then
		x = x %% c.frames
	end
	x = math.clamp(x, 0, c.frames - 1)
	local a = math.floor(x)
	local b = math.min(a + 1, c.frames - 1)
	local w = x - a
	local out = {}
	for i, name in ipairs(c.bones) do
		out[name] = c.data[a + 1][i]:Lerp(c.data[b + 1][i], w)
	end
	return out
end

local rests: {[Instance]: CFrame} = {}

-- Pose every Bone under rig named in the clip at time t. The rig must be at rest the first time: a Bone's WorldCFrame
-- carries its ancestors' Transforms (only its own is left out), so every rest is read before any Bone is posed.
function Clip.apply(rig: Instance, t: number, loop: boolean?)
	local d = Clip.sample(t, loop)
	local bones = {}
	for _, bone in ipairs(rig:GetDescendants()) do
		if bone:IsA("Bone") and d[bone.Name] then
			table.insert(bones, bone)
		end
	end
	for _, bone in ipairs(bones) do
		if not rests[bone] then
			rests[bone] = bone.WorldCFrame
		end
	end
	for _, bone in ipairs(bones) do
		do
			local rest = rests[bone]
			local p = Clip.parents[bone.Name]
			local inner = d[bone.Name]
			if p and d[p] then
				inner = d[p]:Inverse() * inner
			end
			bone.Transform = rest:Inverse() * inner * rest
		end
	end
end

return Clip
'''


def luau_module(name, text, parents):
    import json
    items = ", ".join("[%s] = %s" % (json.dumps(k), json.dumps(v)) for k, v in sorted(parents.items()) if v)
    text = text.replace("\r\n", "\n").replace("\r", "\n")   # the module splits on LF: a CR would stay on a name
    if "]==]" in text:
        raise ValueError("the packed text holds the long-bracket close")
    return LUAU % {"name": json.dumps(name), "parents": "{" + items + "}", "text": text}
