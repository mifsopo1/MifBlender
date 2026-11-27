"""An offline fit check for rigid accessories and held tools on a stand-in R15 body. Pure Python.

THE STAND-IN IS MEASURED, NOT DRAWN BY EYE. Every part size, part position and attachment below was read from Studio
0.748's default R15 body (Players:CreateHumanoidModelFromDescriptionAsync, rbx_mannequin) through the bridge (the internal notes): the HumanoidRootPart at the origin, the parts' centres relative to it, each attachment's CFrame in its part.
The head is drawn as an upright cylinder of the measured head's width and height (L91: a cube head gives false
poke-through at its corners and makes every helmet too big); every other part is a box of its measured size.

PLACEMENT FROM THE MANIFEST ALONE, as Studio places it. A piece's mesh origin is its attachment point (a tool's: the grip
point) and the manifest records center, the box centre relative to that origin. The Handle (a MeshPart, placed by its
box centre, L76) then sits at  part * attachment * CFrame.new(center), whose Attachment is CFrame.new(-center) - an
Accessory weld joins the two attachments; a Tool's Grip is CFrame.new(-center) on the hand's RightGripAttachment.
Each vertex is placed at  handle * (v - box centre of the file)  with v the file vertex on Roblox axes (fbx_mesh), so a
wrong center or a wrong turn in the manifest shows as a wrong placement rather than being hidden by the export.

THE TOOL POSE. A held tool is checked with the right arm raised forward 90 degrees about the shoulder joint
(UpperTorso.RightShoulderRigAttachment): CFrame.Angles(pi/2, 0, 0) takes the hanging arm's down (0, -1, 0) to the
front, (0, 0, -1). RightGripAttachment is turned -90 degrees about X (measured, RB2160), so a raised arm points the
Handle's -Z forward (L91).

THE FIT REPORT. For each piece, every stand-in surface point of a part whose box meets the piece's box is tested: the
segment from that part's centre to the point crossing one of the piece's triangles means the point lies beyond the
piece's inner surface - it pokes into or through the piece. The report lists the count per part and the worst points
with how far past the surface they lie. A held tool's grip passes through the hand holding it, so that hand is not
counted against the tool (notCounted says so).
"""
import math

from mif_roblox import fbx_mesh as F
from mif_roblox import kit as K

STUDIO = "0.748.0.7481526"
# part -> (size [x, y, z], centre relative to the HumanoidRootPart) - RB2160
PARTS = {
    "Head": ([1.159, 1.182, 1.1606], [0.0, 1.6744, 0.0]),
    "UpperTorso": ([1.9432, 1.698, 1.004], [0.0, 0.2496, 0.0]),
    "LowerTorso": ([1.9914, 0.4007, 1.004], [0.0, -0.7997, 0.0]),
    "LeftUpperArm": ([1.0011, 1.2416, 1.0019], [-1.4721, 0.4282, 0.0]),
    "LeftLowerArm": ([1.0011, 1.1175, 1.0019], [-1.4721, -0.2016, 0.0]),
    "LeftHand": ([0.9843, 0.3158, 1.0283], [-1.4721, -0.8653, 0.0]),
    "RightUpperArm": ([1.0011, 1.2416, 1.0019], [1.4721, 0.4282, 0.0]),
    "RightLowerArm": ([1.0011, 1.1175, 1.0019], [1.4721, -0.2016, 0.0]),
    "RightHand": ([0.9843, 0.3158, 1.0283], [1.4721, -0.8653, 0.0]),
    "LeftUpperLeg": ([0.9928, 1.3629, 0.9727], [-0.4979, -1.4714, 0.0]),
    "LeftLowerLeg": ([0.9928, 1.3007, 0.9728], [-0.4979, -2.3339, 0.0]),
    "LeftFoot": ([1.0089, 0.312, 1.0011], [-0.4979, -3.0364, 0.0]),
    "RightUpperLeg": ([0.9928, 1.3629, 0.9727], [0.4979, -1.4714, 0.0]),
    "RightLowerLeg": ([0.9928, 1.3007, 0.9728], [0.4979, -2.3339, 0.0]),
    "RightFoot": ([1.0089, 0.312, 1.0011], [0.4979, -3.0364, 0.0]),
}
CYLINDER = {"Head"}
_I = [1, 0, 0, 0, 1, 0, 0, 0, 1]
_RX_M90 = [1, 0, 0, 0, 0, 1, 0, -1, 0]           # CFrame.Angles(-pi/2, 0, 0): the grip attachments' turn (RB2160)
# attachment -> (part, position in the part, rotation 9) - RB2160; the accessory and tool attachments only
ATTACHMENTS = {
    "HatAttachment": ("Head", [0, 0.5896, 0], _I), "HairAttachment": ("Head", [0, 0.5896, 0], _I),
    "FaceFrontAttachment": ("Head", [0, 0, -0.5813], _I), "FaceCenterAttachment": ("Head", [0, -0.0924, 0], _I),
    "NeckAttachment": ("UpperTorso", [0, 0.849, 0], _I),
    "BodyFrontAttachment": ("UpperTorso", [0, -0.2122, -0.502], _I),
    "BodyBackAttachment": ("UpperTorso", [0, -0.2122, 0.502], _I),
    "LeftCollarAttachment": ("UpperTorso", [-0.9716, 0.849, 0], _I),
    "RightCollarAttachment": ("UpperTorso", [0.9716, 0.849, 0], _I),
    "WaistFrontAttachment": ("LowerTorso", [0, -0.2002, -0.502], _I),
    "WaistBackAttachment": ("LowerTorso", [0, -0.2002, 0.502], _I),
    "WaistCenterAttachment": ("LowerTorso", [0, -0.2002, 0], _I),
    "LeftShoulderAttachment": ("LeftUpperArm", [0, 0.6208, 0], _I),
    "RightShoulderAttachment": ("RightUpperArm", [0, 0.6208, 0], _I),
    "LeftGripAttachment": ("LeftHand", [0, -0.1578, 0], _RX_M90),
    "RightGripAttachment": ("RightHand", [0, -0.1578, 0], _RX_M90),
    "LeftFootAttachment": ("LeftFoot", [0.0184, -0.1596, -0.0503], _I),
    "RightFootAttachment": ("RightFoot", [-0.0184, -0.1596, -0.0503], _I),
}
RIGHT_SHOULDER = [0.9716, 0.2496 + 0.5975, 0.0]   # UpperTorso.RightShoulderRigAttachment, relative to the root
RIGHT_ARM = ("RightUpperArm", "RightLowerArm", "RightHand")
SAMPLE = 0.1                                       # studs between stand-in surface points
MAX_TRIANGLES = 200000


def _rx(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return [1, 0, 0, 0, c, -s, 0, s, c]


def body(pose="rest"):
    """{part: (R, p)} of the stand-in, the root at the origin; pose 'tool' raises the right arm forward 90 degrees."""
    out = {p: K.cf_new(c) for p, (_s, c) in PARTS.items()}
    if pose == "tool":
        turn = K.cf_mul(K.cf_mul(K.cf_new(RIGHT_SHOULDER), K.cf_from_components([0, 0, 0] + _rx(90))),
                        K.cf_new([-v for v in RIGHT_SHOULDER]))
        for p in RIGHT_ARM:
            out[p] = K.cf_mul(turn, out[p])
    return out


def surface_points(part, step=SAMPLE):
    """Points on the stand-in part's surface in its own frame: a box's six faces, or an upright cylinder's side and caps."""
    sx, sy, sz = PARTS[part][0]
    pts = []

    def rng(a):
        n = max(2, int(math.ceil(a / step)) + 1)
        return [-a / 2 + a * i / (n - 1) for i in range(n)]
    if part in CYLINDER:
        r = min(sx, sz) / 2
        n = max(12, int(math.ceil(2 * math.pi * r / step)))
        for y in rng(sy):
            for i in range(n):
                a = 2 * math.pi * i / n
                pts.append([r * math.cos(a), y, r * math.sin(a)])
        for rr in [r * k / 4 for k in range(4)]:
            m = max(1, int(math.ceil(2 * math.pi * rr / step)))
            for i in range(m):
                a = 2 * math.pi * i / m
                pts += [[rr * math.cos(a), sy / 2, rr * math.sin(a)], [rr * math.cos(a), -sy / 2, rr * math.sin(a)]]
        return pts
    for u in rng(sx):
        for v in rng(sz):
            pts += [[u, sy / 2, v], [u, -sy / 2, v]]
    for u in rng(sx):
        for v in rng(sy):
            pts += [[u, v, sz / 2], [u, v, -sz / 2]]
    for u in rng(sy):
        for v in rng(sz):
            pts += [[sx / 2, u, v], [-sx / 2, u, v]]
    return pts


def read_manifest(m):
    """{pieces: [{name, file, attachment | tool: true, center [x, y, z]}]} -> (pieces, None) or (None, why)."""
    if not isinstance(m, dict) or not isinstance(m.get("pieces"), list) or not m["pieces"]:
        return None, "the manifest is {pieces: [{name, file, attachment or tool true, center [x, y, z]}]}"
    out = []
    for i, p in enumerate(m["pieces"]):
        if not isinstance(p, dict) or not isinstance(p.get("file"), str) or not isinstance(p.get("name"), str):
            return None, "pieces[%d] needs name and file" % i
        c = p.get("center")
        if not (isinstance(c, list) and len(c) == 3 and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in c)):
            return None, "pieces[%d].center is [x, y, z]: the box centre relative to the mesh origin" % i
        tool = p.get("tool") is True
        att = "RightGripAttachment" if tool else p.get("attachment")
        if att not in ATTACHMENTS:
            return None, "pieces[%d].attachment %r is not one of the measured R15 attachments: %s" % (
                i, att, ", ".join(sorted(ATTACHMENTS)))
        out.append({"name": p["name"], "file": p["file"], "attachment": att, "tool": tool, "center": [float(v) for v in c]})
    return out, None


def handle_cframe(piece, parts):
    part, pos, rot = ATTACHMENTS[piece["attachment"]]
    return K.cf_mul(K.cf_mul(parts[part], K.cf_from_components(list(pos) + list(rot))), K.cf_new(piece["center"]))


def piece_triangles(sc, handle_cf):
    """The file's triangles placed at handle_cf, on Roblox axes: handle * (v - the file's box centre)."""
    _size, centre = F.scene_rbx_box(sc)
    tris = []
    for m in sc.meshes:
        pts = [F.rbx_point(p) for p in m["positions"]]
        wp = [K.cf_point(handle_cf, [p[0] - centre[0], p[1] - centre[1], p[2] - centre[2]]) for p in pts]
        for poly in m["polyList"]:
            for k in range(1, len(poly) - 1):
                tris.append((wp[poly[0]], wp[poly[k]], wp[poly[k + 1]]))
    return tris


def _sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _hit(o, d, tri, eps=1e-9):
    """Moller-Trumbore: t along o + t d where the ray meets tri, or None."""
    e1, e2 = _sub(tri[1], tri[0]), _sub(tri[2], tri[0])
    h = _cross(d, e2)
    a = _dot(e1, h)
    if abs(a) < eps:
        return None
    f = 1.0 / a
    s = _sub(o, tri[0])
    u = f * _dot(s, h)
    if u < 0 or u > 1:
        return None
    q = _cross(s, e1)
    v = f * _dot(d, q)
    if v < 0 or u + v > 1:
        return None
    t = f * _dot(e2, q)
    return t


class _Grid:
    def __init__(self, tris, cell=0.25):
        self.cell, self.cells = cell, {}
        for i, t in enumerate(tris):
            lo = [min(p[k] for p in t) for k in range(3)]
            hi = [max(p[k] for p in t) for k in range(3)]
            for x in range(int(math.floor(lo[0] / cell)), int(math.floor(hi[0] / cell)) + 1):
                for y in range(int(math.floor(lo[1] / cell)), int(math.floor(hi[1] / cell)) + 1):
                    for z in range(int(math.floor(lo[2] / cell)), int(math.floor(hi[2] / cell)) + 1):
                        self.cells.setdefault((x, y, z), []).append(i)

    def along(self, a, b):
        n = max(1, int(math.ceil(math.dist(a, b) / (self.cell / 3))))
        seen = set()
        for i in range(n + 1):
            p = [a[k] + (b[k] - a[k]) * i / n for k in range(3)]
            c = tuple(int(math.floor(p[k] / self.cell)) for k in range(3))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        seen.update(self.cells.get((c[0] + dx, c[1] + dy, c[2] + dz), ()))
        return seen


def _box_of(points):
    return [min(p[k] for p in points) for k in range(3)], [max(p[k] for p in points) for k in range(3)]


def fit_report(tris, parts, step=SAMPLE, worst=10):
    """Stand-in surface points beyond the piece's inner surface: per part, and the worst few."""
    if not tris:
        return {"ok": False, "error": "the piece has no triangles"}
    lo, hi = _box_of([p for t in tris for p in t])
    grid = _Grid(tris)
    per, rows, tested = {}, [], 0
    for part, cf in parts.items():
        size = PARTS[part][0]
        corners = [K.cf_point(cf, [sx * size[0] / 2, sy * size[1] / 2, sz * size[2] / 2])
                   for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        plo, phi = _box_of(corners)
        if any(phi[k] < lo[k] or plo[k] > hi[k] for k in range(3)):
            continue
        centre = cf[1]
        n = 0
        for q in surface_points(part, step):
            p = K.cf_point(cf, q)   # every point of the part: one poking out past the piece lies outside its box
            tested += 1
            d = _sub(p, centre)
            best = None
            for i in grid.along(centre, p):
                t = _hit(centre, d, tris[i])
                if t is not None and 1e-6 < t < 1 - 1e-6 and (best is None or t < best):
                    best = t
            if best is not None:
                n += 1
                rows.append({"part": part, "point": [round(v, 4) for v in p],
                             "pastSurface": round((1 - best) * math.sqrt(_dot(d, d)), 4)})
        per[part] = n
    rows.sort(key=lambda r: -r["pastSurface"])
    return {"ok": True, "tested": tested, "outside": sum(per.values()), "byPart": per, "worst": rows[:worst]}


def plan(pieces, load=F.load):
    """Each piece placed from the manifest alone: its handle CFrame, the file's box, the fit report."""
    rest, toolpose = body("rest"), body("tool")
    out = []
    for p in pieces:
        try:
            sc = load(p["file"])
        except (OSError, ValueError) as exc:
            return None, "%s did not read: %s" % (p["file"], exc)
        if not sc.rbx_axes():
            return None, "%s has axis settings %s, not the ones the import was measured on (RB1156)" % (
                p["file"], sc.axis_settings)
        size, centre = F.scene_rbx_box(sc)
        parts = toolpose if p["tool"] else rest
        hcf = handle_cframe(p, parts)
        tris = piece_triangles(sc, hcf)
        if len(tris) > MAX_TRIANGLES:
            return None, "%s has %d triangles, over %d" % (p["file"], len(tris), MAX_TRIANGLES)
        # a held tool's grip passes through the hand that holds it by design: that hand is not counted against it
        rep = fit_report(tris, {k: v for k, v in parts.items() if not (p["tool"] and k == "RightHand")})
        if p["tool"]:
            rep["notCounted"] = ["RightHand"]
        lo, hi = _box_of([q for t in tris for q in t]) if tris else ([0] * 3, [0] * 3)
        out.append(dict(p, handle=[round(v, 6) for v in K.cf_components(hcf)], meshSize=size, fileCentre=centre,
                        studsPerUnit=100.0 / sc.unit_scale_cm if sc.unit_scale_cm else None,   # UnitScaleFactor 0
                        placedBox={"min": [round(v, 4) for v in lo], "max": [round(v, 4) for v in hi]},
                        centerMatchesFile=all(abs(p["center"][k] - centre[k]) <= 0.01 for k in range(3)),
                        fit=rep, pose="tool" if p["tool"] else "rest"))
    return out, None
