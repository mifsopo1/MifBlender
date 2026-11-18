"""Roads along a spline - build_road and road_report (2026-11-16).

build_road LAYS A ROAD THE WAY ROADS ARE DESIGNED, NOT AS A RIBBON SWEPT ALONG A CURVE. The points (or a curve's control
polygon) are the tangent polygon of a highway layout: each inner point is where two straights meet, and the road rounds
it with a spiral-circle-spiral - a clothoid whose curvature rises linearly from zero, a circular arc, and a clothoid back
to zero - so curvature is continuous everywhere and the road lies exactly on the straights between corners. Heights are
the points' heights joined by grades and rounded at every grade break by a parabolic vertical curve. Banking comes from
the corner's radius and speed by the superelevation formula, and is ramped at a fixed rate in and out of every corner.

ON THAT ALIGNMENT IT BUILDS A RACE TRACK'S CROSS-SECTION: the racing surface with a crown on straights, painted edge and
centre lines built into the surface mesh (their own material slot, no decal plane to fight it), a paved strip at each
edge where raised rumble-strip curbs sit on the apex and exit of every corner, a shoulder by type (grass, gravel or tarmac
run-off), and barriers - concrete wall, steel guardrail on posts, tyre stacks at impact points, catch fencing where a car
can fly. Optional jumps add a ramp and a landing shaped to the jump speed's trajectory.

EVERYTHING IS CHUNKED FOR AN ENGINE. The road is cut into chunks of at most 5 collision segments; each chunk is a surface
mesh, a curb mesh and a barrier mesh per side, each with two LODs the builder makes itself (every second, then every fourth
row - seam rows kept, so neighbouring chunks at different LODs still meet) under a LodGroup Empty (build_lod_chain's
shape), and UCX_ convex hulls - at most 10 per mesh. Hulls of one part share their end faces exactly, so collision is
continuous: a car slides along a barrier instead of snagging on a joint. Every piece sits at the world origin, so placing
every imported mesh at 0,0,0 rebuilds the track.

UVs: UVMap (UV0) is a world-scale planar projection on the surface (uvTile meters per repeat), so the asphalt never
stretches anywhere - a texture mapped along a curved ribbon must stretch at the inside of a tight corner (a 7 m half
width on a 15 m radius is 47%), which is why directional detail gets its own layer: RoadUV (UV1) runs along the road, u
0-1 across the racing surface and v in meters along it, for wear on the racing line and skid marks. Curbs and barriers,
which are narrow, take UV0 along their own arc length.

road_report MEASURES THE MESH. It reads the surface's station rows back from the vertices (attributes mif_s and mif_col),
not the design, and judges each corner and crest against the thresholds in _THRESHOLDS - every one names its source: a
real-world standard (cited) or a design choice for a driving game (said so). It also checks the collision against the
visual surface by ray casts, barrier coverage, curb coverage and UV stretch. Given points or a curve instead of a road,
it measures that alignment alone - a kinked spline is reported before a road is built on it.
"""
import bisect
import json
import math
import os
import re
import time

import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from .ops_common import (MifOpError, check_output_path, file_before, finite_float, get_object, reject_unknown,
                         remove_objects,
                         replaced_fields, select_only, selection_restore, selection_snapshot, take, take_bool,
                         take_float)
from .ops_create import _resolve_collection
from .ops_kit import _find_type, _make_lod_group, _resharpen, _sock, _tris
from .ops_mesh import FBX_EXPORT_ARGS, _UE_HULL_VERT_LIMIT, _build_hull, _hull_audit, _resolve_out_path

# MODULE-LEVEL AND PLAIN LITERALS: describe_op and parity_check read an accepted-key list statically.
BUILD_ROAD_PARAMS = (
    "name", "points", "curve", "closed", "width", "designSpeed", "sideFriction", "maxBank", "bankRate", "crown",
    "smoothing", "shoulder", "curbs", "barriers", "markings", "sections", "jumps", "chunkLength", "lods", "uvTile",
    "materials", "collision", "collection", "file", "overwrite",
)
ROAD_REPORT_PARAMS = ("road", "points", "curve", "closed", "designSpeed", "thresholds")

_ROAD_NOTES = {
    "radius": "a corner's radius goes on its point: points [[x, y, z], {\"co\": [x, y, z], \"radius\": 40}, ...]; "
              "without one the largest radius that fits between the neighbouring corners is used.",
    "speed": "designSpeed (km/h) is the whole road's; a corner's own speed goes on its point: {\"co\": [...], "
             "\"speed\": 60}.",
    "bank": "banking is computed from radius and speed (maxBank caps it, bankRate ramps it); sections cannot set it.",
    "banking": "banking is computed from radius and speed (maxBank caps it, bankRate ramps it).",
    "superelevation": "banking is computed from radius and speed (maxBank in degrees caps it).",
    "camber": "crown is the cross slope on straights, a fraction (0.02 = 2%).",
    "lanes": "the racing surface is one 'width' in meters; markings sets edge and centre lines.",
    "resolution": "rows are placed by curvature, bank and grade change (at most 2 m apart); chunkLength caps a chunk.",
    "lod": "lods true (the default) makes two LODs per part; false makes none.",
    "material": "materials takes a dict by role: asphalt, marking, curb, grass, gravel, runoff, wall, guardrail, "
                "tyre, fence.",
    "barrier": "barriers takes {\"left\": type, \"right\": type, \"tyres\": bool, \"fence\": \"auto\"|bool}; "
               "types wall, guardrail, none.",
    "shoulders": "shoulder takes {\"type\": grass|gravel|runoff, \"width\": m}.",
    "jump": "jumps takes a list of {\"at\": station m, ...}.",
}
_REPORT_NOTES = {
    "object": "road names the road's root (the name build_road was given); points or curve measure an alignment alone.",
    "name": "road names the road's root (the name build_road was given).",
    "speed": "designSpeed (km/h) is read from the road when it was built by build_road; pass it for points or a curve.",
}

# ------------------------------------------------------------------------------------------------- sources
# WHERE EVERY RULE COMES FROM. A standard is cited by document and chapter; a rule that is a choice for a driving game,
# not a road standard, says so - a racing game corners harder, banks steeper and jumps, which no highway manual covers.
_SOURCES = {
    "aashto": "AASHTO, A Policy on Geometric Design of Highways and Streets (the Green Book), 7th edition, 2018",
    "mutcd": "FHWA, Manual on Uniform Traffic Control Devices, 2009 edition, Part 3 (pavement markings)",
    "fia": "FIA International Sporting Code, Appendix O (circuit standards): a new circuit's track at least 12 m wide",
    "mgs": "Midwest Guardrail System (MASH): W-beam rail top at 787 mm (31 in), posts every 1,905 mm (6 ft 3 in)",
    "nj": "New Jersey concrete safety shape: 810 mm (32 in) tall, 610 mm base, 75 mm reveal, 55 then 84 degree faces",
    "uefn": "Epic, Creating Fortnite-Ready Assets in UEFN: three LODs, each half the previous; at most 10 UCX hulls",
    "design": "MifBlender design choice for a driving game - not a road standard",
    "physics": "MifBlender design choice derived from physics (stated with the rule)",
}

# ------------------------------------------------------------------------------------------------- the rules
_G = 9.81
_EYE_H, _OBJ_H = 1.08, 0.60          # aashto chapter 3, sight distance: driver's eye 1.08 m, object 0.60 m
_SPIRAL_P = 0.5                      # aashto chapter 3, spiral length L = sqrt(24 p R), p from 0.20 to 1.0 m: the middle
_DASH_ON, _DASH_OFF = 3.0, 9.0       # mutcd 3A: a broken line is 10 ft segments and 30 ft gaps (3 m and 9 m)
_EDGE_LINE_W, _CENTRE_LINE_W = 0.20, 0.15   # mutcd 3A: a normal line 100-150 mm, a wide line twice that - edge wide
_HULL_BUDGET = 10                    # uefn: "Try to limit your mesh to 10 UCX collision meshes or less."
_SEG_MAX = 12.0                      # design: a collision segment is at most 12 m
_TOL_INNER = 0.30                    # design: a hull may overhang the inside of a corner (off the road) by 30 cm
_TOL_BARRIER = 0.05                  # design: an outer barrier's hull stands at most 5 cm proud of its face
_TOL_SURFACE = 0.035                 # design: a surface hull stands at most 3.5 cm over the racing surface (measured)
_TOL_CURB = 0.03                     # design: an outside curb's hull stands at most 3 cm into the road
_ROW_MAX = 2.0                       # design: surface rows at most 2 m apart, 2 degrees of heading, 0.5 degree of bank
_ROW_DEG, _ROW_BANK, _ROW_GRADE = 2.0, 0.5, 0.005
_DENSE = 0.25                        # the alignment is sampled every 25 cm for banking and stations
_CURB_ROW = 0.15                     # design: curb rows every 15 cm carry a rumble rib every 60 cm
_RIB_PERIOD, _RIB_AMP, _STRIPE = 0.6, 0.012, 1.2
_TYRE_R, _TYRE_H, _TYRE_N, _TYRE_PITCH = 0.30, 0.22, 4, 0.62
_FENCE_H, _FENCE_POST = 3.5, 3.0
_MGS_TOP, _MGS_POST = 0.787, 1.905   # mgs
_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_SHOULDERS = ("grass", "gravel", "runoff")
_BARRIERS = ("wall", "guardrail", "none")
# a shoulder by type: (its slope away from the road, the drop at its inner edge) - design (a gravel trap sits lower)
_SHOULDER_SHAPE = {"grass": (0.04, 0.03), "gravel": (0.02, 0.06), "runoff": (0.02, 0.0)}
_DEPTH = {"wall": 0.61, "guardrail": 0.433, "none": 0.0}
_HEIGHT = {"wall": 0.81, "guardrail": _MGS_TOP, "none": 0.0}

# THE VALIDATOR'S THRESHOLDS: (default, unit, source key, what it says). road_report's 'thresholds' overrides values.
_THRESHOLDS = {
    "minRadius": (15.0, "m", "design", "the tightest corner; a 14 m road's inside edge then keeps an 8 m radius"),
    "maxCurvatureRate": (0.01, "1/m2", "design",
                         "curvature change per meter; a kink in a spline shows as a spike far over it (a 3 degree kink "
                         "measures about 0.013), while a spiral of aashto length into a 15 m hairpin is 0.005"),
    "maxGrade": (0.10, "fraction", "design", "steepest grade; highways stop near 6% (aashto ch. 8), a race track can "
                                             "climb harder"),
    "maxVerticalAccelG": (0.7, "g", "physics",
                          "v^2 times vertical curvature at the design speed; 1 g over a crest lifts the car, so crests "
                          "and sags stay 30% under it"),
    "bankToleranceDeg": (0.5, "deg", "design", "measured bank at an apex against the superelevation formula e/100 + f = "
                                               "V^2/(127 R) (aashto ch. 3), capped at maxBank"),
    "maxBankRateDegPerM": (0.3, "deg/m", "design", "how fast the bank changes along the road (build default 0.25)"),
    "crownMin": (0.015, "fraction", "aashto", "cross slope on straights: 1.5-2% for high-type pavements (aashto ch. 4)"),
    "crownMax": (0.025, "fraction", "aashto", "upper bound with 0.5% slack over aashto's 2%"),
    "widthMin": (12.0, "m", "fia", "racing surface width"),
    "widthMax": (16.0, "m", "design", "racing surface width upper bound"),
    "previewTime": (1.5, "s", "design", "sight distance over a crest must cover this many seconds at the design speed "
                                        "(eye 1.08 m, object 0.60 m - aashto ch. 3)"),
    "curbCoverage": (0.9, "fraction", "design", "share of a corner's apex (inside) and exit (outside) with a curb, for "
                                                "corners tighter than the road's curb radius"),
    "maxBarrierGap": (0.05, "m", "design", "longest gap in a barrier's collision outside declared open sections"),
    "maxCollisionDeviation": (0.05, "m", "design", "collision hull top against the visual surface, by ray cast"),
    "maxUvStretch": (0.03, "fraction", "design", "asphalt UVMap: largest scale error against uvTile on any face"),
    "maxPartUvStretch": (0.10, "fraction", "design", "curbs and barriers: their own along-the-part UVs"),
    "runUp": (40.0, "m", "design", "straight before a jump's ramp"),
    "runOut": (60.0, "m", "design", "straight and flat after a jump's landing"),
    "straightRadius": (1000.0, "m", "design", "a radius above this counts as straight for run-up, run-out and crown"),
    "flatGrade": (0.03, "fraction", "design", "a run-out's steepest grade"),
    "landingToleranceDeg": (5.0, "deg", "physics", "touchdown angle against the landing slope at the jump speed"),
    "maxJumpStep": (0.005, "m", "design", "a jump's entry and exit rows against the road surface under them - a step a "
                                          "tyre hits"),
    "maxJumpEntryAngleDeg": (1.0, "deg", "design", "the ramp's first face against the road's grade: it leaves the road "
                                                   "tangent"),
    "jumpWidthCoverage": (1.0, "fraction", "design", "a ramp's width against the paved road (racing surface and both "
                                                     "curb strips)"),
    "approachFaceToleranceDeg": (1.0, "deg", "design",
                                 "a face toward the oncoming car may be no steeper than the ramp's takeoff angle (on the "
                                 "ramp) or the landing's own slope (on the landing), plus this"),
}


def _src(key):
    return _SOURCES[_THRESHOLDS[key][2]]


# ============================================================================================== parsing
def _num(value, key, lo=None, hi=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MifOpError("'%s' must be a finite number, got %r. NOTHING was created." % (key, value))
    v = finite_float(value, key)
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        raise MifOpError("'%s' must be from %s to %s, got %r. NOTHING was created." % (key, lo, hi, value))
    return v


def _dict(params, key, allowed):
    raw = params.get(key)
    if raw is None or isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw
    if not isinstance(raw, dict):
        raise MifOpError("'%s' takes a dict (keys %s), got %r. NOTHING was created." % (key, ", ".join(allowed), raw))
    bad = sorted(set(raw) - set(allowed))
    if bad:
        raise MifOpError("'%s' does not take %s; its keys are %s. NOTHING was created."
                         % (key, ", ".join(bad), ", ".join(allowed)))
    return raw


def _parse_points(raw):
    """[[x, y, z] | [x, y] | {co, radius, speed, spiral}] -> list of corner dicts."""
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        raise MifOpError("'points' takes a list of at least 2 points, [x, y, z] or {\"co\": [x, y, z], \"radius\": m, "
                         "\"speed\": km/h, \"spiral\": m}. NOTHING was created.")
    out = []
    for i, p in enumerate(raw):
        extra = {}
        if isinstance(p, dict):
            bad = sorted(set(p) - {"co", "radius", "speed", "spiral"})
            if bad:
                raise MifOpError("points[%d] does not take %s; a point takes co, radius, speed, spiral. NOTHING was "
                                 "created." % (i, ", ".join(bad)))
            extra = p
            p = p.get("co")
        if not isinstance(p, (list, tuple)) or len(p) not in (2, 3):
            raise MifOpError("points[%d] must be [x, y] or [x, y, z], got %r. NOTHING was created." % (i, p))
        co = [_num(c, "points[%d]" % i) for c in p] + ([0.0] if len(p) == 2 else [])
        row = {"co": tuple(co), "radius": None, "speed": None, "spiral": None}
        for k, lo, hi in (("radius", 2.0, 100000.0), ("speed", 5.0, 500.0), ("spiral", 0.0, 2000.0)):
            if extra.get(k) is not None:
                row[k] = _num(extra[k], "points[%d].%s" % (i, k), lo, hi)
        out.append(row)
    for a, b in zip(out, out[1:]):
        if math.hypot(b["co"][0] - a["co"][0], b["co"][1] - a["co"][1]) < 0.5:
            raise MifOpError("two consecutive points are under 0.5 m apart in plan (%s and %s). NOTHING was created."
                             % (list(a["co"]), list(b["co"])))
    return out


def _curve_points(obj):
    """The control polygon of a curve's first spline, in world space: Bezier knots, poly and NURBS points."""
    if obj.type != "CURVE":
        raise MifOpError("'%s' is a %s, not a curve. NOTHING was created." % (obj.name, obj.type))
    splines = list(obj.data.splines)
    if not splines:
        raise MifOpError("the curve '%s' has no splines. NOTHING was created." % obj.name)
    sp = splines[0]
    mw = obj.matrix_world
    if sp.type == "BEZIER":
        cos = [mw @ bp.co for bp in sp.bezier_points]
    else:
        cos = [mw @ Vector(p.co[:3]) for p in sp.points]
    notes = []
    if len(splines) > 1:
        notes.append("'%s' has %d splines; the first was used." % (obj.name, len(splines)))
    return [[c.x, c.y, c.z] for c in cos], bool(sp.use_cyclic_u), sp.type, notes


# ============================================================================================== alignment
class _Fillet:
    """A spiral-circle-spiral turning by `delta` (signed, left positive) at radius R with spirals Ls, in a local frame
    starting at the origin heading +x. Positions come from a 5 cm midpoint integration of the exact heading."""

    def __init__(self, R, Ls, delta):
        self.R, self.sign, self.delta = R, (1.0 if delta > 0 else -1.0), abs(delta)
        self.Ls = min(Ls, R * self.delta)
        self.Lc = R * self.delta - self.Ls
        self.length = 2.0 * self.Ls + self.Lc
        n = max(8, int(math.ceil(self.length / 0.05)))
        self.h = self.length / n
        xs, ys, x, y = [0.0], [0.0], 0.0, 0.0
        for i in range(n):
            t = self.theta((i + 0.5) * self.h)
            x += self.h * math.cos(t)
            y += self.h * math.sin(t)
            xs.append(x)
            ys.append(y)
        self.xs, self.ys = xs, ys
        ex, ey = xs[-1], ys[-1]
        d = self.delta
        # the tangent length: where the outgoing straight (through the end, at heading delta) crosses the x axis
        self.T = ex - ey * math.cos(d) / math.sin(d)

    def theta(self, s):
        """unsigned heading at local station s"""
        k, Ls, Lc, L = 1.0 / self.R, self.Ls, self.Lc, self.length
        if Ls > 0 and s < Ls:
            return k * s * s / (2.0 * Ls)
        if s <= Ls + Lc:
            return k * Ls / 2.0 + k * (s - Ls)
        r = max(0.0, L - s)
        return self.delta - (k * r * r / (2.0 * Ls) if Ls > 0 else 0.0)

    def kappa(self, s):
        k, Ls, Lc = 1.0 / self.R, self.Ls, self.Lc
        if Ls > 0 and s < Ls:
            return self.sign * k * s / Ls
        if s <= Ls + Lc:
            return self.sign * k
        return self.sign * k * max(0.0, self.length - s) / Ls if Ls > 0 else self.sign * k

    def local(self, s):
        f = min(max(s / self.h, 0.0), len(self.xs) - 1.0)
        i = min(int(f), len(self.xs) - 2)
        t = f - i
        x = self.xs[i] + (self.xs[i + 1] - self.xs[i]) * t
        y = self.ys[i] + (self.ys[i + 1] - self.ys[i]) * t
        return x, self.sign * y, self.sign * self.theta(s), self.kappa(s)


def _spiral_len(R, p=_SPIRAL_P):
    return math.sqrt(24.0 * p * R)


class _Alignment:
    """The road's centreline: elements (straights and fillets) in order, station s from 0 to length, heights and bank.

    elements: dicts {kind, s0, length, x, y, th, fillet, corner}; th is the heading at the element's start."""

    def __init__(self, elements, closed, corners, kinks, pi_station):
        self.el, self.closed, self.corners, self.kinks = elements, closed, corners, kinks
        self.pi_station = pi_station       # point index -> its station (a corner's middle; an end's 0 or length)
        self.length = elements[-1]["s0"] + elements[-1]["length"]
        self._s0 = [e["s0"] for e in elements]
        self.vpis, self.vcurves, self.bank_d = [], [], None

    def _find(self, s):
        if self.closed:
            s = s % self.length
        else:
            s = min(max(s, 0.0), self.length)
        lo, hi = 0, len(self._s0) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._s0[mid] <= s + 1e-9:
                lo = mid
            else:
                hi = mid - 1
        return self.el[lo], s - self.el[lo]["s0"]

    def plan(self, s):
        """(x, y, heading, kappa) at station s."""
        e, sl = self._find(s)
        if e["kind"] == "line":
            return (e["x"] + sl * math.cos(e["th"]), e["y"] + sl * math.sin(e["th"]), e["th"], 0.0)
        lx, ly, lt, k = e["fillet"].local(sl)
        c, sn = math.cos(e["th"]), math.sin(e["th"])
        return (e["x"] + lx * c - ly * sn, e["y"] + lx * sn + ly * c, e["th"] + lt, k)

    def corner_at(self, s):
        e, _ = self._find(s)
        return e.get("corner")

    # ------------------------------------------------------------------------------------------ vertical
    def set_profile(self, vpis, radius_crest, radius_sag, smooth):
        """vpis: [(station, z)] in order. Grades join them; each break is rounded by a parabola of length R * |A|."""
        L = self.length
        pts = sorted(vpis)
        if self.closed:
            pts = [(pts[-1][0] - L, pts[-1][1])] + pts + [(pts[0][0] + L, pts[0][1])]
        self.vpts = pts
        grades = [(b[1] - a[1]) / max(1e-6, b[0] - a[0]) for a, b in zip(pts, pts[1:])]
        self.grades = grades
        curves = []
        for i in range(1, len(pts) - 1):
            A = grades[i] - grades[i - 1]
            if not smooth or abs(A) < 1e-5:
                curves.append(None)
                continue
            R = radius_crest if A < 0 else radius_sag
            Lv = R * abs(A)
            room = 0.9 * min(pts[i][0] - pts[i - 1][0], pts[i + 1][0] - pts[i][0])
            curves.append({"s": pts[i][0], "z": pts[i][1], "A": A, "L": min(Lv, room), "asked": Lv,
                           "gin": grades[i - 1], "kind": "crest" if A < 0 else "sag"})
        self.vcurves = curves
        # neighbours must not overlap: shrink both in proportion
        for i in range(len(curves) - 1):
            a, b = curves[i], curves[i + 1]
            if a and b:
                gap = b["s"] - a["s"]
                if a["L"] / 2 + b["L"] / 2 > 0.95 * gap:
                    f = 0.95 * gap / (a["L"] / 2 + b["L"] / 2)
                    a["L"] *= f
                    b["L"] *= f

    def z(self, s):
        if self.closed:
            s = s % self.length
        pts = self.vpts
        lo, hi = 0, len(pts) - 2
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if pts[mid][0] <= s:
                lo = mid
            else:
                hi = mid - 1
        a = pts[lo]
        z = a[1] + self.grades[lo] * (s - a[0])
        offs = (0.0, self.length, -self.length) if self.closed else (0.0,)
        for c in self.vcurves:
            if not c:
                continue
            for off in offs:
                cs = c["s"] + off
                if abs(s - cs) < c["L"] / 2:
                    x = s - (cs - c["L"] / 2)
                    return c["z"] - c["gin"] * c["L"] / 2 + c["gin"] * x + c["A"] / (2 * c["L"]) * x * x
        return z

    def grade(self, s, d=0.25):
        return (self.z(s + d) - self.z(s - d)) / (2 * d)

    # ------------------------------------------------------------------------------------------ bank
    def set_bank(self, design_speed, friction, max_bank_deg, rate_deg, smooth):
        """Signed bank in degrees on a 25 cm grid: aashto's e = V^2/(127 R) - f, capped at maxBank, then ramped at
        rate_deg per meter by a forward and backward pass (a cone dilation: full bank is kept through every corner and
        the ramps start before it). An S-bend whose ramps overlap is clamped to the rate afterwards, which can lower its
        peaks - road_report then says so against the formula."""
        n = max(2, int(math.ceil(self.length / _DENSE)))
        self.ds = self.length / n
        m = n if self.closed else n + 1
        e_max = math.tan(math.radians(max_bank_deg))
        tgt = []
        for i in range(m):
            s = i * self.ds
            _x, _y, _t, k = self.plan(s)
            corner = self.corner_at(s)
            v = (corner or {}).get("speed") or design_speed
            e = min(max(v * v * abs(k) / 127.0 - friction, 0.0), e_max) if smooth else 0.0
            tgt.append(math.copysign(math.degrees(math.atan(e)), k) if e > 0 else 0.0)
        r = rate_deg * self.ds

        def cone(vals, sign):
            a = [max(0.0, sign * v) for v in vals]
            rounds = 2 if self.closed else 1
            for _ in range(rounds):
                for i in range(1, m):
                    a[i] = max(a[i], a[i - 1] - r)
                if self.closed:
                    a[0] = max(a[0], a[-1] - r)
                for i in range(m - 2, -1, -1):
                    a[i] = max(a[i], a[i + 1] - r)
                if self.closed:
                    a[-1] = max(a[-1], a[0] - r)
            return [sign * v for v in a]

        pos, neg = cone(tgt, 1.0), cone(tgt, -1.0)
        b = [p + q for p, q in zip(pos, neg)]
        for _ in range(3 if self.closed else 1):
            for i in range(1, m):
                b[i] = min(max(b[i], b[i - 1] - r), b[i - 1] + r)
            if self.closed:
                b[0] = min(max(b[0], b[-1] - r), b[-1] + r)
            for i in range(m - 2, -1, -1):
                b[i] = min(max(b[i], b[i + 1] - r), b[i + 1] + r)
        self.bank_d, self.bank_target = b, tgt
        self.n_dense = m

    def bank(self, s):
        if self.closed:
            s = s % self.length
        f = min(max(s / self.ds, 0.0), self.n_dense - 1.0)
        i = int(f)
        j = (i + 1) % self.n_dense if self.closed else min(i + 1, self.n_dense - 1)
        t = f - i
        return self.bank_d[i] * (1 - t) + self.bank_d[j] * t


def _layout(pis, closed, smooth, design_speed):
    """The horizontal alignment from the tangent polygon `pis` (corner dicts). Returns an _Alignment."""
    n = len(pis)
    P = [(p["co"][0], p["co"][1]) for p in pis]
    if closed:
        if n < 3:
            raise MifOpError("a closed road needs at least 3 points. NOTHING was created.")
        if math.hypot(P[0][0] - P[-1][0], P[0][1] - P[-1][1]) < 0.5:
            raise MifOpError("a closed road's last point repeats its first; leave it off - closed joins them. NOTHING "
                             "was created.")
    idx = list(range(n)) if closed else list(range(1, n - 1))

    def edge(i, j):
        dx, dy = P[j][0] - P[i][0], P[j][1] - P[i][1]
        return math.hypot(dx, dy), math.atan2(dy, dx)

    def nxt(i):
        return (i + 1) % n

    def prv(i):
        return (i - 1) % n

    elements, corners, kinks = [], [], []

    if not smooth:
        # THE POLYLINE AS GIVEN: straights meeting at kinks. For comparison and for road_report to find the kinks.
        s = 0.0
        order = list(range(n)) + ([0] if closed else [])
        for a, b in zip(order, order[1:]):
            ln, th = edge(a, b)
            elements.append({"kind": "line", "s0": s, "length": ln, "x": P[a][0], "y": P[a][1], "th": th,
                             "pi": a})
            if s > 0 or closed:
                kinks.append(s)
            s += ln
        stations = {e["pi"]: e["s0"] for e in elements}
        if not closed:
            stations[n - 1] = s
        return _Alignment(elements, closed, corners, kinks, stations)

    fil = {}
    for i in idx:
        l_in, th_in = edge(prv(i), i)
        l_out, th_out = edge(i, nxt(i))
        d = math.atan2(math.sin(th_out - th_in), math.cos(th_out - th_in))
        if abs(d) < 1e-4:
            continue
        if abs(d) > math.radians(175.0):
            raise MifOpError("the road turns %.1f degrees at point %d; at most 175 degrees at one point - add a point "
                             "to split the turn. NOTHING was created." % (math.degrees(abs(d)), i))
        given = pis[i]["radius"]
        spiral = pis[i]["spiral"]

        def make(R):
            return _Fillet(R, spiral if spiral is not None else _spiral_len(R), d)

        if given is not None:
            f = make(given)
        else:
            share_in = 0.5 if (closed or prv(i) != 0) else 1.0
            share_out = 0.5 if (closed or nxt(i) != n - 1) else 1.0
            room = 0.98 * min(l_in * share_in, l_out * share_out)
            lo, hi = 2.0, 20000.0
            if make(lo).T > room:
                raise MifOpError("point %d's straights are too short to round even a 2 m radius (%.2f m of room); "
                                 "move the points apart. NOTHING was created." % (i, room))
            for _ in range(60):
                mid = math.sqrt(lo * hi)
                if make(mid).T <= room:
                    lo = mid
                else:
                    hi = mid
            f = make(lo)
        fil[i] = f
    # every straight must hold both tangents
    for i in (range(n) if closed else range(n - 1)):
        j = nxt(i)
        ln, _ = edge(i, j)
        need = (fil[i].T if i in fil else 0.0) + (fil[j].T if j in fil else 0.0)
        if need > ln + 1e-6:
            raise MifOpError("the straight from point %d to point %d is %.2f m and its two corners need %.2f m of "
                             "tangent; lower a radius or move the points. NOTHING was created." % (i, j, ln, need))

    def tangent_pt(i, side):
        """TS (side -1) or ST (side +1) of point i, or the point itself when it has no fillet."""
        if i not in fil:
            return P[i]
        if side < 0:
            ln, th = edge(prv(i), i)
            return (P[i][0] - fil[i].T * math.cos(th), P[i][1] - fil[i].T * math.sin(th))
        ln, th = edge(i, nxt(i))
        return (P[i][0] + fil[i].T * math.cos(th), P[i][1] + fil[i].T * math.sin(th))

    s = 0.0
    seq = (list(range(1, n)) + [0]) if closed else list(range(1, n))
    start = tangent_pt(0, +1) if closed else P[0]
    pi_station = {} if closed else {0: 0.0}
    for i in seq:
        ts = tangent_pt(i, -1)
        ln = math.hypot(ts[0] - start[0], ts[1] - start[1])
        _, th = edge(prv(i), i)
        if ln > 1e-6:
            elements.append({"kind": "line", "s0": s, "length": ln, "x": start[0], "y": start[1], "th": th})
            s += ln
        if i in fil:
            f = fil[i]
            corner = {"pi": i, "radius": f.R, "spiral": f.Ls, "arc": f.Lc, "deflectionDeg": math.degrees(f.delta),
                      "direction": "left" if f.sign > 0 else "right", "s0": s, "s1": s + f.length,
                      "speed": pis[i]["speed"] or design_speed, "radiusAuto": pis[i]["radius"] is None}
            elements.append({"kind": "fillet", "s0": s, "length": f.length, "x": ts[0], "y": ts[1], "th": th,
                             "fillet": f, "corner": corner})
            corners.append(corner)
            pi_station[i] = s + f.length / 2
            s += f.length
            start = tangent_pt(i, +1)
        else:
            pi_station[i] = s
            start = ts
    if not closed:
        pi_station[n - 1] = s
    return _Alignment(elements, closed, corners, kinks, pi_station)


# ============================================================================================== cross-section
class _Section:
    """Per-side configuration along the road: shoulder type, barrier type, curbs on/off, tyres, fence - the defaults,
    then the automatic places (tyres at impact points, fences at jumps and crests), then the caller's sections."""

    def __init__(self):
        self.base = {}
        self.ranges = {1: [], -1: []}       # (s0, s1, key, value) in order of priority (later wins)

    def get(self, side, s, key):
        val = self.base[side][key]
        for a, b, k, v in self.ranges[side]:
            if k == key and a - 1e-6 <= s <= b + 1e-6:
                val = v
        return val


def _side_slopes(bank_deg, crown):
    """(left, right): height change per meter going outward on each side of the crown, from the signed bank."""
    x = abs(math.tan(math.radians(bank_deg)))
    inner = -max(crown, x)
    outer = x if x >= crown else (-crown + 2.0 * x)
    if bank_deg > 0:
        return inner, outer
    if bank_deg < 0:
        return outer, inner
    return -crown, -crown


class _Road:
    """Everything build_road needs to place a vertex: the alignment, the cross-section sizes, the section table."""

    def __init__(self, al, cfg, sect):
        self.al, self.cfg, self.sect = al, cfg, sect
        self.hw = cfg["width"] / 2.0
        self.cw = cfg["curbWidth"]
        self.jump_profiles = []          # (station, profile) of every full-width jump: the barriers ride up on them

    def frame(self, s):
        x, y, th, k = self.al.plan(s)
        return x, y, th, k, self.al.z(s), self.al.bank(s)

    def side_cols(self, s, side, slope):
        """[(code, n, h, slot_after)] outward on one side; h relative to the centre; slot_after is the role of the face
        from this column to the next one out."""
        hw, cw = self.hw, self.cw
        sh = self.sect.get(side, s, "shoulder")
        sw = self.sect.get(side, s, "shoulderWidth")
        ss, drop = _SHOULDER_SHAPE[sh]
        ss = min(slope, -ss)
        cols = []
        if self.cfg["edgeLines"]:
            cols.append((2, hw - _EDGE_LINE_W, slope * (hw - _EDGE_LINE_W), "marking"))
        cols.append((1, hw, slope * hw, "asphalt"))
        zs = slope * (hw + cw)
        cols.append((3, hw + cw, zs, sh))
        step = min(0.1, sw / 4.0)
        cols.append((4, hw + cw + step, zs - drop + ss * step, sh))
        cols.append((5, hw + cw + sw, zs - drop + ss * sw, None))
        return cols

    def columns(self, s, dash_on):
        """The surface's columns across, left (+n) to right: [(code, n, h, role_of_face_to_the_right)]."""
        sl, sr = _side_slopes(self.al.bank(s), self.cfg["crown"])
        left = self.side_cols(s, 1, sl)
        right = self.side_cols(s, -1, sr)
        if self.cfg["centre"] != "none":
            w = _CENTRE_LINE_W / 2.0
            mid = [(6, w, sl * w), (0, 0.0, 0.0), (-6, -w, sr * w)]
            m = "marking" if (self.cfg["centre"] == "solid" or dash_on) else "asphalt"
            centre_roles = ["asphalt", m, m, "asphalt"]
        else:
            mid = [(0, 0.0, 0.0)]
            centre_roles = ["asphalt", "asphalt"]
        cols = [(c[0], c[1], c[2]) for c in reversed(left)] + mid + [(-c[0], -c[1], c[2]) for c in right]
        # the face between a side's columns i-1 (inner) and i (outer) takes column i-1's slot_after
        roles = ([left[i - 1][3] for i in range(len(left) - 1, 0, -1)] + centre_roles
                 + [right[i - 1][3] for i in range(1, len(right))])
        return cols, roles


def _dash_on(s0, s1):
    mid = 0.5 * (s0 + s1)
    return (mid % (_DASH_ON + _DASH_OFF)) < _DASH_ON


# ============================================================================================== mesh buffer
class _Buf:
    """Vertices, faces, per-face roles and per-corner UV layers, written to a mesh in one go."""

    def __init__(self):
        self.v, self.f, self.role, self.uv0, self.uv1, self.col, self.attr_s, self.attr_c = ([], [], [], [], [], [],
                                                                                             [], [])

    def vert(self, co, s=0.0, code=99):
        self.v.append(co)
        self.attr_s.append(s)
        self.attr_c.append(code)
        return len(self.v) - 1

    def face(self, ids, role, uv0=None, uv1=None, color=None, tile=1.0):
        """uv0 None projects the face onto its own plane at `tile` meters per repeat - exact for a planar face, so a
        post, a cap or an end face carries no stretch whichever way it faces."""
        if uv0 is None:
            uv0 = self.planar(ids, tile)
        self.f.append(tuple(ids))
        self.role.append(role)
        self.uv0.append(uv0)
        self.uv1.append(uv1)
        self.col.append(color)

    def planar(self, ids, tile):
        p = [Vector(self.v[i]) for i in ids]
        nrm = Vector((0.0, 0.0, 0.0))
        for k in range(len(p)):
            a, b = p[k], p[(k + 1) % len(p)]
            nrm += Vector(((a.y - b.y) * (a.z + b.z), (a.z - b.z) * (a.x + b.x), (a.x - b.x) * (a.y + b.y)))
        e1 = p[1] - p[0]
        if nrm.length < 1e-12 or e1.length < 1e-12:
            return [(0.0, 0.0)] * len(p)
        nrm.normalize()
        e1.normalize()
        e2 = nrm.cross(e1)
        return [((q - p[0]).dot(e1) / tile, (q - p[0]).dot(e2) / tile) for q in p]

    def to_mesh(self, name, slot_roles, materials, uv1_name=None, attrs=False, colors=False, sharp=30.0):
        me = bpy.data.meshes.new(name)
        me.from_pydata([tuple(c) for c in self.v], [], self.f)
        me.update()
        idx = {r: i for i, r in enumerate(slot_roles)}
        for r in slot_roles:
            me.materials.append(materials[r])
        me.polygons.foreach_set("material_index", [idx[r] for r in self.role])
        me.polygons.foreach_set("use_smooth", [True] * len(self.f))
        uv = me.uv_layers.new(name="UVMap")
        flat0 = []
        for fi, poly in enumerate(me.polygons):
            u = self.uv0[fi]
            for k in range(poly.loop_total):
                flat0.extend(u[k])
        uv.data.foreach_set("uv", flat0)
        if uv1_name:
            uv1 = me.uv_layers.new(name=uv1_name)
            flat1 = []
            for fi, poly in enumerate(me.polygons):
                u = self.uv1[fi]
                for k in range(poly.loop_total):
                    flat1.extend(u[k])
            uv1.data.foreach_set("uv", flat1)
        me.uv_layers.active = me.uv_layers["UVMap"]
        if colors:
            ca = me.color_attributes.new("Col", "BYTE_COLOR", "CORNER")
            flat = []
            for fi, poly in enumerate(me.polygons):
                c = self.col[fi] or (1.0, 1.0, 1.0)
                for _k in range(poly.loop_total):
                    flat.extend((c[0], c[1], c[2], 1.0))
            ca.data.foreach_set("color", flat)
        if attrs:
            a = me.attributes.new("mif_s", "FLOAT", "POINT")
            a.data.foreach_set("value", self.attr_s)
            b = me.attributes.new("mif_col", "INT", "POINT")
            b.data.foreach_set("value", self.attr_c)
        me.validate(clean_customdata=False)
        _resharpen(me, sharp)
        return me


def _srgb_to_linear(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


# ============================================================================================== materials
_MAT_LOOK = {
    # role: (default name, base color (linear), roughness)
    "asphalt": ("MifRoad_Asphalt", (0.030, 0.030, 0.032), 0.8),
    "marking": ("MifRoad_Marking", (0.80, 0.80, 0.78), 0.55),
    "curb": ("MifRoad_Curb", (1.0, 1.0, 1.0), 0.6),
    "grass": ("MifRoad_Grass", (0.06, 0.13, 0.03), 0.9),
    "gravel": ("MifRoad_Gravel", (0.20, 0.17, 0.12), 0.95),
    "runoff": ("MifRoad_Runoff", (0.06, 0.06, 0.065), 0.8),
    "wall": ("MifRoad_Concrete", (0.45, 0.45, 0.43), 0.8),
    "guardrail": ("MifRoad_Steel", (0.55, 0.57, 0.6), 0.35),
    "tyre": ("MifRoad_Tyre", (0.015, 0.015, 0.015), 0.75),
    "fence": ("MifRoad_Fence", (0.4, 0.42, 0.45), 0.5),
}
_ROLES = tuple(_MAT_LOOK)


def _asphalt_look(nt, bsdf, tex, color, rough):
    """Asphalt as an aggregate: dark binder between stones about 11 cm and 3 cm across (two Voronoi cell scales, the
    stone tops lighter and raised by a gentle bump), broad patches of older and newer surface (a 7 m noise), a high
    roughness and a lowered specular level - a smooth grey with a sheen is what asphalt does NOT look like (the first
    proof renders)."""
    big = tex("ShaderNodeTexVoronoi", 9.0)
    chip = tex("ShaderNodeTexVoronoi", 32.0)
    patch = tex("ShaderNodeTexNoise", 0.15, 2.0)

    def math_node(op, a, b=None, clamp=False):
        n = nt.nodes.new("ShaderNodeMath")
        n.operation = op
        n.use_clamp = clamp
        for i, v in enumerate((a, b)):
            if v is None:
                continue
            if isinstance(v, float):
                n.inputs[i].default_value = v
            else:
                nt.links.new(v, n.inputs[i])
        return n.outputs[0]

    # stone tops: 1 near a cell's centre (distance small), 0 in the binder between
    stones = math_node("SUBTRACT", 1.0, math_node("MULTIPLY", _sock(big.outputs, "Distance"), 2.2), clamp=True)
    chips = math_node("SUBTRACT", 1.0, math_node("MULTIPLY", _sock(chip.outputs, "Distance"), 2.6), clamp=True)
    stone = math_node("MAXIMUM", math_node("MULTIPLY", stones, 0.8), math_node("MULTIPLY", chips, 0.55))
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.15
    ramp.color_ramp.elements[0].color = tuple(c * 0.8 for c in color) + (1.0,)
    ramp.color_ramp.elements[1].position = 0.9
    ramp.color_ramp.elements[1].color = tuple(min(1.0, c * 3.2) for c in color) + (1.0,)
    nt.links.new(stone, ramp.inputs[0])
    # patches: older surface lighter and greyer, newer darker
    shade = nt.nodes.new("ShaderNodeMapRange")
    _sock(shade.inputs, "To Min").default_value = 0.8
    _sock(shade.inputs, "To Max").default_value = 1.25
    nt.links.new(_sock(patch.outputs, "Fac", "Factor"), _sock(shade.inputs, "Value"))
    tint = nt.nodes.new("ShaderNodeMixRGB") if hasattr(bpy.types, "ShaderNodeMixRGB") else None
    if tint is not None:
        tint.blend_type = "MULTIPLY"
        tint.inputs[0].default_value = 1.0
        nt.links.new(ramp.outputs[0], tint.inputs[1])
        nt.links.new(_sock(shade.outputs, "Result"), tint.inputs[2])
        nt.links.new(tint.outputs[0], _sock(bsdf.inputs, "Base Color"))
    else:
        nt.links.new(ramp.outputs[0], _sock(bsdf.inputs, "Base Color"))
    rmap = nt.nodes.new("ShaderNodeMapRange")
    _sock(rmap.inputs, "To Min").default_value = max(0.0, rough + 0.02)
    _sock(rmap.inputs, "To Max").default_value = min(1.0, rough + 0.15)
    nt.links.new(stone, _sock(rmap.inputs, "Value"))
    nt.links.new(_sock(rmap.outputs, "Result"), _sock(bsdf.inputs, "Roughness"))
    spec = bsdf.inputs.get("Specular IOR Level") or bsdf.inputs.get("Specular")
    if spec is not None:
        spec.default_value = 0.3
    bump = nt.nodes.new("ShaderNodeBump")
    _sock(bump.inputs, "Strength").default_value = 0.2
    _sock(bump.inputs, "Distance").default_value = 0.004
    nt.links.new(stone, _sock(bump.inputs, "Height"))
    nt.links.new(_sock(bump.outputs, "Normal"), _sock(bsdf.inputs, "Normal"))


def _road_material(role, name):
    """A look for judging the road in Blender; in an engine each slot takes the project's own material. Asphalt is an
    aggregate: dark binder, light stone speckle at two scales, broad patching and a polished racing line read through
    roughness; noise runs in world space so it continues across chunks. The curb reads its stripes from the Col
    attribute; the fence is a wire mesh cut by alpha."""
    _n, color, rough = _MAT_LOOK[role]
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = _find_type(nt.nodes, "BSDF_PRINCIPLED")
    _sock(bsdf.inputs, "Base Color").default_value = tuple(color) + (1.0,)
    _sock(bsdf.inputs, "Roughness").default_value = rough
    if role == "guardrail":
        _sock(bsdf.inputs, "Metallic").default_value = 0.9
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    pos = _sock(geo.outputs, "Position")

    def tex(kind, scale, detail=None):
        n = nt.nodes.new(kind)
        _sock(n.inputs, "Scale").default_value = scale
        if detail is not None and n.inputs.get("Detail") is not None:
            n.inputs["Detail"].default_value = detail
        nt.links.new(pos, _sock(n.inputs, "Vector"))
        return n

    if role in ("asphalt", "runoff"):
        _asphalt_look(nt, bsdf, tex, color, rough)
    elif role in ("gravel", "grass", "wall"):
        # LOW OCTAVES ON PURPOSE: a fine noise of 8 octaves reaches 0.4 mm detail, which a render samples at
        # centimeters and shows as streaks along the view at a grazing angle (the first proof renders); the stone
        # speckle is colour only, never bump, for the same reason
        speck = tex("ShaderNodeTexVoronoi", {"gravel": 14.0, "grass": 6.0}.get(role, 40.0))
        broad = tex("ShaderNodeTexNoise", 0.08 if role != "grass" else 0.3, 3.0)
        fine = tex("ShaderNodeTexNoise", 4.0 if role != "gravel" else 14.0, 2.0)
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        lift = {"asphalt": 2.6, "runoff": 2.2, "gravel": 1.5, "grass": 1.6, "wall": 1.25}[role]
        ramp.color_ramp.elements[0].position = 0.0
        ramp.color_ramp.elements[0].color = tuple(c * 0.75 for c in color) + (1.0,)
        ramp.color_ramp.elements[1].position = 1.0
        ramp.color_ramp.elements[1].color = tuple(min(1.0, c * lift) for c in color) + (1.0,)
        mix = nt.nodes.new("ShaderNodeMath")
        mix.operation = "MULTIPLY_ADD"
        mix.inputs[1].default_value = 0.55
        nt.links.new(_sock(speck.outputs, "Distance"), mix.inputs[0])
        nt.links.new(_sock(broad.outputs, "Fac", "Factor"), mix.inputs[2])
        clamp = nt.nodes.new("ShaderNodeMath")
        clamp.operation = "MULTIPLY"
        clamp.use_clamp = True
        clamp.inputs[1].default_value = 0.8
        nt.links.new(mix.outputs[0], clamp.inputs[0])
        nt.links.new(clamp.outputs[0], ramp.inputs[0])
        nt.links.new(ramp.outputs[0], _sock(bsdf.inputs, "Base Color"))
        rmap = nt.nodes.new("ShaderNodeMapRange")
        _sock(rmap.inputs, "To Min").default_value = rough - 0.2
        _sock(rmap.inputs, "To Max").default_value = min(1.0, rough + 0.1)
        nt.links.new(_sock(broad.outputs, "Fac", "Factor"), _sock(rmap.inputs, "Value"))
        nt.links.new(_sock(rmap.outputs, "Result"), _sock(bsdf.inputs, "Roughness"))
        bump = nt.nodes.new("ShaderNodeBump")
        _sock(bump.inputs, "Strength").default_value = 0.6 if role == "gravel" else 0.12
        _sock(bump.inputs, "Distance").default_value = 0.01 if role == "gravel" else 0.003
        nt.links.new(_sock(fine.outputs, "Fac", "Factor"), _sock(bump.inputs, "Height"))
        nt.links.new(_sock(bump.outputs, "Normal"), _sock(bsdf.inputs, "Normal"))
    elif role == "curb":
        attr = nt.nodes.new("ShaderNodeVertexColor")
        attr.layer_name = "Col"
        nt.links.new(_sock(attr.outputs, "Color"), _sock(bsdf.inputs, "Base Color"))
    elif role == "fence":
        wave = nt.nodes.new("ShaderNodeTexBrick") if False else nt.nodes.new("ShaderNodeTexChecker")
        _sock(wave.inputs, "Scale").default_value = 40.0
        uvn = nt.nodes.new("ShaderNodeUVMap")
        nt.links.new(_sock(uvn.outputs, "UV"), _sock(wave.inputs, "Vector"))
        alpha = _sock(bsdf.inputs, "Alpha")
        nt.links.new(_sock(wave.outputs, "Fac", "Factor"), alpha)
        try:
            mat.blend_method = "CLIP"
        except (AttributeError, TypeError):
            pass
    return mat


def _materials(raw, used):
    """{role: material}: the caller's named materials, else the default look made once and reused by name."""
    raw = raw or {}
    if not isinstance(raw, dict):
        raise MifOpError("'materials' takes a dict by role (%s), got %r. NOTHING was created." % (", ".join(_ROLES), raw))
    bad = sorted(set(raw) - set(_ROLES))
    if bad:
        raise MifOpError("'materials' does not take %s; its roles are %s. NOTHING was created."
                         % (", ".join(bad), ", ".join(_ROLES)))
    for role, name in raw.items():
        if not isinstance(name, str) or bpy.data.materials.get(name) is None:
            raise MifOpError("materials.%s names %r, which is not a material in this file. NOTHING was created."
                             % (role, name))
    return raw


def _resolve_materials(raw, used):
    out, made = {}, []
    for role in used:
        name = raw.get(role) or _MAT_LOOK[role][0]
        mat = bpy.data.materials.get(name)
        if mat is None:
            mat = _road_material(role, name)
            made.append(mat.name)
        out[role] = mat
    return out, made


# ============================================================================================== hulls
def _hull_object(name, points, coll, made):
    """One UCX_ convex hull of a point set, audited as create_collision_hull audits; a hull failing its audit raises."""
    bm = _build_hull([tuple(p) for p in points], 0.001)
    try:
        mesh = bpy.data.meshes.new(name)
        bm.to_mesh(mesh)
    finally:
        bm.free()
    audit = _hull_audit(mesh)
    bad = [k for k in ("boundaryEdges", "nonManifoldEdges", "looseVertices", "convexityViolations") if audit[k]]
    if audit["eulerCharacteristic"] != 2 or audit["volume"] <= 1e-9 or audit["vertices"] > _UE_HULL_VERT_LIMIT:
        bad.append("eulerCharacteristic %s, volume %s, vertices %s" % (audit["eulerCharacteristic"], audit["volume"],
                                                                      audit["vertices"]))
    if bad:
        bpy.data.meshes.remove(mesh)
        raise MifOpError("the collision hull %s failed its audit (%s) - a defect in build_road, not in the request."
                         % (name, ", ".join(bad)))
    obj = bpy.data.objects.new(name, mesh)
    obj.display_type = "WIRE"
    obj.hide_render = True
    coll.objects.link(obj)
    made.append(obj)
    return obj


# ============================================================================================== the op
def _parse_build(params):
    name =take(params, "name", default="Road", kind=str)
    if not _NAME_RE.match(name) or len(name) > 24:
        raise MifOpError("'name' must start with a letter and hold only letters, digits and underscores (an engine's "
                         "and Fab's rule for asset names), at most 24 characters; got %r. NOTHING was created." % name)
    pts_raw, curve_name = params.get("points"), take(params, "curve", kind=str)
    if (pts_raw is None) == (curve_name is None):
        raise MifOpError("build_road takes 'points' or 'curve' (a curve object's control polygon), one of them. "
                         "NOTHING was created.")
    notes = []
    closed = params.get("closed")
    if curve_name is not None:
        cobj = get_object(curve_name)
        raw, cyc, ctype, cnotes = _curve_points(cobj)
        notes += cnotes
        if ctype == "BEZIER":
            notes.append("'%s' is a Bezier curve: its knots are taken as the corners of the tangent polygon (handles "
                         "are not read) and every corner is rounded by a spiral-circle-spiral." % curve_name)
        pis = _parse_points(raw)
        closed = cyc if closed is None else take_bool(params, "closed")
    else:
        pis = _parse_points(pts_raw)
        closed = take_bool(params, "closed", default=False)
    cfg = {"name": name, "closed": closed, "notes": notes}
    cfg["width"] = _num(take_float(params, "width", default=14.0), "width", 3.0, 60.0)
    cfg["designSpeed"] = _num(take_float(params, "designSpeed", default=120.0), "designSpeed", 10.0, 500.0)
    cfg["sideFriction"] = _num(take_float(params, "sideFriction", default=0.15), "sideFriction", 0.0, 2.0)
    cfg["maxBank"] = _num(take_float(params, "maxBank", default=8.0), "maxBank", 0.0, 45.0)
    cfg["bankRate"] = _num(take_float(params, "bankRate", default=0.25), "bankRate", 0.01, 10.0)
    cfg["crown"] = _num(take_float(params, "crown", default=0.02), "crown", 0.0, 0.1)
    sm = _dict(params, "smoothing", ("spiral", "verticalRadius"))
    if isinstance(sm, str):
        raise MifOpError("'smoothing' takes true, false or {spiral, verticalRadius}, got %r. NOTHING was created." % sm)
    cfg["smooth"] = sm is not False
    sm = sm if isinstance(sm, dict) else {}
    cfg["spiralP"] = _num(sm.get("spiral", _SPIRAL_P), "smoothing.spiral", 0.05, 5.0)
    cfg["verticalRadius"] = (_num(sm["verticalRadius"], "smoothing.verticalRadius", 10.0, 1e6)
                             if sm.get("verticalRadius") is not None else None)
    sh = _dict(params, "shoulder", ("type", "width"))
    if sh is False:
        raise MifOpError("a road keeps a shoulder (its barriers stand at its outer edge); give it a narrow one, "
                         "{\"width\": 0.5}. NOTHING was created.")
    if isinstance(sh, str):
        sh = {"type": sh}
    sh = sh if isinstance(sh, dict) else {}
    cfg["shoulder"] = sh.get("type", "grass")
    if cfg["shoulder"] not in _SHOULDERS:
        raise MifOpError("shoulder type must be one of %s, got %r. NOTHING was created." % (", ".join(_SHOULDERS),
                                                                                          cfg["shoulder"]))
    cfg["shoulderWidth"] = _num(sh.get("width", 4.0), "shoulder.width", 0.5, 50.0)
    cu = _dict(params, "curbs", ("width", "height", "radius"))
    cfg["curbs"] = cu is not False
    cu = cu if isinstance(cu, dict) else {}
    cfg["curbWidth"] = _num(cu.get("width", 1.2), "curbs.width", 0.4, 3.0)
    cfg["curbHeight"] = _num(cu.get("height", 0.05), "curbs.height", 0.01, 0.2)
    cfg["curbRadius"] = _num(cu.get("radius", 300.0), "curbs.radius", 5.0, 5000.0)
    ba = _dict(params, "barriers", ("left", "right", "tyres", "fence"))
    if isinstance(ba, str):
        ba = {"left": ba, "right": ba}
    if ba is False:
        ba = {"left": "none", "right": "none", "tyres": False, "fence": False}
    ba = ba if isinstance(ba, dict) else {}
    for k in ("left", "right"):
        if ba.get(k, "guardrail") not in _BARRIERS:
            raise MifOpError("barriers.%s must be one of %s, got %r. NOTHING was created." % (k, ", ".join(_BARRIERS),
                                                                                            ba.get(k)))
    cfg["barrierL"], cfg["barrierR"] = ba.get("left", "guardrail"), ba.get("right", "guardrail")
    cfg["tyres"] = ba.get("tyres", True)
    cfg["fence"] = ba.get("fence", "auto")
    if not isinstance(cfg["tyres"], bool) or cfg["fence"] not in ("auto", True, False):
        raise MifOpError("barriers.tyres takes true or false and barriers.fence 'auto', true or false. NOTHING was "
                         "created.")
    mk = _dict(params, "markings", ("edge", "centre"))
    if mk is False:
        mk = {"edge": False, "centre": "none"}
    mk = mk if isinstance(mk, dict) else {}
    cfg["edgeLines"] = mk.get("edge", True)
    cfg["centre"] = mk.get("centre", "none")
    if not isinstance(cfg["edgeLines"], bool) or cfg["centre"] not in ("none", "dashed", "solid"):
        raise MifOpError("markings takes {\"edge\": bool, \"centre\": none|dashed|solid}. NOTHING was created.")
    cfg["chunkLength"] = _num(take_float(params, "chunkLength", default=60.0), "chunkLength", 10.0, 500.0)
    cfg["lods"] = take_bool(params, "lods", default=True)
    cfg["uvTile"] = _num(take_float(params, "uvTile", default=4.0), "uvTile", 0.1, 100.0)
    cfg["collision"] = take_bool(params, "collision", default=True)
    cfg["materials"] = _materials(params.get("materials"), None)
    sections = params.get("sections") or []
    if not isinstance(sections, list):
        raise MifOpError("'sections' takes a list of {from, to, side, barrier, shoulder, curbs, tyres, fence}. NOTHING "
                         "was created.")
    for i, sec in enumerate(sections):
        if not isinstance(sec, dict):
            raise MifOpError("sections[%d] must be a dict. NOTHING was created." % i)
        bad = sorted(set(sec) - {"from", "to", "side", "barrier", "shoulder", "curbs", "tyres", "fence"})
        if bad:
            raise MifOpError("sections[%d] does not take %s. NOTHING was created." % (i, ", ".join(bad)))
        if sec.get("side", "both") not in ("left", "right", "both"):
            raise MifOpError("sections[%d].side must be left, right or both. NOTHING was created." % i)
        _num(sec.get("from"), "sections[%d].from" % i, 0.0)
        _num(sec.get("to"), "sections[%d].to" % i, 0.0)
        if sec["to"] <= sec["from"]:
            raise MifOpError("sections[%d]: 'to' must be past 'from'. NOTHING was created." % i)
        if "barrier" in sec and sec["barrier"] not in _BARRIERS:
            raise MifOpError("sections[%d].barrier must be one of %s. NOTHING was created." % (i, ", ".join(_BARRIERS)))
        if "shoulder" in sec and sec["shoulder"] not in _SHOULDERS:
            raise MifOpError("sections[%d].shoulder must be one of %s. NOTHING was created." % (i, ", ".join(_SHOULDERS)))
        for k in ("curbs", "tyres", "fence"):
            if k in sec and not isinstance(sec[k], bool):
                raise MifOpError("sections[%d].%s takes true or false. NOTHING was created." % (i, k))
    cfg["sections"] = sections
    jumps = params.get("jumps") or []
    if not isinstance(jumps, list):
        raise MifOpError("'jumps' takes a list of {at, takeoff, height, speed, width, runUp, runOut}. NOTHING was "
                         "created.")
    cfg["jumps"] = []
    for i, j in enumerate(jumps):
        if not isinstance(j, dict) or j.get("at") is None:
            raise MifOpError("jumps[%d] must be a dict with 'at' (the ramp's start station, m). NOTHING was created."
                             % i)
        bad = sorted(set(j) - {"at", "takeoff", "height", "speed", "width", "runUp", "runOut"})
        if bad:
            raise MifOpError("jumps[%d] does not take %s. NOTHING was created." % (i, ", ".join(bad)))
        cfg["jumps"].append({
            "at": _num(j["at"], "jumps[%d].at" % i, 0.0),
            "takeoff": _num(j.get("takeoff", 14.0), "jumps[%d].takeoff" % i, 3.0, 40.0),
            "height": _num(j.get("height", 1.2), "jumps[%d].height" % i, 0.2, 10.0),
            "speed": _num(j.get("speed", cfg["designSpeed"] * 0.75), "jumps[%d].speed" % i, 10.0, 400.0),
            # NONE IS THE WHOLE ROAD: racing surface, curb strip and shoulders, out to the barriers, which ride up on it
            "width": (None if j.get("width") is None else
                      _num(j["width"], "jumps[%d].width" % i, 1.0, 2.0 * (cfg["width"] / 2.0 + cfg["curbWidth"]
                                                                          + cfg["shoulderWidth"]))),
            "runUp": _num(j.get("runUp", 40.0), "jumps[%d].runUp" % i, 0.0, 1000.0),
            "runOut": _num(j.get("runOut", 60.0), "jumps[%d].runOut" % i, 0.0, 1000.0)})
    cfg["file"] = take(params, "file", kind=str)
    cfg["overwrite"] = take_bool(params, "overwrite", default=True)
    if cfg["file"] is not None and not cfg["file"].lower().endswith(".fbx"):
        raise MifOpError("'file' must end in .fbx (LOD groups and UCX_ hulls are FBX conventions). NOTHING was "
                         "created.")
    return cfg, pis


def _vertical_radii(cfg):
    """aashto-style vertical curve radii for the design speed, unless the caller set one: a crest round enough that
    the car stays 30% under lifting off (R >= v^2 / 0.7 g) and that the sight distance over it covers previewTime
    (R >= S^2 / (2 (sqrt h1 + sqrt h2)^2), aashto ch. 3 with S < L), a sag under the same 0.7 g - each with 10% margin."""
    if cfg["verticalRadius"]:
        return cfg["verticalRadius"], cfg["verticalRadius"]
    v = cfg["designSpeed"] / 3.6
    lift = v * v / (_THRESHOLDS["maxVerticalAccelG"][0] * _G)
    S = v * _THRESHOLDS["previewTime"][0]
    sight = S * S / (2.0 * (math.sqrt(_EYE_H) + math.sqrt(_OBJ_H)) ** 2)
    return 1.1 * max(lift, sight), 1.1 * lift


_LAND_UNDER_DEG = 3.0      # physics: the landing is 3 degrees shallower than the descent, so the car meets it from above
_JUMP_EPS = 0.002          # a jump piece's top stands 2 mm over the road, so its entry and exit are flush, not coplanar


def _jump_shape(j, g=_G):
    """The profile of one jump in its own frame (x along the road from the ramp's start, y up from the road): a TABLE-TOP,
    ramp and landing one continuous surface with no face toward the approach anywhere.

    THE RAMP rises from the road on a circular arc - tangent to the road at its start, so the entry has no lip - to the
    takeoff angle, then a 60 cm straight lip at that angle, so the lip's last face measures the angle exactly.

    THE TABLE runs level at lip height from the lip to the knuckle. The first build put a landing with a 45-degree face
    and a deck taller than the lip there: from the ramp it read as a dark wall across the road (2026-11-16, flagged on
    the proof render, and road_report passed it). A car short of the knuckle now lands on the flat table.

    THE LANDING IS THE LOCUS OF TOUCHDOWNS from the knuckle (at lip height) to the road: every point's slope is 3
    degrees shallower than the descent of the trajectory passing through it, so every speed whose trajectory crosses
    it meets it 3 degrees shallow, from above. A landing that falls only the lip's height to the road takes a narrow
    band of speeds - about 1.5% of the jump speed, whatever the lip height (measured over lips of 1.5 and 2 m) - so the
    knuckle is placed to centre that band on the jump speed; slower cars land on the table, faster ones on the road past
    the foot. The answer reports the band."""
    a = math.radians(j["takeoff"])
    h = j["height"]
    lip = 0.6
    Rr = (h - lip * math.sin(a)) / (1.0 - math.cos(a))
    xarc = Rr * math.sin(a)
    xr = xarc + lip * math.cos(a)
    v = j["speed"] / 3.6
    ca, ta = math.cos(a), math.tan(a)
    under = math.radians(_LAND_UNDER_DEG)

    def speed_through(x, y):
        d = h + ta * (x - xr) - y
        return math.sqrt(g * (x - xr) ** 2 / (2.0 * ca * ca * d)) if d > 1e-9 and x > xr else None

    def locus(xk):
        curve, x, y, dx = [(xk, h)], xk, h, 0.05
        speeds = [speed_through(x, y)]
        while y > 0.0 and x < xk + 400.0:
            vv = speed_through(x, y)
            if vv is None:
                break
            desc = math.atan(-(ta - g * (x - xr) / (vv * vv * ca * ca)))
            y -= math.tan(max(math.radians(1.0), desc - under)) * dx
            x += dx
            curve.append((x, max(0.0, y)))
            speeds.append(speed_through(x, max(1e-3, y)))
        return curve, [s for s in speeds if s is not None]

    best = None
    for k in range(0, 81):
        f = 0.94 + k * 0.00075
        xk = _traj_cross(v * f, a, h, xr, h + 0.05)
        if xk is None or xk < xr + 1.0:
            continue
        curve, sp = locus(xk)
        if not sp:
            continue
        lo, hi = sp[0] / v, sp[-1] / v
        score = min(1.0 - lo, hi - 1.0)
        if best is None or score > best[0]:
            best = (score, xk, curve, sp, lo, hi)
    if best is None:
        raise MifOpError("no landing takes the jump at %.1f m (%.0f km/h off a %.1f m lip at %.1f degrees); raise the "
                         "jump speed or the takeoff angle. NOTHING was created." % (j["at"], j["speed"], h, j["takeoff"]))
    _score, xk, curve, sp, lo, hi = best
    xf = curve[-1][0]
    step = max(1, int(round(0.5 / 0.05)))
    pts = curve[::step]
    if pts[-1] != curve[-1]:
        pts.append(curve[-1])
    # the foot: the last stretch of the curve replaced by a quadratic into the road, so the exit is tangent too
    foot = min(1.5, 0.3 * (xf - xk))
    pts = [p for p in pts if p[0] <= xf - foot] or [pts[0]]
    x0, y0 = pts[-1]
    tb = y0 / max(1e-6, xf - x0)
    blend = []
    for i in range(1, 9):
        t = i / 8.0
        blend.append((x0 + 2 * (xf - x0) * t, max(0.0, y0 - tb * 2 * (xf - x0) * (t - t * t / 2.0))))
    blend[-1] = (blend[-1][0], 0.0)
    land = pts + blend
    rows_r = []
    n = max(8, int(math.ceil(xarc / 0.4)))
    for i in range(n + 1):
        xx = xarc * i / n
        th = math.asin(min(1.0, xx / Rr))
        rows_r.append((xx, Rr * (1.0 - math.cos(th))))
    rows_r.append((xr, h))
    # the table, in rows a couple of meters apart so a barrier riding it has rows to follow
    nt = max(1, int(math.ceil((xk - xr) / 2.0)))
    rows_r += [(xr + (xk - xr) * i / nt, h) for i in range(1, nt + 1)]
    xt = next((cx for (cx, _cy), s in zip(curve, [speed_through(cx, max(1e-3, cy)) for cx, cy in curve])
               if s is not None and s >= v), None)
    steep = max(math.degrees(math.atan2(ya - yb, xb - xa)) for (xa, ya), (xb, yb) in zip(land, land[1:]))
    return {"rampRows": rows_r, "landRows": land, "rampRadius": Rr, "lipX": xr, "knuckleX": xk, "knuckleHeight": h,
            "tableLength": xk - xr, "footX": land[-1][0], "landingDeg": steep, "touchdownX": xt,
            "speedWindow": (lo, hi), "length": land[-1][0]}


def _jump_profile_y(shapes_and_at, s):
    """The height a jump lifts the road at station s (0 outside every jump) - what a barrier beside it rides on."""
    for at, prof in shapes_and_at:
        x = s - at
        if prof[0][0] <= x <= prof[-1][0]:
            for (xa, ya), (xb, yb) in zip(prof, prof[1:]):
                if xa <= x <= xb:
                    t = 0.0 if xb == xa else (x - xa) / (xb - xa)
                    return ya + (yb - ya) * t + _JUMP_EPS
    return 0.0

def _traj_cross(v, a, h, xr, y):
    """x where the trajectory from the lip falls through height y (its descending branch), or None."""
    va, vb = v * math.cos(a), v * math.sin(a)
    disc = vb * vb + 2 * _G * (h - y)
    if disc < 0:
        return None
    return xr + va * (vb + math.sqrt(disc)) / _G


def _ranges_union(rs):
    rs = sorted(rs)
    out = []
    for a, b in rs:
        if out and a <= out[-1][1] + 1e-6:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def op_build_road(params):
    """Build a race road along points or a curve: a smoothed alignment (spiral-circle-spiral corners, rounded vertical
    curves), banking from radius and speed, crown, curbs, shoulders, barriers, markings and optional jumps, as LOD'd
    chunks with UCX_ hulls, and optionally write them to an FBX. Every piece sits at the world origin.

    params:
      name (str)            root name, letters/digits/underscores, default Road; pieces are <name>_C00 ...
      points (list)         the tangent polygon: [x, y, z] or {co, radius (m), speed (km/h), spiral (m)} per point
      curve (str)           or a curve object: its first spline's control points are the tangent polygon
      closed (bool)         a circuit (default false; a curve's own cyclic flag when 'curve' is given)
      width (m)             racing surface, default 14
      designSpeed (km/h)    default 120; a point's speed overrides it for its corner
      sideFriction          f in e = V^2/(127 R) - f, default 0.15
      maxBank (deg)         default 8
      bankRate (deg/m)      default 0.25
      crown (fraction)      cross slope on straights, default 0.02
      smoothing             true (default), false (the raw polyline, kinks and all), or {spiral: p m, verticalRadius: m}
      shoulder              {type: grass|gravel|runoff, width: m}, default grass 4 m
      curbs                 true (default), false, or {width, height, radius: curbs on corners tighter than this}
      barriers              {left, right: wall|guardrail|none, tyres: bool, fence: auto|bool}, default guardrail
      markings              {edge: bool, centre: none|dashed|solid}, default edge lines only
      sections (list)       {from, to, side: left|right|both, barrier, shoulder, curbs, tyres, fence} overrides
      jumps (list)          {at: station m, takeoff deg 14, height m 1.2, speed km/h, width m, runUp m, runOut m}
      chunkLength (m)       longest chunk, default 60
      lods (bool)           two LODs per part, default true
      uvTile (m)            meters per texture repeat, default 4
      materials (dict)      role -> existing material: asphalt, marking, curb, grass, gravel, runoff, wall, guardrail,
                            tyre, fence; missing roles get a default look
      collision (bool)      UCX_ hulls, default true
      collection (str)      parent collection for the road's own collection <name>
      file (str)            write every piece, LOD and hull to this .fbx
      overwrite (bool)      default true
    """
    reject_unknown(params, BUILD_ROAD_PARAMS, "build_road", _ROAD_NOTES)
    started = time.perf_counter()
    cfg, pis = _parse_build(params)
    name = cfg["name"]
    parent_coll = _resolve_collection(params)
    if bpy.data.collections.get(name) is not None or bpy.data.objects.get(name) is not None:
        raise MifOpError("a collection or object named '%s' already exists; a road owns its name. Pick another name "
                         "or remove it first. NOTHING was created." % name)
    taken = [o.name for o in bpy.data.objects if o.name.startswith(name + "_") or o.name.startswith("UCX_" + name + "_")]
    if taken:
        raise MifOpError("objects named like this road's pieces already exist (%s); remove them or pick another name. "
                         "NOTHING was created." % ", ".join(taken[:5]))
    path = None
    before_file = None
    if cfg["file"] is not None:
        path = _resolve_out_path(cfg["file"])
        check_output_path(cfg["file"], path, "exported")
        before_file = file_before(path)
        if before_file and not cfg["overwrite"]:
            raise MifOpError("%s exists and overwrite is false. NOTHING was created." % path)

    # ------------------------------------------------------------------ alignment
    al = _layout(pis, cfg["closed"], cfg["smooth"], cfg["designSpeed"])
    L = al.length
    rc, rs = _vertical_radii(cfg)
    vpis = [(al.pi_station[i], pis[i]["co"][2]) for i in sorted(al.pi_station)]
    al.set_profile(vpis, rc, rs, cfg["smooth"])
    al.set_bank(cfg["designSpeed"], cfg["sideFriction"], cfg["maxBank"], cfg["bankRate"], cfg["smooth"])
    road = _Road(al, cfg, _Section())

    # ------------------------------------------------------------------ the section table
    sect = road.sect
    for side, bk in ((1, "barrierL"), (-1, "barrierR")):
        sect.base[side] = {"shoulder": cfg["shoulder"], "shoulderWidth": cfg["shoulderWidth"], "barrier": cfg[bk],
                           "curbs": cfg["curbs"], "tyres": False, "fence": cfg["fence"] is True}
    curb_runs, tyre_zones, fence_zones = [], [], []
    for c in al.corners:
        if c["radius"] > cfg["curbRadius"]:
            continue
        inside = 1 if c["direction"] == "left" else -1
        # the apex zone (inside): where curvature is at least half the peak, 5 m either side; the exit (outside): from
        # the apex to where curvature falls under a quarter of the peak, plus 10 m
        a0 = c["s0"] + c["spiral"] * 0.5 - 5.0
        a1 = c["s1"] - c["spiral"] * 0.5 + 5.0
        apex = 0.5 * (c["s0"] + c["s1"])
        e1 = c["s1"] - c["spiral"] * 0.25 + 10.0
        curb_runs.append({"side": inside, "s0": a0, "s1": a1, "zone": "apex", "corner": c["pi"]})
        curb_runs.append({"side": -inside, "s0": apex, "s1": e1, "zone": "exit", "corner": c["pi"]})
        if cfg["tyres"] and c["radius"] <= 150.0:
            tyre_zones.append((-inside, apex, c["s1"] + 15.0))
    jshapes = []
    for j in cfg["jumps"]:
        shp = _jump_shape(j)
        jshapes.append(shp)
        if j["width"] is None:
            road.jump_profiles.append((j["at"], shp["rampRows"] + shp["landRows"][1:]))
        if cfg["fence"] == "auto":
            for side in (1, -1):
                fence_zones.append((side, j["at"] - 10.0, j["at"] + shp["length"] + 30.0))
        if not al.closed and j["at"] + shp["length"] > L:
            raise MifOpError("jump at %.1f m runs %.1f m and the road ends at %.1f m. NOTHING was created."
                             % (j["at"], shp["length"], L))
    if cfg["fence"] == "auto":
        for c in al.vcurves:
            if c and c["kind"] == "crest" and abs(c["A"]) >= 0.03:
                for side in (1, -1):
                    fence_zones.append((side, c["s"] - c["L"] / 2 - 10.0, c["s"] + c["L"] / 2 + 10.0))

    def clip(a, b):
        """[a, b] inside [0, L]; on a circuit, the parts that wrap past either end come back round."""
        out = []
        for sh in ((-L, 0.0, L) if al.closed else (0.0,)):
            lo, hi = max(0.0, a + sh), min(L, b + sh)
            if hi > lo:
                out.append((lo, hi))
        return out

    for side, a, b in tyre_zones:
        for a2, b2 in clip(a, b):
            sect.ranges[side].append((a2, b2, "tyres", True))
    for side, a, b in fence_zones:
        for a2, b2 in clip(a, b):
            sect.ranges[side].append((a2, b2, "fence", True))
    for sec in cfg["sections"]:
        sides = {"left": (1,), "right": (-1,), "both": (1, -1)}[sec.get("side", "both")]
        for side in sides:
            for k in ("barrier", "shoulder", "curbs", "tyres", "fence"):
                if k in sec:
                    sect.ranges[side].append((float(sec["from"]), float(sec["to"]), k, sec[k]))
    # curbs: the automatic runs, less the sections that turn curbs off
    # A CURB KEEPS ITS OWN STATIONS ON A CIRCUIT, even past 0 or the length (the first corner's exit runs over the
    # start), so its taper and stripes stay whole; each chunk takes it at whichever lap offset overlaps it (curb_pieces)
    runs = []
    for r in curb_runs:
        if al.closed:
            runs.append(dict(r))
        else:
            for a, b in clip(r["s0"], r["s1"]):
                runs.append(dict(r, s0=a, s1=b))
    merged = {1: [], -1: []}
    for side in (1, -1):
        merged[side] = _ranges_union([(r["s0"], r["s1"]) for r in runs if r["side"] == side])
    curb_list = []
    for side in (1, -1):
        for a, b in merged[side]:
            # cut out sections with curbs false
            pieces = [(a, b)]
            for sa, sb, k, v in sect.ranges[side]:
                if k == "curbs" and v is False:
                    nxt = []
                    for pa, pb in pieces:
                        if sb <= pa or sa >= pb:
                            nxt.append((pa, pb))
                        else:
                            if sa > pa:
                                nxt.append((pa, sa))
                            if sb < pb:
                                nxt.append((sb, pb))
                    pieces = nxt
            if not cfg["curbs"]:
                pieces = []
            for pa, pb in pieces:
                if pb - pa >= 2.0:
                    curb_list.append({"side": side, "s0": pa, "s1": pb})

    # ------------------------------------------------------------------ the inside of every corner must not fold
    for c in al.corners:
        inside = 1 if c["direction"] == "left" else -1
        sw = max(sect.get(inside, s, "shoulderWidth") for s in (c["s0"], c["s1"]))
        ext = road.hw + road.cw + sw + _DEPTH[sect.get(inside, 0.5 * (c["s0"] + c["s1"]), "barrier")] + 0.3
        if c["radius"] <= ext + 1.0:
            raise MifOpError("the corner at point %d has a %.1f m radius and the road reaches %.1f m to its inside (half "
                             "width, curb strip, shoulder, barrier); the inside edge would fold. Give it a radius over "
                             "%.0f m or a narrower shoulder there. NOTHING was created."
                             % (c["pi"], c["radius"], ext, ext + 1.0))

    # ------------------------------------------------------------------ segments and chunks
    forced = set()
    for side in (1, -1):
        for a, b, k, v in sect.ranges[side]:
            if k in ("barrier", "tyres", "fence", "shoulder"):
                forced.update((a % L if al.closed else a, b % L if al.closed else b))
    forced = sorted(s for s in forced if 0.0 < s < L)

    def seg_len(s):
        """A surface hull's length from s, MEASURED. Its top is the convex hull of every row in it, exact over a crest,
        but the road is not convex everywhere: a sag, a bank change (a twisted surface the hull spans flat across its
        diagonal) and a banked corner (a bowl - height rises with the radius) all leave the hull standing over the
        surface. Formulas for each were tried and under-read the combination (4.8 cm measured where they predicted
        2), so each candidate hull is built from rows every 50 cm - at least as dense as the real rows, so its top is
        at least as high - and ray cast against the surface (_hull_dev); a hull standing more than _TOL_SURFACE over
        it is shortened. The inside of a corner, off the road, may overhang the infield by _TOL_INNER."""
        best = _SEG_MAX
        for t in range(0, 13, 2):
            ss = s + t
            if not al.closed and ss > L:
                break
            _x, _y, _th, k = al.plan(ss)
            if abs(k) > 1e-6:
                R = 1.0 / abs(k)
                inside = 1 if k > 0 else -1
                ext = road.hw + road.cw + sect.get(inside, ss % L if al.closed else ss, "shoulderWidth")
                best = min(best, math.sqrt(8.0 * max(0.5, R - ext) * _TOL_INNER))
        best = max(1.0, min(best, (L - s) if not al.closed else best))
        while best > 1.0 and _hull_dev(road, s, s + best) > _TOL_SURFACE:
            best = max(1.0, best * 0.8)
        return best

    def barrier_len(s, side):
        """A barrier hull's length: on the outside of a corner its road face is concave toward the car, and the hull's
        chord stands proud of it by L^2 / (8 R) - kept under _TOL_BARRIER."""
        best = _SEG_MAX
        for t in range(0, 13, 3):
            ss = s + t
            _x, _y, _th, k = al.plan(ss)
            if abs(k) > 1e-6 and (1 if k > 0 else -1) == -side:
                R = 1.0 / abs(k) + road.hw + road.cw + sect.get(side, ss % L if al.closed else ss, "shoulderWidth")
                best = min(best, math.sqrt(8.0 * R * _TOL_BARRIER))
        return max(1.0, best)

    def curb_len(s, side):
        best = 6.0
        for t in (0.0, 3.0, 6.0):
            _x, _y, _th, k = al.plan(s + t)
            if abs(k) > 1e-6 and (1 if k > 0 else -1) == -side:
                best = min(best, math.sqrt(8.0 * (1.0 / abs(k) + road.hw) * _TOL_CURB))
        return max(0.6, best)

    cuts = [0.0]
    fi = 0
    while cuts[-1] < L - 1e-6:
        s = cuts[-1]
        nxt = min(L, s + seg_len(s))
        while fi < len(forced) and forced[fi] <= s + 1e-6:
            fi += 1
        if fi < len(forced) and forced[fi] < nxt - 0.05:
            nxt = forced[fi]
        if L - nxt < 1.0:
            nxt = L
        cuts.append(nxt)
    segs = list(zip(cuts, cuts[1:]))

    def curbs_in(a, b, side):
        out = []
        for c in curb_list:
            if c["side"] != side:
                continue
            for sh in ((-L, 0.0, L) if al.closed else (0.0,)):
                if c["s0"] + sh < b - 1e-6 and c["s1"] + sh > a + 1e-6:
                    out.append(dict(c, s0=c["s0"] + sh, s1=c["s1"] + sh))
        return out

    def plan(segs_c):
        """Every hull one chunk would hold, by part: a surface hull per segment; curb hulls per side cut by curb_len;
        barrier hulls per side, each component kind's segments merged up to barrier_len."""
        a, b = segs_c[0][0], segs_c[-1][1]
        out = {"surface": list(segs_c)}
        for side in (1, -1):
            pieces = []
            for c in curbs_in(a, b, side):
                lo, hi = max(a, c["s0"]), min(b, c["s1"])
                s = lo
                while s < hi - 0.05:
                    e = min(hi, s + curb_len(s, side))
                    if hi - e < 0.3:
                        e = hi
                    pieces.append((c, s, e))
                    s = e
            out[("curb", side)] = pieces
            out[("barrier", side)] = _group(_barrier_components(road, side, segs_c, None),
                                            lambda s0, kind, _side=side: barrier_len(s0, _side))
        return out

    chunks, cur = [], []
    for a, b in segs:
        if cur:
            trial = plan(cur + [(a, b)])
            over = any(len(v) > _HULL_BUDGET for v in trial.values())
            if over or (b - cur[0][0]) > cfg["chunkLength"] + 1e-6:
                chunks.append(cur)
                cur = []
        cur.append((a, b))
    if cur:
        chunks.append(cur)
    if len(chunks) > 999:
        raise MifOpError("the road would make %d chunks; at most 999. Raise chunkLength or shorten the road. NOTHING "
                         "was created." % len(chunks))

    # ------------------------------------------------------------------ station rows
    row_cuts = set(cuts)
    if cfg["centre"] == "dashed":
        k = 0
        while k * (_DASH_ON + _DASH_OFF) < L:
            row_cuts.add(k * (_DASH_ON + _DASH_OFF))
            row_cuts.add(k * (_DASH_ON + _DASH_OFF) + _DASH_ON)
            k += 1
    for j, shp in zip(cfg["jumps"], jshapes):
        # rows wherever the jump profile bends, so the barriers riding it bend there too
        for xo, _yy in shp["rampRows"] + shp["landRows"]:
            row_cuts.add(j["at"] + xo)
    for s in al.kinks:
        row_cuts.add(s)
    for side in (1, -1):
        for a, b, kk, v in sect.ranges[side]:
            if kk == "shoulder":
                for x in (a - 0.5, a + 0.5, b - 0.5, b + 0.5):
                    row_cuts.add(x)
    rows = set()
    last = None
    i = 0
    while i * _DENSE <= L + 1e-9:
        s = min(L, i * _DENSE)
        x, y, th, k = al.plan(s)
        bnk, gr = al.bank(s), al.grade(s)
        if last is None or (s - last[0] >= _ROW_MAX - 1e-9 or abs(th - last[1]) >= math.radians(_ROW_DEG)
                            or abs(bnk - last[2]) >= _ROW_BANK or abs(gr - last[3]) >= _ROW_GRADE):
            rows.add(s)
            last = (s, th, bnk, gr)
        i += 1
    rows.add(L)
    # THE SEGMENT CUTS ARE ROWS EXACTLY (hulls end on them); any other cut or row within 5 cm of a kept cut is dropped,
    # so no two rows lie a millimeter apart - a dash end 0.9 mm from a hull's end once made a sliver row there
    kept = sorted(set(round(c, 6) for c in cuts))
    for c in sorted(c for c in row_cuts if 0.0 <= c <= L):
        i = bisect.bisect_left(kept, c)
        if all(abs(c - kept[j]) > 0.05 for j in (i - 1, i) if 0 <= j < len(kept)):
            kept.insert(i, round(c, 6))
    rows = sorted(set(kept) | {round(r, 6) for r in rows
                               if all(abs(r - kept[j]) > 0.05 for j in (bisect.bisect_left(kept, r) - 1,
                                                                         bisect.bisect_left(kept, r))
                                      if 0 <= j < len(kept))})

    # ------------------------------------------------------------------ materials
    used = {"asphalt", "marking"} | {cfg["shoulder"]}
    for side in (1, -1):
        used.add(sect.base[side]["shoulder"])
        for a, b, k, v in sect.ranges[side]:
            if k == "shoulder":
                used.add(v)
            if k == "barrier" and v != "none":
                used.add(v)
        if sect.base[side]["barrier"] != "none":
            used.add(sect.base[side]["barrier"])
    if curb_list:
        used.add("curb")
    if tyre_zones or any(k == "tyres" and v for side in (1, -1) for a, b, k, v in sect.ranges[side]):
        used.add("tyre")
    if fence_zones or cfg["fence"] is True or any(k == "fence" and v for side in (1, -1) for a, b, k, v in
                                                  sect.ranges[side]):
        used.add("fence")
    if jshapes:
        used.update(("wall", "curb", "marking"))
    used = [r for r in _ROLES if r in used]
    mats, made_mats = _resolve_materials(cfg["materials"], used)

    # ------------------------------------------------------------------ build
    coll = bpy.data.collections.new(name)
    parent_coll.children.link(coll)
    made = []
    try:
        root = bpy.data.objects.new(name, None)
        root.empty_display_type = "PLAIN_AXES"
        coll.objects.link(root)
        made.append(root)
        parts_out = []
        tile = cfg["uvTile"]
        for ci, segs_c in enumerate(chunks):
            a, b = segs_c[0][0], segs_c[-1][1]
            crow = [r for r in rows if a - 1e-6 <= r <= b + 1e-6]
            base = "%s_C%02d" % (name, ci) if len(chunks) <= 100 else "%s_C%03d" % (name, ci)
            # EVERY PART IS A SIBLING NAME - <chunk>_Road, <chunk>_CurbL, <chunk>_BarrierR - never the chunk's own name:
            # a hull attaches by UCX_<mesh>_NN, so a surface named <chunk> would also claim UCX_<chunk>_CurbL_00 (found
            # by uefn_check counting 22-25 hulls on each surface of the first proof circuit)
            sname = base + "_Road"
            # surface
            levels = [crow] + ([_thin(crow, 2), _thin(crow, 4)] if cfg["lods"] else [])
            objs = []
            for li, lr in enumerate(levels):
                buf = _surface(road, lr, tile, drop_marks=(li == 2))
                me = buf.to_mesh(sname if li == 0 else "%s_LOD%d" % (sname, li), _slot_roles(buf), mats, "RoadUV",
                                 attrs=(li == 0))
                o = bpy.data.objects.new(me.name, me)
                coll.objects.link(o)
                made.append(o)
                objs.append(o)
            _tag(objs[0], name, ci, "surface")
            hulls = []
            if cfg["collision"]:
                for k, (sa, sb) in enumerate(segs_c):
                    pts = _surface_hull_points(road, [r for r in rows if sa - 1e-6 <= r <= sb + 1e-6])
                    hulls.append(_hull_object("UCX_%s_%02d" % (sname, k), pts, coll, made))
                for h in hulls:
                    _tag(h, name, ci, "hull", sname)
            if cfg["lods"]:
                _quiet_lods(objs[1:])
                made.append(_make_lod_group(objs[0], objs[1:], coll))
            parts_out.append({"object": sname, "role": "surface", "s0": round(a, 3), "s1": round(b, 3),
                              "lods": [_tris(o.data) for o in objs], "hulls": len(hulls)})
            cplan = plan(segs_c)
            # curbs, a mesh per side
            for side, tag in ((1, "CurbL"), (-1, "CurbR")):
                cs = curbs_in(a, b, side)
                if not cs:
                    continue
                cname = "%s_%s" % (base, tag)
                objs = []
                for li in range(3 if cfg["lods"] else 1):
                    buf = _curbs(road, cs, a, b, li, tile)
                    me = buf.to_mesh(cname if li == 0 else "%s_LOD%d" % (cname, li), ["curb"], mats, colors=True)
                    o = bpy.data.objects.new(me.name, me)
                    coll.objects.link(o)
                    made.append(o)
                    objs.append(o)
                _tag(objs[0], name, ci, "curbs")
                hulls = []
                if cfg["collision"]:
                    for c, lo, hi in cplan[("curb", side)]:
                        hulls.append(_hull_object("UCX_%s_%02d" % (cname, len(hulls)),
                                                  _curb_hull_points(road, c, lo, hi), coll, made))
                    for h in hulls:
                        _tag(h, name, ci, "hull", cname)
                if cfg["lods"]:
                    _quiet_lods(objs[1:])
                    made.append(_make_lod_group(objs[0], objs[1:], coll))
                parts_out.append({"object": cname, "role": "curbs", "s0": round(a, 3), "s1": round(b, 3),
                                  "lods": [_tris(o.data) for o in objs], "hulls": len(hulls)})
            # barriers, a mesh per side
            for side, tag in ((1, "BarrierL"), (-1, "BarrierR")):
                comps = _barrier_components(road, side, segs_c, rows)
                if not comps:
                    continue
                bname = "%s_%s" % (base, tag)
                objs = []
                for li in range(3 if cfg["lods"] else 1):
                    buf = _barriers(road, side, comps, li, tile)
                    if not buf.f:
                        break
                    # SMOOTH UP TO 50 DEGREES: an 8-sided tyre stack meets at 45, and marked sharp every corner split into
                    # its own engine vertex - 43,000-50,000 per barrier mesh on the first proof circuit, over UEFN's 30,000
                    me = buf.to_mesh(bname if li == 0 else "%s_LOD%d" % (bname, li), _slot_roles(buf), mats, sharp=50.0)
                    o = bpy.data.objects.new(me.name, me)
                    coll.objects.link(o)
                    made.append(o)
                    objs.append(o)
                if not objs:
                    continue
                role = "barrier" + ("L" if side > 0 else "R")
                _tag(objs[0], name, ci, role)
                hulls = []
                if cfg["collision"]:
                    for kind, g0, g1 in cplan[("barrier", side)]:
                        grows = [r for r in rows if g0 - 1e-6 <= r <= g1 + 1e-6]
                        hulls.append(_hull_object("UCX_%s_%02d" % (bname, len(hulls)),
                                                  _barrier_hulls(road, side, (kind, g0, g1, grows)), coll, made))
                    for h in hulls:
                        _tag(h, name, ci, "hull", bname)
                if cfg["lods"] and len(objs) > 1:
                    _quiet_lods(objs[1:])
                    made.append(_make_lod_group(objs[0], objs[1:], coll))
                parts_out.append({"object": bname, "role": role, "s0": round(a, 3), "s1": round(b, 3),
                                  "lods": [_tris(o.data) for o in objs], "hulls": len(hulls),
                                  "kinds": sorted({k for k, _a, _b in cplan[("barrier", side)]})})
        # jumps
        jumps_out = []
        for ji, (j, shp) in enumerate(zip(cfg["jumps"], jshapes)):
            for piece, prow in (("Ramp", shp["rampRows"]), ("Landing", shp["landRows"])):
                jname = "%s_Jump%d_%s" % (name, ji, piece)
                objs = []
                for li in range(3 if cfg["lods"] else 1):
                    buf = _jump_piece(road, j, prow, piece == "Ramp", li, tile)
                    me = buf.to_mesh(jname if li == 0 else "%s_LOD%d" % (jname, li), _slot_roles(buf), mats, colors=True,
                                     attrs=(li == 0))
                    o = bpy.data.objects.new(me.name, me)
                    coll.objects.link(o)
                    made.append(o)
                    objs.append(o)
                _tag(objs[0], name, ji, piece.lower())
                hulls = []
                if cfg["collision"]:
                    for k, pts in enumerate(_jump_hulls(road, j, prow, piece == "Ramp")):
                        hulls.append(_hull_object("UCX_%s_%02d" % (jname, k), pts, coll, made))
                    for h in hulls:
                        _tag(h, name, ji, "hull", jname)
                if cfg["lods"]:
                    _quiet_lods(objs[1:])
                    made.append(_make_lod_group(objs[0], objs[1:], coll))
                parts_out.append({"object": jname, "role": piece.lower(), "lods": [_tris(o.data) for o in objs],
                                  "hulls": len(hulls)})
            jumps_out.append({"index": ji, "at": j["at"], "takeoffDeg": j["takeoff"], "lipHeight": j["height"],
                              "speedKmh": j["speed"], "rampRadius": round(shp["rampRadius"], 2),
                              "lipStation": round(j["at"] + shp["lipX"], 2),
                              "knuckleStation": round(j["at"] + shp["knuckleX"], 2),
                              "footStation": round(j["at"] + shp["footX"], 2),
                              "knuckleHeight": round(shp["knuckleHeight"], 3),
                              "tableLength": round(shp["tableLength"], 2),
                              "landingSlopeMaxDeg": round(shp["landingDeg"], 2),
                              "width": j["width"],
                              "speedWindowKmh": [round(j["speed"] * f, 1) for f in shp["speedWindow"]],
                              "touchdownStation": (round(j["at"] + shp["touchdownX"], 2)
                                                   if shp["touchdownX"] is not None else None),
                              "runUp": j["runUp"], "runOut": j["runOut"]})
    except Exception:
        for o in reversed(made):
            try:
                remove_objects([o])
            except ReferenceError:
                pass
        try:
            bpy.data.collections.remove(coll)
        except ReferenceError:
            pass
        raise

    # ------------------------------------------------------------------ the design, stored for road_report
    design = {
        "version": 1, "name": name, "closed": al.closed, "length": L, "width": cfg["width"],
        "designSpeed": cfg["designSpeed"], "sideFriction": cfg["sideFriction"], "maxBank": cfg["maxBank"],
        "bankRate": cfg["bankRate"], "crown": cfg["crown"], "curbWidth": cfg["curbWidth"],
        "curbRadius": cfg["curbRadius"], "uvTile": tile, "smoothed": cfg["smooth"],
        "barrierReach": cfg["curbWidth"] + cfg["shoulderWidth"] + 2.0,
        "materials": {r: m.name for r, m in mats.items()},
        "corners": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()} for c in al.corners],
        "curbs": [{"side": c["side"], "s0": round(c["s0"], 3), "s1": round(c["s1"], 3)} for c in curb_list],
        "open": {str(side): [[round(a2, 3), round(b2, 3)] for a2, b2 in _open_ranges(sect, side, L)]
                 for side in (1, -1)},
        "jumps": jumps_out,
    }
    root["mif_road"] = json.dumps(design)
    out = {"ok": True, "road": name, "collection": coll.name, "length": round(L, 3), "closed": al.closed,
           "chunks": len(chunks), "segments": len(segs), "rows": len(rows),
           "corners": [{"point": c["pi"], "direction": c["direction"], "radius": round(c["radius"], 2),
                        "radiusAuto": c["radiusAuto"], "spiral": round(c["spiral"], 2), "arc": round(c["arc"], 2),
                        "deflectionDeg": round(c["deflectionDeg"], 2), "s0": round(c["s0"], 2),
                        "s1": round(c["s1"], 2), "speedKmh": c["speed"],
                        "bankDeg": round(max((abs(al.bank(c["s0"] + t * (c["s1"] - c["s0"]))) for t in
                                              (0.4, 0.5, 0.6)), default=0.0), 2)}
                       for c in al.corners],
           "verticalCurves": [{"station": round(c["s"], 2), "kind": c["kind"], "gradeChange": round(c["A"], 4),
                               "length": round(c["L"], 2), "radius": round(c["L"] / abs(c["A"]), 1),
                               "shortened": c["L"] < c["asked"] - 1e-6}
                              for c in al.vcurves if c],
           "jumps": jumps_out, "curbRuns": len(curb_list), "parts": parts_out,
           "materials": {r: m.name for r, m in mats.items()}, "materialsMade": made_mats,
           "pivot": "every piece sits at the world origin: place each imported mesh at 0, 0, 0",
           "uv": {"UVMap": "world-planar on the surface, %g m per repeat; along each part's own length on curbs and "
                           "barriers" % tile,
                  "RoadUV": "surface only: u 0-1 across the racing surface (left to right), v meters along / uvTile"},
           "notes": cfg["notes"] or None,
           "next": "road_report {\"road\": \"%s\"} measures it" % name}
    if path is not None:
        out["file"] = _export(coll, root, path, before_file)
    out["elapsedSeconds"] = round(time.perf_counter() - started, 2)
    return out


def _open_ranges(sect, side, L):
    """Stretches a side's barrier is 'none' on purpose - not gaps."""
    out = []
    if sect.base[side]["barrier"] == "none":
        out.append((0.0, L))
    for a, b, k, v in sect.ranges[side]:
        if k == "barrier" and v == "none":
            out.append((a, b))
    return _ranges_union(out)


def _thin(rows, step):
    out = rows[::step]
    if out[-1] != rows[-1]:
        out.append(rows[-1])
    return out


def _tag(obj, road, chunk, role, of=None):
    obj["mif_road_part"] = json.dumps({"road": road, "chunk": chunk, "role": role, "of": of})


def _quiet_lods(lods):
    """LOD1 and LOD2 lie on LOD0, so in Blender, which draws every LOD at once, they z-fight it - streaks and blotches
    across the asphalt in the first proof renders. An engine swaps them by screen size; here they are left out of the
    render and drawn as bounds. Neither stops a selection, so the FBX still carries them."""
    for o in lods:
        o.hide_render = True
        o.display_type = "BOUNDS"


def _slot_roles(buf):
    seen = []
    for r in buf.role:
        if r not in seen:
            seen.append(r)
    return [r for r in _ROLES if r in seen]


def _export(coll, root, path, before_file):
    sel = selection_snapshot()
    try:
        if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        objs = [o for o in coll.objects if o is not root]
        select_only(objs)
        args = dict(FBX_EXPORT_ARGS)
        args["object_types"] = {"MESH", "EMPTY"}
        bpy.ops.export_scene.fbx(filepath=path, use_selection=True, **args)
    finally:
        selection_restore(sel)
    size = os.path.getsize(path) if os.path.exists(path) else None
    if not size:
        raise MifOpError("the FBX exporter returned and %s is missing or empty. The road WAS built in the scene." % path)
    return dict({"path": path, "bytes": size, "objects": len(objs)}, **replaced_fields(before_file))


# ============================================================================================== part builders
def _pt(x, y, th, n, z, m=1.0):
    return (x - math.sin(th) * n * m, y + math.cos(th) * n * m, z)


def _surface(road, rows, tile, drop_marks=False):
    buf = _Buf()
    hw, w = road.hw, road.cfg["width"]
    grid = []
    roles_rows = []
    for ri, s in enumerate(rows):
        x, y, th, k, z, _b = road.frame(s)
        nxt = rows[ri + 1] if ri + 1 < len(rows) else s
        cols, roles = road.columns(s, _dash_on(s, nxt))
        if drop_marks:
            keep = [i for i, c in enumerate(cols) if abs(c[0]) not in (2, 6)]
            roles = [_role_between(cols, roles, keep[t], keep[t + 1]) for t in range(len(keep) - 1)]
            cols = [cols[i] for i in keep]
        ids = []
        for code, n, h in cols:
            co = _pt(x, y, th, n, z + h)
            ids.append(buf.vert(co, s, code))
        grid.append((ids, cols, s))
        roles_rows.append(roles)
    for ri in range(len(grid) - 1):
        a, ca, sa = grid[ri]
        b, cb, sb = grid[ri + 1]
        if len(a) != len(b):
            raise MifOpError("rows %.2f and %.2f have different column counts - a defect in build_road." % (sa, sb))
        for c in range(len(a) - 1):
            role = roles_rows[ri][c]
            ids = (a[c], a[c + 1], b[c + 1], b[c])
            uv0 = [(buf.v[i][0] / tile, buf.v[i][1] / tile) for i in ids]
            uv1 = [((hw - ca[c][1]) / w, sa / tile), ((hw - ca[c + 1][1]) / w, sa / tile),
                   ((hw - cb[c + 1][1]) / w, sb / tile), ((hw - cb[c][1]) / w, sb / tile)]
            buf.face(ids, role, uv0, uv1)
    return buf


def _role_between(cols, roles, i, j):
    # the role over the widest face between two kept columns
    best, width = roles[i], -1.0
    for t in range(i, j):
        wdt = abs(cols[t][1] - cols[t + 1][1])
        if wdt > width:
            best, width = roles[t], wdt
    return best


def _surface_hull_points(road, rows):
    pts = []
    for i, s in enumerate(rows):
        x, y, th, k, z, _b = road.frame(s)
        cols, _r = road.columns(s, False)
        for code, n, h in cols:
            if abs(code) in (2, 6):
                continue
            pts.append(_pt(x, y, th, n, z + h))
            if i in (0, len(rows) - 1) and abs(code) == 5:
                pts.append(_pt(x, y, th, n, z + h - 0.6))
    return pts


def _hull_dev(road, s0, s1):
    """How far the convex hull of the surface rows in [s0, s1] (sampled every 50 cm) stands over the racing surface:
    the largest gap at 7 stations by 3 offsets (the crown and 50 cm inside each edge), by ray cast."""
    n = max(2, int(math.ceil((s1 - s0) / 0.5)))
    bm = _build_hull(_surface_hull_points(road, [s0 + (s1 - s0) * i / n for i in range(n + 1)]), 0.001)
    try:
        tree = BVHTree.FromBMesh(bm)
    finally:
        bm.free()
    worst = 0.0
    down = Vector((0.0, 0.0, -1.0))
    for q in range(1, 8):
        s = s0 + (s1 - s0) * q / 8.0
        x, y, th, _k, z, bnk = road.frame(s)
        sl, sr = _side_slopes(bnk, road.cfg["crown"])
        for off in (0.0, road.hw - 0.5, -(road.hw - 0.5)):
            zt = z + (sl if off > 0 else sr) * abs(off)
            hit = tree.ray_cast(Vector(_pt(x, y, th, off, zt + 5.0)), down, 20.0)[0]
            if hit is None:
                return float("inf")
            worst = max(worst, abs(hit.z - zt))
    return worst


def _curb_profile(cw, height):
    """(offset from the lip outward, height over the strip) - a raised curb with a lip and a rounded outer edge."""
    return [(0.0, 0.004), (0.0, 0.3 * height), (0.25, height), (cw - 0.2, height), (cw - 0.05, 0.4 * height),
            (cw, 0.004)]


def _curb_rows(c, a, b, step):
    lo, hi = max(a, c["s0"]), min(b, c["s1"])
    rows = []
    k0 = int(math.ceil((lo - c["s0"]) / step - 1e-9))
    s = c["s0"] + k0 * step
    rows.append(lo)
    while s < hi - 1e-6:
        if s > lo + 1e-6:
            rows.append(s)
        s += step
    rows.append(hi)
    return rows


def _curbs(road, cs, a, b, lod, tile):
    buf = _Buf()
    cw, ht = road.cw, road.cfg["curbHeight"]
    prof = _curb_profile(cw, ht)
    if lod == 2:
        prof = [prof[0], prof[2], prof[3], prof[5]]
    step = _CURB_ROW * (1, 2, 4)[lod]
    red, white = (0.75, 0.04, 0.03), (0.92, 0.92, 0.9)
    for c in cs:
        side = c["side"]
        rows = _curb_rows(c, a, b, step)
        grid, along, prev = [], 0.0, None
        for s in rows:
            x, y, th, k, z, bnk = road.frame(s)
            sl, sr = _side_slopes(bnk, road.cfg["crown"])
            slope = sl if side > 0 else sr
            taper = max(0.15, min(1.0, (s - c["s0"]) / 0.6, (c["s1"] - s) / 0.6))
            rib = 0.0 if lod else _RIB_AMP * 0.5 * (1.0 + math.cos(2.0 * math.pi * (s - c["s0"]) / _RIB_PERIOD))
            ids = []
            for u, hh in prof:
                n = side * (road.hw + u)
                lift = 0.004 + max(0.0, taper) * (hh - 0.004 + (rib if hh >= ht * 0.99 else 0.0))
                ids.append(buf.vert(_pt(x, y, th, n, z + slope * abs(n) + lift), s))
            mid = buf.v[ids[len(ids) // 2]]
            if prev is not None:
                along += math.dist(mid[:2], prev[:2])
            prev = mid
            grid.append((ids, s, along))
        for ri in range(len(grid) - 1):
            ia, sa, va = grid[ri]
            ib, sb, vb = grid[ri + 1]
            stripe = int(math.floor(((sa + sb) / 2 - c["s0"]) / _STRIPE)) % 2
            col = red if stripe == 0 else white
            acc = 0.0
            for t in range(len(prof) - 1):
                du = math.hypot(prof[t + 1][0] - prof[t][0], prof[t + 1][1] - prof[t][1])
                ids = (ia[t], ia[t + 1], ib[t + 1], ib[t]) if side < 0 else (ia[t + 1], ia[t], ib[t], ib[t + 1])
                if side < 0:
                    uv = [(acc / tile, va / tile), ((acc + du) / tile, va / tile), ((acc + du) / tile, vb / tile),
                          (acc / tile, vb / tile)]
                else:
                    uv = [((acc + du) / tile, va / tile), (acc / tile, va / tile), (acc / tile, vb / tile),
                          ((acc + du) / tile, vb / tile)]
                buf.face(ids, "curb", uv, None, col)
                acc += du
    return buf


def _curb_hull_points(road, c, lo, hi):
    cw, ht = road.cw, road.cfg["curbHeight"]
    side = c["side"]
    pts = []
    rows = [lo] + [s for s in _curb_rows(c, lo, hi, 0.6) if lo < s < hi] + [hi]
    for s in rows:
        x, y, th, k, z, bnk = road.frame(s)
        sl, sr = _side_slopes(bnk, road.cfg["crown"])
        slope = sl if side > 0 else sr
        taper = max(0.0, min(1.0, (s - c["s0"]) / 0.6, (c["s1"] - s) / 0.6))
        for u, hh in ((0.0, 0.004), (0.25, ht), (cw - 0.2, ht), (cw, 0.004)):
            n = side * (road.hw + u)
            lift = 0.004 + taper * (hh - 0.004 + _RIB_AMP * (hh >= ht * 0.99))
            pts.append(_pt(x, y, th, n, z + slope * abs(n) + lift))
            pts.append(_pt(x, y, th, n, z + slope * abs(n) - 0.15))
    return pts


def _barrier_components(road, side, segs_c, rows):
    """[(kind, s0, s1, rows)] for one side of one chunk, a component per segment (segments are cut at section
    boundaries). rows None leaves the rows out - the chunk planner needs only the kinds and stations."""
    sect = road.sect
    comps = []
    for sa, sb in segs_c:
        mid = 0.5 * (sa + sb)
        kind = sect.get(side, mid, "barrier")
        r = None if rows is None else [x for x in rows if sa - 1e-6 <= x <= sb + 1e-6]
        if kind != "none":
            comps.append(("barrier:" + kind, sa, sb, r))
            if sect.get(side, mid, "tyres"):
                comps.append(("tyres", sa, sb, r))
        if sect.get(side, mid, "fence"):
            comps.append(("fence:" + kind, sa, sb, r))
    return comps


def _group(comps, maxlen):
    """Consecutive components of one kind merged into hull groups no longer than maxlen(start, kind): [(kind, s0, s1)].
    Neighbouring groups share their end stations, so their hulls share end faces."""
    out, last = [], {}
    for kind, s0, s1, _rows in comps:
        g = last.get(kind)
        if g is not None and abs(g[2] - s0) < 1e-6 and (s1 - g[1]) <= maxlen(g[1], kind) + 1e-6:
            g[2] = s1
        else:
            g = [kind, s0, s1]
            out.append(g)
            last[kind] = g
    return [tuple(g) for g in out]


def _barrier_base(road, side, s):
    """(x, y, th, n at the barrier's road face, z of the ground there, shoulder slope sign) for one side."""
    x, y, th, k, z, bnk = road.frame(s)
    sl, sr = _side_slopes(bnk, road.cfg["crown"])
    cols = road.side_cols(s, side, sl if side > 0 else sr)
    n = cols[-1][1]
    return x, y, th, side * n, z + cols[-1][2] + _jump_profile_y(road.jump_profiles, s)


_WALL = [(0.0, 0.0), (0.0, 0.075), (0.152, 0.33), (0.229, 0.81), (0.381, 0.81), (0.457, 0.33), (0.61, 0.075),
         (0.61, 0.0)]
_RAIL = [(0.04, 0.787), (0.0, 0.735), (0.0, 0.70), (0.06, 0.632), (0.0, 0.564), (0.0, 0.529), (0.04, 0.477)]


def _sweep(buf, road, side, rows, prof, role, tile, closed_ends=True, z_sink=0.05):
    """Extrude a profile [(depth outward, height)] along the barrier line of `side` over rows; faces outward. A profile
    point at ground height is sunk z_sink into the ground, and the UVs measure the profile as sunk."""
    prof = [(d, hh - (z_sink if hh <= 0.0 else 0.0)) for d, hh in prof]
    z_sink = 0.0
    grid, plan, run, prev = [], 0.0, 0.0, None
    for s in rows:
        x, y, th, n0, zg = _barrier_base(road, side, s)
        ids = []
        for d, hh in prof:
            n = n0 + side * d
            ids.append(buf.vert(_pt(x, y, th, n, zg + hh - (z_sink if hh <= 0.0 else 0.0)), s))
        p = buf.v[ids[0]]
        if prev is not None:
            plan += math.dist(p[:2], prev[:2])
            run += math.dist(p, prev)
        prev = p
        grid.append((ids, plan, run, zg))
    # UV, UNROLLED PER STRIP OF THE PROFILE. Where a barrier climbs a jump each strip is a parallelogram, and one
    # rectangle laid on all of them sheared the climbing faces 14% (road_report, the second jump build). A steep strip
    # (a wall's face) unrolls exactly as (plan length along, base height + distance round the profile); a flat one (a
    # wall's top) as (length along in 3D, distance round the profile). On level ground the two agree.
    for ri in range(len(grid) - 1):
        ia, pa, ra, za = grid[ri]
        ib, pb, rb, zb = grid[ri + 1]
        acc = 0.0
        for t in range(len(prof) - 1):
            dd, dh = prof[t + 1][0] - prof[t][0], prof[t + 1][1] - prof[t][1]
            du = math.hypot(dd, dh)
            if side > 0:
                ids = (ia[t], ib[t], ib[t + 1], ia[t + 1])
            else:
                ids = (ia[t], ia[t + 1], ib[t + 1], ib[t])
            if abs(dh) >= abs(dd):
                # the base height enters with the strip's own sign: on a strip running DOWN the profile (a back face)
                # height falls as the distance round grows, and adding it there mirrored the climb into a 28% shear
                sg = 1.0 if dh >= 0 else -1.0
                ua, ub = pa / tile, pb / tile
                a0, a1 = (sg * za + acc) / tile, (sg * za + acc + du) / tile
                b0, b1 = (sg * zb + acc) / tile, (sg * zb + acc + du) / tile
            else:
                ua, ub = ra / tile, rb / tile
                a0, a1, b0, b1 = acc / tile, (acc + du) / tile, acc / tile, (acc + du) / tile
            uv = [(ua, a0), (ub, b0), (ub, b1), (ua, a1)] if side > 0 else [(ua, a0), (ua, a1), (ub, b1), (ub, b0)]
            buf.face(ids, role, uv)
            acc += du
    if closed_ends and len(grid) > 1:
        for gi, ids in ((0, grid[0][0]), (-1, grid[-1][0])):
            ring = list(ids) if (gi == 0) == (side > 0) else list(reversed(ids))
            buf.face(ring, role, None, tile=tile)


def _box(buf, road, side, s, d0, d1, z0, z1, half_len, role, tile):
    x, y, th, n0, zg = _barrier_base(road, side, s)
    c, sn = math.cos(th), math.sin(th)
    vs = []
    for along in (-half_len, half_len):
        for d in (d0, d1):
            for hz in (z0, z1):
                px, py, pz = _pt(x, y, th, n0 + side * d, zg + hz)
                vs.append(buf.vert((px + along * c, py + along * sn, pz), s))
    # indices: (along, depth, height) bits
    def v(a, d, h):
        return vs[a * 4 + d * 2 + h]
    quads = [(v(0, 0, 0), v(0, 1, 0), v(0, 1, 1), v(0, 0, 1)), (v(1, 0, 0), v(1, 0, 1), v(1, 1, 1), v(1, 1, 0)),
             (v(0, 0, 0), v(0, 0, 1), v(1, 0, 1), v(1, 0, 0)), (v(0, 1, 0), v(1, 1, 0), v(1, 1, 1), v(0, 1, 1)),
             (v(0, 0, 1), v(0, 1, 1), v(1, 1, 1), v(1, 0, 1))]
    for q in quads:
        if side > 0:
            q = tuple(reversed(q))
        buf.face(q, role, None, tile=tile)


def _cylinder(buf, x, y, z0, r, rings, sides, role, tile, s):
    ids = []
    for hz, rr in rings:
        ring = []
        for k in range(sides):
            a = 2 * math.pi * k / sides
            ring.append(buf.vert((x + rr * math.cos(a), y + rr * math.sin(a), z0 + hz), s))
        ids.append(ring)
    for ri in range(len(ids) - 1):
        for k in range(sides):
            q = (ids[ri][k], ids[ri][(k + 1) % sides], ids[ri + 1][(k + 1) % sides], ids[ri + 1][k])
            buf.face(q, role, [(k * 2 * math.pi * r / sides / tile, rings[ri][0] / tile),
                               ((k + 1) * 2 * math.pi * r / sides / tile, rings[ri][0] / tile),
                               ((k + 1) * 2 * math.pi * r / sides / tile, rings[ri + 1][0] / tile),
                               (k * 2 * math.pi * r / sides / tile, rings[ri + 1][0] / tile)])
    top = ids[-1]
    buf.face(tuple(top), role, None, tile=tile)


def _runs(comps):
    """The chunk's components merged into continuous runs per kind: one sweep (and one pair of end caps) per run,
    while the hulls stay one per segment."""
    order, runs = [], {}
    for kind, s0, s1, rows in comps:
        lst = runs.setdefault(kind, [])
        if kind not in order:
            order.append(kind)
        if lst and abs(lst[-1][1] - s0) < 1e-6:
            lst[-1][1] = s1
            lst[-1][2] = lst[-1][2] + [r for r in rows if r > lst[-1][2][-1] + 1e-6]
        else:
            lst.append([s0, s1, list(rows)])
    return [(k, a, b, r) for k in order for a, b, r in runs[k]]


def _barriers(road, side, comps, lod, tile):
    buf = _Buf()
    stride = (1, 2, 4)[lod]
    for kind, s0, s1, rows in _runs(comps):
        r = _thin(rows, stride)
        if kind.startswith("barrier:"):
            typ = kind.split(":")[1]
            if typ == "wall":
                prof = _WALL if lod < 2 else [_WALL[0], _WALL[3], _WALL[4], _WALL[7]]
                _sweep(buf, road, side, r, prof, "wall", tile)
            else:
                # the W-beam's road face from its foot up, then its back from the top down: faces outward
                front = list(reversed(_RAIL)) if lod == 0 else [(0.0, 0.477), (0.0, 0.787)]
                back = [(d + 0.012, hh) for d, hh in _RAIL] if lod == 0 else [(0.083, 0.787), (0.083, 0.477)]
                ring = front + back + [front[0]]
                _sweep(buf, road, side, r, ring, "guardrail", tile, closed_ends=False, z_sink=0.0)
                pitch = _MGS_POST * (1 if lod < 2 else 2)
                k = math.ceil(s0 / pitch - 1e-9)
                while k * pitch <= s1 - 0.1 + 1e-9:
                    sp = k * pitch
                    if sp >= s0 + 0.1:
                        _box(buf, road, side, sp, 0.283, 0.433, -0.05, 0.75, 0.075, "guardrail", tile)
                        if lod == 0:
                            _box(buf, road, side, sp, 0.083, 0.283, 0.48, 0.75, 0.075, "guardrail", tile)
                    k += 1
        elif kind == "tyres":
            pitch = _TYRE_PITCH
            sides = (8, 6, 4)[lod]
            k = math.ceil(s0 / pitch - 1e-9)
            while k * pitch <= s1 - pitch * 0.5 + 1e-9:
                sp = k * pitch + pitch * 0.5
                x, y, th, n0, zg = _barrier_base(road, side, sp)
                for row in (1, 2):
                    d = -(_TYRE_R + 0.02) * (2 * row - 1)
                    px, py, _pz = _pt(x, y, th, n0 + side * d, zg)
                    if lod == 0:
                        # a groove ring at every tyre boundary and a full ring at each tyre's middle: 9 rings a stack
                        rings = []
                        for t in range(_TYRE_N):
                            z0 = t * _TYRE_H
                            rings += [(z0, _TYRE_R - 0.04), (z0 + _TYRE_H / 2.0, _TYRE_R)]
                        rings.append((_TYRE_N * _TYRE_H, _TYRE_R - 0.04))
                    else:
                        rings = [(0.0, _TYRE_R), (_TYRE_N * _TYRE_H, _TYRE_R)]
                    _cylinder(buf, px, py, zg - 0.03, _TYRE_R, rings, sides, "tyre", tile, sp)
                k += 1
        elif kind.startswith("fence:"):
            typ = kind.split(":")[1]
            d0 = _DEPTH.get(typ, 0.0) + 0.1
            prof = [(d0, 0.9), (d0, _FENCE_H), (d0 + 0.02, _FENCE_H), (d0 + 0.02, 0.9), (d0, 0.9)]
            _sweep(buf, road, side, r, prof, "fence", tile, closed_ends=False, z_sink=0.0)
            pitch = _FENCE_POST * (1, 2, 2)[lod]
            k = math.ceil(s0 / pitch - 1e-9)
            while k * pitch <= s1 - 0.1 + 1e-9:
                sp = k * pitch
                if sp >= s0 + 0.05:
                    _box(buf, road, side, sp, d0 + 0.02, d0 + 0.1, -0.05, _FENCE_H + 0.05, 0.04, "fence", tile)
                k += 1
    return buf


def _barrier_hulls(road, side, comp):
    kind, s0, s1, rows = comp
    if kind.startswith("barrier:"):
        typ = kind.split(":")[1]
        d0, d1, h = 0.0, _DEPTH[typ], _HEIGHT[typ]
    elif kind == "tyres":
        d0, d1, h = -(4 * (_TYRE_R + 0.02)), 0.0, _TYRE_N * _TYRE_H
    else:
        typ = kind.split(":")[1]
        d0 = _DEPTH.get(typ, 0.0) + 0.08
        d1, h = d0 + 0.06, _FENCE_H
    pts = []
    for s in rows:
        x, y, th, n0, zg = _barrier_base(road, side, s)
        for d in (d0, d1):
            for hz in (-0.1, h):
                pts.append(_pt(x, y, th, n0 + side * d, zg + hz))
    return pts


def _jump_cols(road, j, s):
    """A jump's columns across at station s: the road's own columns (racing surface, curb strip, shoulders), so the
    piece follows the crown or the bank and meets the barriers - clipped to +/- width/2 when a narrower jump is asked.
    Face roles: the racing surface and its lines as on the road, the curb strip as curb (the curbs ride up the jump),
    the shoulders as asphalt (a jump is paved edge to edge)."""
    cols, roles = road.columns(s, False)
    roles = [{"grass": "asphalt", "gravel": "asphalt", "runoff": "asphalt"}.get(r, r) for r in roles]
    roles = ["curb" if {abs(cols[i][0]), abs(cols[i + 1][0])} == {1, 3} else r for i, r in enumerate(roles)]
    w = j.get("width")
    if w is None or w / 2.0 >= abs(cols[0][1]) - 1e-6:
        return cols, roles
    hw = w / 2.0
    out, out_roles = [], []
    for i, (c, n, h) in enumerate(cols):
        if abs(n) < hw - 1e-6:
            out.append((c, n, h))
        if i + 1 < len(cols):
            n2, h2 = cols[i + 1][1], cols[i + 1][2]
            for edge in (hw, -hw):
                if (n - edge) * (n2 - edge) < 0:
                    t = (edge - n) / (n2 - n)
                    out.append((7 if edge > 0 else -7, edge, h + (h2 - h) * t))
    out.sort(key=lambda c: -c[1])
    for a, b in zip(out, out[1:]):
        mid = 0.5 * (a[1] + b[1])
        k = next(i for i in range(len(cols) - 1) if cols[i][1] >= mid >= cols[i + 1][1])
        out_roles.append(roles[k])
    return out, out_roles


def _jump_piece(road, j, prow, is_ramp, lod, tile):
    """A ramp-and-table or a landing: its top is the road's own cross-section raised by the profile (2 mm over it
    where the profile is 0, so the entry and the exit are flush), the curb strip in curb stripes, the sides down to the
    ground at the shoulders' outer edge, where the barriers ride up on it. No end faces: the ramp starts and the
    landing ends at road height, and the two meet on one shared row at the knuckle - so nothing faces the approach."""
    buf = _Buf()
    rows = prow if lod == 0 else _thin(prow, 2 if lod == 1 else 3)
    grid = []
    red, white = (0.75, 0.04, 0.03), (0.92, 0.92, 0.9)
    for xo, yy in rows:
        s = j["at"] + xo
        x, y, th, k, z, bnk = road.frame(s)
        cols, roles = _jump_cols(road, j, s)
        ids = [buf.vert(_pt(x, y, th, n, z + h + yy + _JUMP_EPS), xo, c) for c, n, h in cols]
        lo_l = buf.vert(_pt(x, y, th, cols[0][1], z + cols[0][2] - 0.05), xo, 21)
        lo_r = buf.vert(_pt(x, y, th, cols[-1][1], z + cols[-1][2] - 0.05), xo, -21)
        grid.append((ids, lo_l, lo_r, roles, xo))
    for ri in range(len(grid) - 1):
        a, al, ar, roles, xa = grid[ri]
        b, bl, br, _r, xb = grid[ri + 1]
        stripe = red if int(math.floor(0.5 * (xa + xb) / _STRIPE)) % 2 == 0 else white
        for c in range(len(a) - 1):
            role = roles[c]
            buf.face((a[c], a[c + 1], b[c + 1], b[c]), role, None, None, stripe if role == "curb" else None,
                     tile=tile)
        # sides: the left (+n) one faces +n, the right one -n
        buf.face((al, a[0], b[0], bl), "wall", None, tile=tile)
        buf.face((ar, br, b[-1], a[-1]), "wall", None, tile=tile)
    return buf


def _jump_hulls(road, j, prow, is_ramp):
    """Convex slices. The ramp's arc (concave) in slices of about 2 m, its top a chord under 2 cm over the arc; the
    table in one; the landing's curve (it steepens as it falls, so it is convex) in up to 5; each step of the foot's
    blend (concave) on its own."""
    if is_ramp:
        h = max(y for _x, y in prow)
        arc = [p for p in prow if p[1] < h - 1e-9] + [next(p for p in prow if p[1] >= h - 1e-9)]
        table = [p for p in prow if p[1] >= h - 1e-9]
        groups, cur = [], [arc[0]]
        for p in arc[1:]:
            cur.append(p)
            if p[0] - cur[0][0] >= 2.0:
                groups.append(cur)
                cur = [p]
        if len(cur) > 1:
            groups.append(cur)
        if len(table) > 1:
            groups.append(table)
    else:
        # the foot's blend is the last 8 steps, two to a hull
        curve = prow[:len(prow) - 8]
        k = max(1, int(math.ceil((len(curve) - 1) / 5.0)))
        groups = [curve[i:i + k + 1] for i in range(0, len(curve) - 1, k)]
        groups += [prow[i:i + 3] for i in range(len(prow) - 9, len(prow) - 1, 2)]
        groups = [g for g in groups if len(g) >= 2]
    out = []
    for g in groups:
        pts = []
        for xo, yy in g:
            s = j["at"] + xo
            x, y, th, k, z, bnk = road.frame(s)
            cols, _roles = _jump_cols(road, j, s)
            for c, n, h in cols:
                if abs(c) in (2, 6):
                    continue
                pts.append(_pt(x, y, th, n, z + h + yy + _JUMP_EPS))
            for c, n, h in (cols[0], cols[-1]):
                pts.append(_pt(x, y, th, n, z + h - 0.3))
        out.append(pts)
    return out


# ============================================================================================== road_report
def _thr(params):
    raw = params.get("thresholds") or {}
    if not isinstance(raw, dict):
        raise MifOpError("'thresholds' takes a dict of overrides by name (%s). NOTHING was read."
                         % ", ".join(_THRESHOLDS))
    bad = sorted(set(raw) - set(_THRESHOLDS))
    if bad:
        raise MifOpError("'thresholds' does not take %s; its names are %s. NOTHING was read."
                         % (", ".join(bad), ", ".join(_THRESHOLDS)))
    t = {k: v[0] for k, v in _THRESHOLDS.items()}
    for k, v in raw.items():
        t[k] = _num(v, "thresholds.%s" % k)
    return t


def _resample_polyline(pts, closed, step=0.5):
    """[(s, x, y, z)] every `step` m along a polyline, every vertex kept (where its kinks are)."""
    seq = list(pts) + ([pts[0]] if closed else [])
    out, s = [], 0.0
    for a, b in zip(seq, seq[1:]):
        ln = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(1, int(math.ceil(ln / step)))
        for i in range(n):
            t = i / n
            out.append((s + ln * t, a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t))
        s += ln
    if not closed:
        out.append((s, seq[-1][0], seq[-1][1], seq[-1][2]))
    return out, s


def _road_rows(root_name):
    """Station rows read back from the road's surface meshes: [(s, centre, left edge, right edge)] in world space."""
    by_s = {}
    surfaces = []
    for o in bpy.data.objects:
        tag = o.get("mif_road_part")
        if o.type != "MESH" or not tag:
            continue
        try:
            t = json.loads(tag)
        except ValueError:
            continue
        if t.get("road") == root_name and t.get("role") == "surface":
            surfaces.append(o)
    for o in surfaces:
        me = o.data
        sa, ca = me.attributes.get("mif_s"), me.attributes.get("mif_col")
        if sa is None or ca is None:
            continue
        mw = o.matrix_world
        for i, v in enumerate(me.vertices):
            code = ca.data[i].value
            if code not in (0, 1, -1):
                continue
            key = round(sa.data[i].value, 3)
            row = by_s.setdefault(key, {})
            row[code] = mw @ v.co
    rows = []
    for s in sorted(by_s):
        r = by_s[s]
        if 0 in r and 1 in r and -1 in r:
            rows.append((s, r[0], r[1], r[-1]))
    return rows, surfaces


def _circle_k(a, b, c):
    """Signed curvature (left positive) of the circle through three plan points."""
    abx, aby = b[0] - a[0], b[1] - a[1]
    acx, acy = c[0] - a[0], c[1] - a[1]
    bcx, bcy = c[0] - b[0], c[1] - b[1]
    cross = abx * acy - aby * acx
    d = math.hypot(abx, aby) * math.hypot(acx, acy) * math.hypot(bcx, bcy)
    return 2.0 * cross / d if d > 1e-12 else 0.0


def _measure(rows, closed, length, stencil=2.0):
    """Per-row measured plan curvature, its rate, grade and vertical curvature from centre points [(s, x, y, z)]:
    neighbours at least `stencil` m away either side, so float32 vertex precision (about 0.1 mm at 2 km from the
    origin) cannot read as curvature."""
    n = len(rows)
    S = [r[0] for r in rows]

    def at(i):
        if closed:
            return rows[i % n], (i // n) * length
        return rows[min(max(i, 0), n - 1)], 0.0

    def nb(i, d):
        j = i
        while True:
            j += d
            if not closed and (j < 0 or j >= n):
                return None
            r, off = at(j)
            if abs((r[0] + off) - S[i]) >= stencil or abs(j - i) > n:
                return j

    K, G, VK = [0.0] * n, [0.0] * n, [0.0] * n
    for i in range(n):
        a, b = nb(i, -1), nb(i, 1)
        if a is None or b is None:
            continue
        ra, oa = at(a)
        rb, ob = at(b)
        r = rows[i]
        K[i] = _circle_k(ra[1:3], r[1:3], rb[1:3])
        sa, sb = ra[0] + oa, rb[0] + ob
        G[i] = (rb[3] - ra[3]) / max(1e-6, sb - sa)
        g1 = (r[3] - ra[3]) / max(1e-6, S[i] - sa)
        g2 = (rb[3] - r[3]) / max(1e-6, sb - S[i])
        VK[i] = (g2 - g1) / max(1e-6, 0.5 * (sb - sa))
    KR = [0.0] * n
    for i in range(n):
        a, b = nb(i, -1), nb(i, 1)
        if a is None or b is None:
            continue
        ra, oa = at(a)
        rb, ob = at(b)
        KR[i] = abs(K[b % n if closed else b] - K[a % n if closed else a]) / max(1e-6, (rb[0] + ob) - (ra[0] + oa))
    return K, KR, G, VK


def _corners(S, K, closed):
    """Contiguous stretches where the radius is under 600 m, merged across gaps under 10 m."""
    thr = 1.0 / 600.0
    spans, cur = [], None
    for i, k in enumerate(K):
        on = abs(k) > thr
        if on and cur is None:
            cur = [i, i]
        elif on:
            if math.copysign(1, k) != math.copysign(1, K[cur[1]]):
                spans.append(cur)
                cur = [i, i]
            else:
                cur[1] = i
        elif cur is not None and S[i] - S[cur[1]] > 10.0:
            spans.append(cur)
            cur = None
    if cur is not None:
        spans.append(cur)
    # TWO CORNERS JOINED BY A SHORT STRAIGHT ARE TWO CORNERS: split a stretch at its lowest curvature when that falls
    # under half of the smaller peak on either side of it
    out, todo = [], list(spans)
    while todo:
        a, b = todo.pop(0)
        if b - a < 4:
            out.append((a, b))
            continue
        m = min(range(a + 1, b), key=lambda i: abs(K[i]))
        lp, rp = max(abs(K[i]) for i in range(a, m + 1)), max(abs(K[i]) for i in range(m, b + 1))
        if abs(K[m]) < 0.5 * min(lp, rp) and m - a >= 2 and b - m >= 2:
            todo[0:0] = [(a, m), (m, b)]
        else:
            out.append((a, b))
    return sorted(out)


def _status(rows_checked):
    return "warn" if any(not ok for _n, ok in rows_checked) else "pass"


def op_road_report(params):
    """Measure a road (or an alignment) and judge each corner and crest against documented thresholds.

    params:
      road (str)            a road's root name (build_road's 'name') - measured on its meshes
      points (list)         or an alignment alone: a polyline [x, y, z] ... (its corners are kinks)
      curve (str)           or a curve object, evaluated as drawn (a Bezier knot with broken handles is a kink)
      closed (bool)         for points (a curve's own cyclic flag otherwise)
      designSpeed (km/h)    for points or a curve, default 120 (a road's is read from it)
      thresholds (dict)     overrides by name; every threshold and its source is in the answer
    """
    reject_unknown(params, ROAD_REPORT_PARAMS, "road_report", _REPORT_NOTES)
    started = time.perf_counter()
    thr = _thr(params)
    road_name = take(params, "road", kind=str)
    curve_name = take(params, "curve", kind=str)
    pts_raw = params.get("points")
    if sum(x is not None for x in (road_name, curve_name, pts_raw)) != 1:
        raise MifOpError("road_report takes one of 'road' (a built road), 'points' or 'curve' (an alignment alone). "
                         "NOTHING was read.")
    design = None
    surfaces = []
    if road_name is not None:
        root = bpy.data.objects.get(road_name)
        if root is None or not root.get("mif_road"):
            raise MifOpError("no road named '%s' (build_road's root Empty, which carries mif_road). NOTHING was read."
                             % road_name)
        design = json.loads(root["mif_road"])
        rows, surfaces = _road_rows(road_name)
        if len(rows) < 5:
            raise MifOpError("the road '%s' has %d readable station rows on its surfaces (mif_s and mif_col); an empty "
                             "measurement is not a clean one. NOTHING was judged." % (road_name, len(rows)))
        closed, L = design["closed"], design["length"]
        centre = [(s, c.x, c.y, c.z) for s, c, _l, _r in rows]
        speed = design["designSpeed"]
        mode = "road"
    else:
        speed = _num(take_float(params, "designSpeed", default=120.0), "designSpeed", 10.0, 500.0)
        if curve_name is not None:
            obj = get_object(curve_name)
            if obj.type != "CURVE":
                raise MifOpError("'%s' is a %s, not a curve. NOTHING was read." % (curve_name, obj.type))
            centre_pts, closed, poly = _evaluate_curve(obj)
        else:
            closed = take_bool(params, "closed", default=False)
            centre_pts = [p for p in (_parse_points(pts_raw))]
            centre_pts = [p["co"] for p in centre_pts]
            poly = True
        if poly:
            centre, L = _resample_polyline(centre_pts, closed)
        else:
            centre, L = _chain(centre_pts, closed)
        rows = None
        mode = "curve" if curve_name else "points"
    if len(centre) < 5:
        raise MifOpError("the alignment has %d samples; nothing to judge. NOTHING was judged." % len(centre))

    S = [c[0] for c in centre]
    K, KR, G, VK = _measure(centre, closed, L)
    v_ms = speed / 3.6
    checks = []

    def judge(name, ok, value, limit, key, **extra):
        row = {"check": name, "status": "pass" if ok else "warn", "value": value, "limit": limit,
               "threshold": key, "source": _src(key)}
        row.update(extra)
        checks.append(row)
        return ok

    # ---------------------------------------------------------------- alignment as a whole
    kr_max = max(KR) if KR else 0.0
    kinks = []
    for i, kr in enumerate(KR):
        if kr > thr["maxCurvatureRate"] and (not kinks or S[i] - kinks[-1]["station"] > 5.0):
            kinks.append({"station": round(S[i], 2), "curvatureRate": round(kr, 5),
                          "radiusThere": round(1.0 / abs(K[i]), 2) if abs(K[i]) > 1e-9 else None})
    judge("curvatureContinuous", not kinks, round(kr_max, 5), thr["maxCurvatureRate"], "maxCurvatureRate",
          kinks=kinks[:50], kinkCount=len(kinks))
    kmax = max((abs(k) for k in K), default=0.0)
    judge("minRadius", kmax <= 1.0 / thr["minRadius"] + 1e-12, round(1.0 / kmax, 2) if kmax > 1e-9 else None,
          thr["minRadius"], "minRadius")
    gmax = max((abs(g) for g in G), default=0.0)
    judge("maxGrade", gmax <= thr["maxGrade"] + 1e-9, round(gmax, 4), thr["maxGrade"], "maxGrade")

    # ---------------------------------------------------------------- corners
    corners_out = []
    bank_rows = None
    if mode == "road":
        bank_rows = []
        for s, c, lft, rgt in rows:
            w = math.hypot(lft.x - rgt.x, lft.y - rgt.y)
            bank_rows.append((s, math.degrees(math.atan2(rgt.z - lft.z, w)), w, c.z - 0.5 * (lft.z + rgt.z)))
    for a, b in _corners(S, K, closed):
        seg = range(a, b + 1)
        ipk = max(seg, key=lambda i: abs(K[i]))
        kp = K[ipk]
        # THE APEX IS THE MIDDLE OF THE TIGHTEST STRETCH (curvature within 5% of its peak), not wherever float noise
        # put the single largest sample - on a circular arc every sample is the peak
        core = [i for i in seg if abs(K[i]) >= 0.95 * abs(kp)]
        ipk = min(core, key=lambda i: abs(S[i] - 0.5 * (S[core[0]] + S[core[-1]])))
        R = 1.0 / abs(kp)
        warn = []
        c_out = {"sStart": round(S[a], 2), "sApex": round(S[ipk], 2), "sEnd": round(S[b], 2),
                 "direction": "left" if kp > 0 else "right", "radiusMin": round(R, 2),
                 "curvatureRateMax": round(max(KR[i] for i in seg), 5)}
        # transitions: from the corner's start to 90% of its peak curvature, and back
        i90 = next(i for i in seg if abs(K[i]) >= 0.9 * abs(kp))
        j90 = next(i for i in reversed(seg) if abs(K[i]) >= 0.9 * abs(kp))
        c_out["transitionIn"] = round(S[i90] - S[a], 2)
        c_out["transitionOut"] = round(S[b] - S[j90], 2)
        if R < thr["minRadius"]:
            warn.append("radius %.1f m is under %.1f m" % (R, thr["minRadius"]))
        if c_out["curvatureRateMax"] > thr["maxCurvatureRate"]:
            warn.append("curvature changes %.4f per m, over %.4f - a kink" % (c_out["curvatureRateMax"],
                                                                             thr["maxCurvatureRate"]))
        if design is not None:
            cs = [c for c in design["corners"] if c["s0"] - 5 <= S[ipk] <= c["s1"] + 5]
            v = (cs[0]["speed"] if cs else design["designSpeed"])
            e = min(max(v * v / (127.0 * R) - design["sideFriction"], 0.0), math.tan(math.radians(design["maxBank"])))
            formula = math.degrees(math.atan(e)) * (1 if kp > 0 else -1)
            # measured at the apex: the mean over the stretch where curvature is within 5% of its peak
            core = [i for i in seg if abs(K[i]) >= 0.95 * abs(kp)]
            meas = sum(_bank_at(bank_rows, S[i]) for i in core) / len(core)
            # a corner whose bank the rate limit could not reach in its length is held to what the rate allows
            c_out.update({"speedKmh": v, "bankApexDeg": round(meas, 2), "bankFormulaDeg": round(formula, 2)})
            if abs(meas - formula) > thr["bankToleranceDeg"]:
                warn.append("bank %.2f deg at the apex, the formula gives %.2f" % (meas, formula))
            lo, hi = S[a] - 60.0, S[b] + 60.0
            br = _bank_rate(bank_rows, lo, hi)
            c_out["bankRateMaxDegPerM"] = round(br, 3)
            if br > thr["maxBankRateDegPerM"]:
                warn.append("bank changes %.3f deg per m, over %.3f" % (br, thr["maxBankRateDegPerM"]))
            if R <= design["curbRadius"]:
                cov = _curb_coverage(design, road_name, rows, S, K, a, b, ipk, kp)
                c_out["curbCoverageApex"], c_out["curbCoverageExit"] = cov
                if cov[0] is None or cov[0] < thr["curbCoverage"]:
                    warn.append("curb covers %s of the apex" % cov[0])
                if cov[1] is None or cov[1] < thr["curbCoverage"]:
                    warn.append("curb covers %s of the exit" % cov[1])
        c_out["status"] = "warn" if warn else "pass"
        c_out["warnings"] = warn or None
        corners_out.append(c_out)
    for c in corners_out:
        checks.append({"check": "corner at %.1f m" % c["sApex"], "status": c["status"], "warnings": c["warnings"]})

    # ---------------------------------------------------------------- crests and sags
    vcs = []
    i = 0
    n = len(S)
    while i < n:
        if abs(VK[i]) > 2e-4:
            j = i
            while j + 1 < n and abs(VK[j + 1]) > 2e-4 and math.copysign(1, VK[j + 1]) == math.copysign(1, VK[i]):
                j += 1
            vcs.append((i, j))
            i = j + 1
        else:
            i += 1
    jump_spans = []
    if design:
        for jd in design["jumps"]:
            jump_spans.append((jd["at"] - 1.0, jd["footStation"] + 1.0))
    crests_out = []
    for a, b in vcs:
        top = max(range(a, b + 1), key=lambda i: abs(VK[i]))
        kv = VK[top]
        kind = "crest" if kv < 0 else "sag"
        acc_g = v_ms * v_ms * abs(kv) / _G
        row = {"kind": kind, "station": round(S[top], 2), "z": round(centre[top][3], 3),
               "gradeIn": round(G[max(a - 1, 0)], 4), "gradeOut": round(G[min(b + 1, n - 1)], 4),
               "verticalRadius": round(1.0 / abs(kv), 1), "verticalAccelG": round(acc_g, 3)}
        warn = []
        if acc_g > thr["maxVerticalAccelG"]:
            warn.append("%.2f g at %.0f km/h, over %.2f g" % (acc_g, speed, thr["maxVerticalAccelG"]))
        if kind == "crest":
            sd = _sight_over(centre, top, closed, L)
            need = v_ms * thr["previewTime"]
            ssd = 0.278 * speed * 2.5 + 0.039 * speed * speed / 3.4
            row.update({"sightDistance": round(sd, 1), "sightRequired": round(need, 1),
                        "aashtoStoppingSight": round(ssd, 1),
                        "chevrons": "advised: the crest hides the road beyond for less than aashto's stopping sight "
                                    "distance" if sd < ssd else None})
            if sd < need:
                warn.append("sight distance %.1f m over the crest, under %.1f m (%.1f s at %.0f km/h)"
                            % (sd, need, thr["previewTime"], speed))
        row["status"] = "warn" if warn else "pass"
        row["warnings"] = warn or None
        crests_out.append(row)
        checks.append({"check": "%s at %.1f m" % (kind, S[top]), "status": row["status"], "warnings": warn or None})

    out = {"ok": True, "mode": mode, "length": round(L, 3), "closed": closed, "samples": len(S),
           "designSpeedKmh": speed, "corners": corners_out, "verticalCurves": crests_out}

    # ---------------------------------------------------------------- cross-section, barriers, collision, UVs, jumps
    if mode == "road":
        # bank_rows and S come from the same rows, index for index
        straight = [b for i, b in enumerate(bank_rows) if abs(K[i]) < 1.0 / thr["straightRadius"] and abs(b[1]) < 0.2]
        hw = design["width"] / 2.0
        crowns = [b[3] / hw for b in straight]
        widths = [b[2] for b in bank_rows]
        if crowns:
            judge("crownOnStraights", thr["crownMin"] - 1e-4 <= min(crowns) and max(crowns) <= thr["crownMax"] + 1e-4,
                  [round(min(crowns), 4), round(max(crowns), 4)], [thr["crownMin"], thr["crownMax"]], "crownMin",
                  rows=len(crowns))
        judge("width", thr["widthMin"] - 1e-3 <= min(widths) and max(widths) <= thr["widthMax"] + 1e-3,
              [round(min(widths), 3), round(max(widths), 3)], [thr["widthMin"], thr["widthMax"]], "widthMin")
        br_all = _bank_rate(bank_rows, -1e9, 1e9)
        judge("bankRate", br_all <= thr["maxBankRateDegPerM"], round(br_all, 3), thr["maxBankRateDegPerM"],
              "maxBankRateDegPerM")
        out["barriers"] = _barrier_check(road_name, design, rows, thr, judge)
        out["collision"] = _collision_check(road_name, design, rows, surfaces, thr, judge)
        out["uvStretch"] = _uv_check(road_name, design, thr, judge)
        out["jumps"] = _jump_check(road_name, design, rows, S, K, G, thr, judge, surfaces)
    warns = [c for c in checks if c["status"] == "warn"]
    out["checks"] = checks
    out["summary"] = {"clean": not warns, "checks": len(checks), "warnings": len(warns),
                      "warned": [c["check"] for c in warns][:40]}
    out["thresholds"] = {k: {"value": thr[k], "unit": v[1], "source": _SOURCES[v[2]], "rule": v[3]}
                         for k, v in _THRESHOLDS.items()}
    out["elapsedSeconds"] = round(time.perf_counter() - started, 2)
    return out


def _near(S, s, d):
    return [i for i, x in enumerate(S) if abs(x - s) <= d] or [min(range(len(S)), key=lambda i: abs(S[i] - s))]


def _bank_at(bank_rows, s):
    best = min(bank_rows, key=lambda b: abs(b[0] - s))
    return best[1]


def _bank_rate(bank_rows, lo, hi):
    """Largest bank change per meter between rows at least 1 m apart in [lo, hi]."""
    sel = [b for b in bank_rows if lo <= b[0] <= hi]
    worst = 0.0
    j = 0
    for i in range(len(sel)):
        while j < len(sel) and sel[j][0] - sel[i][0] < 1.0:
            j += 1
        if j < len(sel):
            worst = max(worst, abs(sel[j][1] - sel[i][1]) / (sel[j][0] - sel[i][0]))
    return worst


def _sight_over(centre, top, closed, L, look_back=200.0, cap=400.0):
    """The shortest sight distance over the crest at index `top`: for each station up to look_back m before it, how
    far ahead an object 0.60 m tall stays visible from an eye 1.08 m up, along the centre line (aashto ch. 3)."""
    n = len(centre)
    pts = centre

    def get(i):
        if closed:
            return pts[i % n][0] + (i // n) * L, pts[i % n][3]
        i = min(max(i, 0), n - 1)
        return pts[i][0], pts[i][3]

    worst = cap
    s_top = get(top)[0]
    i = top
    while True:
        si, zi = get(i)
        if s_top - si > look_back or (not closed and i <= 0):
            break
        eye = zi + _EYE_H
        # an object at j is seen when its sight line climbs at least as steeply as the steepest ground point before it
        steepest, best, j = -1e9, 0.0, i + 1
        while True:
            sj, zj = get(j)
            if sj - si > cap:
                best = cap
                break
            if not closed and j >= n:
                break
            d = max(1e-6, sj - si)
            if (zj + _OBJ_H - eye) / d < steepest:
                break
            best = d
            steepest = max(steepest, (zj - eye) / d)
            j += 1
        worst = min(worst, best)
        i -= 1
    return worst


def _evaluate_curve(obj):
    """World points along a curve as drawn: a Bezier spline evaluated at about 20 cm, a poly spline's points (a
    polyline), a NURBS spline at its own resolution."""
    from mathutils.geometry import interpolate_bezier  # noqa: PLC0415
    sp = obj.data.splines[0]
    mw = obj.matrix_world
    closed = bool(sp.use_cyclic_u)
    if sp.type == "BEZIER":
        bps = list(sp.bezier_points)
        pairs = list(zip(bps, bps[1:])) + ([(bps[-1], bps[0])] if closed else [])
        pts = []
        for a, b in pairs:
            ln = (b.co - a.co).length + (a.handle_right - a.co).length + (b.handle_left - b.co).length
            res = max(4, int(ln / 0.2))
            seg = interpolate_bezier(a.co, a.handle_right, b.handle_left, b.co, res + 1)
            pts.extend([mw @ p for p in seg[:-1]])
        if not closed:
            pts.append(mw @ bps[-1].co)
        return [(p.x, p.y, p.z) for p in pts], closed, False
    if sp.type == "POLY":
        return [tuple(mw @ Vector(p.co[:3])) for p in sp.points], closed, True
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    me = ev.to_mesh()
    try:
        pts = [tuple(mw @ v.co) for v in me.vertices]
    finally:
        ev.to_mesh_clear()
    return pts, closed, False


def _chain(pts, closed):
    out, s = [], 0.0
    for i, p in enumerate(pts):
        if i:
            s += math.hypot(p[0] - pts[i - 1][0], p[1] - pts[i - 1][1])
        out.append((s, p[0], p[1], p[2]))
    if closed:
        s += math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1])
    return out, s


def _parts(road_name, roles):
    out = []
    for o in bpy.data.objects:
        tag = o.get("mif_road_part")
        if o.type != "MESH" or not tag:
            continue
        try:
            t = json.loads(tag)
        except ValueError:
            continue
        if t.get("road") == road_name and t.get("role") in roles:
            out.append((o, t))
    return out


def _bvh(objs):
    verts, polys = [], []
    for o in objs:
        mw = o.matrix_world
        base = len(verts)
        verts.extend(mw @ v.co for v in o.data.vertices)
        polys.extend(tuple(base + i for i in p.vertices) for p in o.data.polygons)
    if not polys:
        return None
    return BVHTree.FromPolygons(verts, polys, all_triangles=False)


def _hulls_of(road_name, part_names):
    return [o for o, t in _parts(road_name, ("hull",)) if t.get("of") in part_names]


def _row_frame(rows, s):
    """Interpolated (centre, unit lateral to the left, left edge, right edge) at station s from measured rows."""
    lo = 0
    hi = len(rows) - 1
    if s <= rows[0][0]:
        i = 0
    elif s >= rows[-1][0]:
        i = len(rows) - 2
    else:
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if rows[mid][0] <= s:
                lo = mid
            else:
                hi = mid
        i = lo
    a, b = rows[i], rows[i + 1]
    t = 0.0 if b[0] == a[0] else min(1.0, max(0.0, (s - a[0]) / (b[0] - a[0])))
    c = a[1].lerp(b[1], t)
    lft = a[2].lerp(b[2], t)
    rgt = a[3].lerp(b[3], t)
    lat = Vector((lft.x - rgt.x, lft.y - rgt.y, 0.0))
    lat.normalize()
    return c, lat, lft, rgt


def _probe_stations(rows, step=0.5):
    """Every row +/- 1 mm (where hulls meet) and every `step` m between."""
    out = []
    for i, r in enumerate(rows):
        out.extend((r[0] - 0.001, r[0] + 0.001))
        if i + 1 < len(rows):
            a, b = r[0], rows[i + 1][0]
            k = int((b - a) / step)
            out.extend(a + (b - a) * (q + 1) / (k + 1) for q in range(k))
    lo, hi = rows[0][0], rows[-1][0]
    return [s for s in out if lo <= s <= hi]


def _curb_coverage(design, road_name, rows, S, K, a, b, ipk, kp):
    curbs = [o for o, t in _parts(road_name, ("curbs",))]
    tree = _bvh(curbs)
    if tree is None:
        return (0.0, 0.0)
    cw = design["curbWidth"]
    hw = design["width"] / 2.0
    inside = 1.0 if kp > 0 else -1.0
    apex = [S[i] for i in range(a, b + 1) if abs(K[i]) >= 0.6 * abs(kp)]
    exit_ = [S[i] for i in range(ipk, b + 1) if abs(K[i]) >= 0.3 * abs(kp)]

    def cover(stations, side):
        if not stations:
            return None
        lo, hi = min(stations), max(stations)
        probes = [lo + (hi - lo) * q / 40.0 for q in range(41)]
        hit = 0
        for s in probes:
            c, lat, _l, _r = _row_frame(rows, s)
            p = c + lat * (side * (hw + cw * 0.5))
            loc, _n, _i, _d = tree.ray_cast(Vector((p.x, p.y, p.z + 3.0)), Vector((0, 0, -1)), 6.0)
            hit += loc is not None
        return round(hit / len(probes), 3)

    return cover(apex, inside), cover(exit_, -inside)


def _barrier_check(road_name, design, rows, thr, judge):
    out = {}
    for side, tag in ((1, "barrierL"), (-1, "barrierR")):
        parts = [o for o, t in _parts(road_name, (tag,))]
        hulls = _hulls_of(road_name, {o.name for o in parts})
        open_r = design["open"].get(str(side), [])
        tree = _bvh(hulls)
        gaps, cur, misses, probes = [], None, 0, 0
        for s in _probe_stations(rows):
            if any(a - 0.01 <= s <= b + 0.01 for a, b in open_r):
                if cur:
                    gaps.append(cur)
                    cur = None
                continue
            c, lat, lft, rgt = _row_frame(rows, s)
            edge = lft if side > 0 else rgt
            # SEVERAL HEIGHTS: the ground under a barrier on the low side of a banked corner sits most of a meter
            # under the racing edge, on the high side about level with it
            hit = None
            if tree:
                # and up past the road along a jump, where the barriers ride up on the ramp, table and landing
                for dz in (-0.9, -0.5, -0.1, 0.3, 0.9, 1.6, 2.4, 3.3, 4.4):
                    hit = tree.ray_cast(Vector((edge.x, edge.y, edge.z + dz)), lat * side, design["barrierReach"])[0]
                    if hit is not None:
                        break
            probes += 1
            if hit is None:
                misses += 1
                cur = [s, s] if cur is None else [cur[0], s]
            elif cur:
                gaps.append(cur)
                cur = None
        if cur:
            gaps.append(cur)
        gap_rows = [{"from": round(a, 3), "to": round(b, 3), "length": round(b - a, 3)} for a, b in gaps]
        longest = max((g["length"] for g in gap_rows), default=0.0)
        # a miss at a single probe is a gap at least as long as the probe spacing around it
        ok = probes > 0 and not gap_rows
        judge("barrier%s" % ("Left" if side > 0 else "Right"), ok, round(longest, 3), thr["maxBarrierGap"],
              "maxBarrierGap", probes=probes, misses=misses, gaps=gap_rows[:20], hulls=len(hulls),
              openSections=open_r)
        out["left" if side > 0 else "right"] = {"hulls": len(hulls), "probes": probes, "gaps": gap_rows[:20],
                                                "openSections": open_r}
    return out


def _collision_check(road_name, design, rows, surfaces, thr, judge):
    hulls = _hulls_of(road_name, {o.name for o in surfaces})
    tree_h = _bvh(hulls)
    tree_v = _bvh(surfaces)
    hw = design["width"] / 2.0
    worst, misses, probes = 0.0, 0, 0
    worst_at = None
    for s in _probe_stations(rows):
        c, lat, lft, rgt = _row_frame(rows, s)
        for off in (0.0, hw - 0.5, -(hw - 0.5)):
            p = c + lat * off
            o = Vector((p.x, p.y, p.z + 5.0))
            vh = tree_v.ray_cast(o, Vector((0, 0, -1)), 12.0)[0]
            hh = tree_h.ray_cast(o, Vector((0, 0, -1)), 12.0)[0] if tree_h else None
            if vh is None:
                continue
            probes += 1
            if hh is None:
                misses += 1
                continue
            d = abs(hh.z - vh.z)
            if d > worst:
                worst, worst_at = d, s
    ok = probes > 0 and misses == 0 and worst <= thr["maxCollisionDeviation"]
    judge("surfaceCollision", ok, round(worst, 4), thr["maxCollisionDeviation"], "maxCollisionDeviation",
          probes=probes, misses=misses, worstAt=round(worst_at, 2) if worst_at is not None else None,
          hulls=len(hulls))
    return {"hulls": len(hulls), "probes": probes, "misses": misses, "maxDeviation": round(worst, 4)}


def _uv_stretch(obj, tile, roles=None):
    """(largest, 95th percentile by area) of max(|s1 * tile - 1|, |s2 * tile - 1|) over triangles, s1 and s2 the
    singular values of the map from the triangle to UVMap."""
    me = obj.data
    me.calc_loop_triangles()
    uv = me.uv_layers.get("UVMap")
    if uv is None:
        return None
    rows = []
    slots = [m.name if m else None for m in me.materials]
    for t in me.loop_triangles:
        if roles is not None and slots[t.material_index] not in roles:
            continue
        p = [me.vertices[i].co for i in t.vertices]
        q = [uv.data[li].uv for li in t.loops]
        e1, e2 = p[1] - p[0], p[2] - p[0]
        n1 = e1.length
        if n1 < 1e-9:
            continue
        ax = e1 / n1
        nrm = e1.cross(e2)
        area = nrm.length / 2.0
        if area < 1e-9:
            continue
        ay = nrm.normalized().cross(ax)
        x1, y1 = n1, 0.0
        x2, y2 = e2.dot(ax), e2.dot(ay)
        u1, v1 = q[1][0] - q[0][0], q[1][1] - q[0][1]
        u2, v2 = q[2][0] - q[0][0], q[2][1] - q[0][1]
        det = x1 * y2 - x2 * y1
        if abs(det) < 1e-12:
            continue
        # J maps (x, y) to (u, v): J = [[a, b], [c, d]]
        a = (u1 * y2 - u2 * y1) / det
        b = (u2 * x1 - u1 * x2) / det
        c = (v1 * y2 - v2 * y1) / det
        d = (v2 * x1 - v1 * x2) / det
        s1s = a * a + b * b + c * c + d * d
        dd = abs(a * d - b * c)
        root = math.sqrt(max(0.0, s1s * s1s / 4.0 - dd * dd))
        s1 = math.sqrt(max(0.0, s1s / 2.0 + root)) * tile
        s2 = math.sqrt(max(0.0, s1s / 2.0 - root)) * tile
        rows.append((max(abs(s1 - 1.0), abs(s2 - 1.0)), area))
    if not rows:
        return None
    rows.sort()
    total = sum(r[1] for r in rows)
    acc, p95 = 0.0, rows[-1][0]
    for v, ar in rows:
        acc += ar
        if acc >= 0.95 * total:
            p95 = v
            break
    return rows[-1][0], p95


def _uv_check(road_name, design, thr, judge):
    tile = design["uvTile"]
    out = {}
    asphalt = design["materials"].get("asphalt")
    surf = _parts(road_name, ("surface",))
    worst_s, p95_s = 0.0, 0.0
    for o, _t in surf:
        r = _uv_stretch(o, tile, {asphalt})
        if r:
            worst_s, p95_s = max(worst_s, r[0]), max(p95_s, r[1])
    judge("uvStretchAsphalt", bool(surf) and worst_s <= thr["maxUvStretch"], round(worst_s, 4), thr["maxUvStretch"],
          "maxUvStretch", p95=round(p95_s, 4), meshes=len(surf))
    out["asphalt"] = {"max": round(worst_s, 4), "p95": round(p95_s, 4)}
    for role in ("curbs", "barrierL", "barrierR"):
        parts = _parts(road_name, (role,))
        worst, p95 = 0.0, 0.0
        for o, _t in parts:
            names = {m.name for m in o.data.materials if m and m.name != design["materials"].get("tyre")}
            r = _uv_stretch(o, tile, names)
            if r:
                worst, p95 = max(worst, r[1]), max(p95, r[1])
        if parts:
            judge("uvStretch%s" % role[0].upper() + role[1:], worst <= thr["maxPartUvStretch"], round(worst, 4),
                  thr["maxPartUvStretch"], "maxPartUvStretch", meshes=len(parts),
                  measure="95th percentile by area (posts, caps and end faces carry their own small UVs)")
            out[role] = {"p95": round(worst, 4)}
    return out


def _piece_edge(obj, first):
    """The top vertices (mif_col under 10) of a jump piece's first or last row, in world space."""
    me = obj.data
    sa, ca = me.attributes.get("mif_s"), me.attributes.get("mif_col")
    xs = [sa.data[i].value for i in range(len(me.vertices))]
    tops = [i for i in range(len(me.vertices)) if abs(ca.data[i].value) < 10]
    if not tops:
        return []
    target = min(xs[i] for i in tops) if first else max(xs[i] for i in tops)
    mw = obj.matrix_world
    return [mw @ me.vertices[i].co for i in tops if abs(xs[i] - target) < 1e-4]


def _jump_check(road_name, design, rows, S, K, G, thr, judge, surfaces=()):
    out = []
    surf_tree = _bvh(list(surfaces))
    down = Vector((0.0, 0.0, -1.0))
    for jd in design["jumps"]:
        idx = jd["index"]
        ramps = [o for o, t in _parts(road_name, ("ramp",)) if t.get("chunk") == idx]
        lands = [o for o, t in _parts(road_name, ("landing",)) if t.get("chunk") == idx]
        warn = []
        row = {"index": idx, "at": jd["at"]}
        ramp = next((o for o in ramps if o.data.attributes.get("mif_col") is not None), None)
        land = next((o for o in lands if o.data.attributes.get("mif_col") is not None), None)
        if ramp is None or land is None:
            judge("jump %d" % idx, False, None, None, "runUp", warnings=["its ramp or landing mesh is missing"])
            continue
        top_r = _top_profile(ramp)
        top_l = _top_profile(land)
        # THE LIP: where the ramp first reaches its full height (a table may run on level after it); the takeoff angle
        # is that last rising face's, in world space
        # (found by slope, not by height: a table on a rising road keeps climbing and its far end is the highest)
        slopes = [(zb - za) / max(1e-9, xb - xa) for (xa, za), (xb, zb) in zip(top_r, top_r[1:])]
        smax = max(slopes) if slopes else 0.0
        il = max(i for i, sl in enumerate(slopes) if sl >= smax - math.tan(math.radians(0.5))) + 1 if slopes else 0
        (x1, z1), (x2, z2) = top_r[max(0, il - 1)], top_r[il]
        ang = math.atan2(z2 - z1, x2 - x1)
        v = jd["speedKmh"] / 3.6
        row.update({"takeoffDegMeasured": round(math.degrees(ang), 2)})
        td = _touchdown(x2, z2, ang, v, top_l)
        if td is None:
            warn.append("at %.0f km/h the car does not land on the landing" % jd["speedKmh"])
        else:
            xt, slope_deg, desc_deg = td
            row.update({"touchdownStation": round(jd["at"] + xt, 2), "landingSlopeDeg": round(slope_deg, 2),
                        "descentDeg": round(desc_deg, 2)})
            if abs(slope_deg - desc_deg) > thr["landingToleranceDeg"]:
                warn.append("touchdown at %.1f deg onto a %.1f deg slope" % (desc_deg, slope_deg))
        window = []
        for f in [x / 1000.0 for x in range(900, 1101)]:
            t2 = _touchdown(x2, z2, ang, v * f, top_l)
            if t2 is not None and abs(t2[1] - t2[2]) <= thr["landingToleranceDeg"] and t2[1] > 3.0:
                window.append(round(jd["speedKmh"] * f, 1))
        row["speedWindowKmh"] = [min(window), max(window)] if window else None
        # A LANDING ENDING AT ROAD LEVEL TAKES A NARROW BAND OF SPEEDS (_jump_shape): the band is reported, and what
        # is required is that the jump's own speed is inside it
        if not window or not min(window) <= jd["speedKmh"] <= max(window):
            warn.append("speeds landing within %.0f degrees: %s km/h - wanted %.0f km/h inside"
                        % (thr["landingToleranceDeg"], row["speedWindowKmh"], jd["speedKmh"]))

        # THE JOINTS WITH THE ROAD, MEASURED: how far the ramp's first row and the landing's last row stand off the
        # road surface under them, and the angle the ramp leaves the road at. The first build's ramp stood 1 cm proud at
        # its crown and 15 cm at its edges (a level top on a crowned road) - a lip a car hits, which nothing measured.
        steps = {}
        for label, obj, first in (("entry", ramp, True), ("exit", land, False)):
            worst = None
            for p in _piece_edge(obj, first):
                hit = surf_tree.ray_cast(Vector((p.x, p.y, p.z + 2.0)), down, 6.0)[0] if surf_tree else None
                if hit is None:
                    continue
                d = abs(p.z - hit.z)
                worst = d if worst is None else max(worst, d)
            steps[label] = worst
        tangent = None
        if len(top_r) > 1:
            gi = min(range(len(S)), key=lambda i: abs(S[i] - jd["at"]))
            (xa, za), (xb, zb) = top_r[0], top_r[1]
            tangent = abs(math.degrees(math.atan2(zb - za, xb - xa) - math.atan(G[gi])))
        row.update({"entryStepM": None if steps["entry"] is None else round(steps["entry"], 4),
                    "exitStepM": None if steps["exit"] is None else round(steps["exit"], 4),
                    "entryAngleDeg": None if tangent is None else round(tangent, 3)})
        for label in ("entry", "exit"):
            if steps[label] is None or steps[label] > thr["maxJumpStep"]:
                warn.append("the %s stands %s m off the road (at most %.3f)" % (label, steps[label],
                                                                                thr["maxJumpStep"]))
        if tangent is None or tangent > thr["maxJumpEntryAngleDeg"]:
            warn.append("the ramp leaves the road at %s deg, over %.1f - not tangent" % (tangent,
                                                                                         thr["maxJumpEntryAngleDeg"]))

        # WIDTH: the ramp's first row against the paved road (racing surface and both curb strips). The first build's
        # ramp was the racing surface's width and stood on it with the curb strips and shoulders bare beside it.
        edge = _piece_edge(ramp, True)
        c, lat, _l, _r = _row_frame(rows, jd["at"])
        offs = [(p - c).dot(lat) for p in edge]
        span = (max(offs) - min(offs)) if offs else 0.0
        paved = design["width"] + 2.0 * design["curbWidth"]
        row.update({"widthM": round(span, 3), "pavedWidthM": round(paved, 3),
                    "widthCoverage": round(span / paved, 4) if paved else None})
        if span / paved < thr["jumpWidthCoverage"] - 1e-3:
            warn.append("the ramp spans %.2f m of the %.2f m paved road" % (span, paved))

        # NOTHING FACES THE APPROACH STEEPER THAN IT SHOULD: a face whose normal points back at the oncoming car is a
        # wall to it unless it is the ramp's own rise (up to the takeoff angle) or a landing no steeper than its own
        # slope. The first build's landing faced the ramp with a 45-degree face up to a deck taller than the lip.
        tdir = Vector((lat.y, -lat.x, 0.0))
        land_steep = max((math.degrees(math.atan2(za - zb, xb - xa)) for (xa, za), (xb, zb) in zip(top_l, top_l[1:])
                          if xb > xa), default=0.0)
        bad, worst_f = 0.0, 0.0
        for objs, limit in ((ramps, jd["takeoffDeg"]), (lands, land_steep)):
            for o in objs:
                mw3 = o.matrix_world.to_3x3()
                for poly in o.data.polygons:
                    nrm = (mw3 @ poly.normal).normalized()
                    if nrm.dot(tdir) >= -1e-3 or poly.area < 1e-4:
                        continue
                    # ITS SLOPE ALONG THE ROAD, which is what a car meets: a shoulder's rollover riding the ramp
                    # leans outward as well, and its overall steepness is not a wall in the car's path
                    steep = math.degrees(math.atan2(-nrm.dot(tdir), max(1e-9, nrm.z)))
                    if steep > limit + thr["approachFaceToleranceDeg"]:
                        bad += poly.area
                        worst_f = max(worst_f, steep)
        row.update({"approachFaceAreaM2": round(bad, 3), "approachFaceSteepestDeg": round(worst_f, 2),
                    "landingSteepestDeg": round(land_steep, 2)})
        if bad > 0.0:
            warn.append("%.2f m2 of face toward the approach, up to %.1f deg - steeper than the ramp's takeoff or the "
                        "landing's %.1f deg slope" % (bad, worst_f, land_steep))

        # MEASURED ON THE ROAD: how far before the ramp it runs straight, and how far after the landing's foot it runs
        # straight and flat
        kst = 1.0 / thr["straightRadius"]
        before = [i for i, s in enumerate(S) if s <= jd["at"]]
        after = [i for i, s in enumerate(S) if s >= jd["footStation"]]
        run_up = jd["at"] - max((S[i] for i in before if abs(K[i]) > kst), default=S[0] if before else jd["at"])
        bad_out = [S[i] for i in after if abs(K[i]) > kst or abs(G[i]) > thr["flatGrade"]]
        run_out = (min(bad_out) if bad_out else (S[after[-1]] if after else jd["footStation"])) - jd["footStation"]
        row.update({"runUpStraight": round(run_up, 1), "runOutStraightFlat": round(run_out, 1)})
        need_up, need_out = max(thr["runUp"], jd.get("runUp") or 0.0), max(thr["runOut"], jd.get("runOut") or 0.0)
        if run_up < need_up:
            warn.append("the road runs straight for %.1f m before the ramp; wanted %.0f m" % (run_up, need_up))
        if run_out < need_out:
            warn.append("the road runs straight and under %.0f%% grade for %.1f m after the landing; wanted %.0f m"
                        % (100 * thr["flatGrade"], run_out, need_out))
        row["status"] = "warn" if warn else "pass"
        row["warnings"] = warn or None
        judge("jump %d" % idx, not warn, row.get("touchdownStation"), None, "landingToleranceDeg",
              warnings=warn or None, entryStepM=row["entryStepM"], widthCoverage=row["widthCoverage"],
              approachFaceAreaM2=row["approachFaceAreaM2"])
        out.append(row)
    return out


def _top_profile(obj):
    """[(x along the jump, z)] of the top centre line (mif_col 0), from the mesh."""
    me = obj.data
    sa, ca = me.attributes.get("mif_s"), me.attributes.get("mif_col")
    mw = obj.matrix_world
    pts = sorted((sa.data[i].value, (mw @ v.co).z) for i, v in enumerate(me.vertices) if ca.data[i].value == 0)
    return pts


def _touchdown(x0, z0, ang, v, top):
    """Where a point launched from (x0, z0) at angle `ang` and speed v first meets the profile `top` beyond x0:
    (x, the profile's slope there in degrees down, the trajectory's descent in degrees), or None."""
    va, vb = v * math.cos(ang), v * math.sin(ang)
    if va <= 0:
        return None
    pts = [p for p in top if p[0] > x0]
    for (xa, za), (xb, zb) in zip(pts, pts[1:]):
        def f(x):
            t = (x - x0) / va
            return z0 + vb * t - 0.5 * _G * t * t - (za + (zb - za) * (x - xa) / max(1e-9, xb - xa))
        if f(xa) >= 0 >= f(xb):
            lo, hi = xa, xb
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if f(mid) >= 0:
                    lo = mid
                else:
                    hi = mid
            t = (lo - x0) / va
            desc = math.degrees(math.atan2(-(vb - _G * t), va))
            slope = math.degrees(math.atan2(za - zb, xb - xa))
            return lo, slope, desc
    return None


OPS = {
    "build_road": op_build_road,
    "road_report": op_road_report,
}
