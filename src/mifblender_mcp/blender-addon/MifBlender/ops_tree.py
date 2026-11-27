"""Trees with real geometry leaves - build_tree.

WHAT WAS MISSING. The environment tooling could place, scatter and light trees and could not make one: MifBlender had no
tree op, Blender 4.2-5.2 ship no tree generator (Sapling was bundled up to 3.6 only), and Unreal's Procedural Vegetation
Editor writes its meshes only through an exporter behind a modal dialog. Decided 2026-09-27: our own builder, real
geometry leaves, made in Blender and handed to Unreal.

WHAT A GAME-READY TREE IS, and so what this builds: ONE object with TWO material slots, bark and leaves; the pivot at
the trunk's base with the root flare reaching below zero, so the tree sits into uneven ground instead of floating; a
base radius of 3-6% of the height; every branch its own tube, not welded into its parent; bark UVs that TILE along the
length with square texels; each leaf mapping the whole 0-1 UV square; leaves of 2-8 triangles, 5-20 cm, by the tens of
thousands - which Nanite carries in Unreal.

THE BRANCHING MODEL is written fresh from the published Weber-Penn paper ("Creation and Rendering of Realistic Trees",
SIGGRAPH 1995): recursive levels, children spaced along the parent and turned by a 137.5-degree phyllotaxis, a down
angle from the parent, first-level lengths trimmed by a crown envelope, curvature, and up or down attraction. No code
from any add-on.

VERTEX COLOR CARRIES WHAT A WIND AND VARIATION MATERIAL NEEDS, without textures: R = a random value per leaf (color
variation; 0.5 on bark), G = wind weight (0 at the trunk's base, rising along every branch, 1 at each leaf's tip),
B = a phase shared by a main branch and everything on it, so a whole limb sways together. A = 1. Each byte is the value
x 255, stored in sRGB space as a painted mask is, so the FBX export carries the numbers and Unreal reads them as written.

PURE PYTHON AND NUMPY. Geometry nodes move between Blender 3.6 and 5.2; mesh.from_pydata, foreach_set, a color
attribute on the CORNER domain and the shader nodes used here do not.
"""
import copy
import math
import random
import time

import bpy

try:
    import numpy as np
except ImportError:                     # pragma: no cover - every Blender 3.6-5.2 bundles numpy
    # THE OFFLINE SUITES IMPORT THIS MODULE with bpy stubbed, in a Python that may have no numpy (the CI gate installs
    # mcp and requests only), and test_blender_refusals B110 imports every ops module to build the op table - so a
    # missing numpy costs this op, not the import. ops_audio does the same.
    np = None

from .ops_common import (MifOpError, finite_float, finite_int, reject_unknown, remove_objects, take, take_bool,
                         take_float, take_int)
from .ops_create import _created, _link_new, _place, _place_values, _resolve_collection
from .ops_mesh import _UE_HULL_VERT_LIMIT, _build_hull, _hull_audit

# MODULE-LEVEL AND A PLAIN LITERAL: parity_check.py resolves an accepted-key expression statically.
BUILD_TREE_PARAMS = (
    "name", "preset", "seed", "height", "trunkRadius", "levels", "crownShape", "leafShape", "leafSize",
    "leafCount", "leafColor", "leafColor2", "barkColor", "collision", "location", "rotation", "collection",
    "leafClusters", "leavesPerCluster", "output", "textureFolder", "textureSize", "lods", "triangleBudget", "stump",
)
_STUMP_KEYS = {"height", "radius", "flare", "flareHeight", "roots", "cut", "segments", "sink"}
_ROOT_KEYS = {"count", "spread", "divePoint", "archHeight", "thickness", "seed"}
_CUT_KEYS = {"kind", "angle", "direction", "notchStep", "hinge", "hingeHeight", "breakHeight"}
_CUT_KINDS = ("SAW", "NOTCH", "BROKEN")
# build_tree keys a stump reads; the rest shape a whole tree and are refused with stump
_STUMP_TREE_KEYS = {"name", "seed", "trunkRadius", "barkColor", "location", "rotation", "collection", "stump"}
# the output gameReady keys: refused with a stump by name (output "geometry" alone is accepted, it is what a stump is)
_STUMP_GAME_KEYS = {"output", "textureFolder", "textureSize", "lods", "triangleBudget"}

_NOTES = {
    "species": "presets are tree FORMS (broadleafRound, broadleafTall, conifer, youngTree, shrub, pyramidal, vase, "
               "oval), not species - a species is a form plus its leaves, so pick the preset and set leafShape and "
               "leafSize (a London plane: broadleafRound with palmate leaves; a pin oak: pyramidal, lobed; a "
               "linden: pyramidal, cordate; a honey locust: vase, pinnate; a callery pear: oval, ovate).",
    "leafTexture": "leaves are real geometry: the leaf's outline IS the mesh, so there is no leaf texture and no "
                   "opacity mask. Color comes from leafColor, leafColor2 and the per-leaf value in vertex color R.",
    "cards": "output gameReady makes leaf cards: quads carrying a cluster of real leaves baked to textures (base "
             "color, normal, alpha) in textureFolder, inside the triangle budget mesh_quality gameReady holds the tree "
             "to, with LODs. The default output is real geometry leaves.",
    "clusters": "leafClusters true groups the leaves into twig-and-leaf units of about 80 triangles (leavesPerCluster "
                "sets how many leaves each carries) - fewer, richer parts than single leaves. A lower leafCount with a "
                "larger leafSize also makes a lighter tree.",
    "lod": "output gameReady makes LODs with build_lod_chain (lods, default [0.5, 0.25]); the default output makes "
           "none: in Unreal that tree is meant for Nanite.",
    "budget": "output gameReady fits the tree to triangleBudget, by default the top of the usual range for its size "
              "class - the number mesh_quality gameReady holds it to.",
    "wind": "wind is driven by the material from vertex color G (weight) and B (phase) - nothing to set here.",
    "scale": "the tree is built in meters at its real size; pass height (and trunkRadius) instead.",
    "branches": "the preset decides the branching; levels changes its depth.",
    "maxTriangles": "the budget is leafCount (up to 400000) times the leaf's triangles, plus the bark - both are in "
                    "the answer's triangles field.",
}

# THE PRESETS - tree forms, each a set of Weber-Penn-style values. Per level (index 0 = the trunk): children per parent,
# a child's length as a fraction of its parent, the down angle (degrees) and its variation, the curvature over the
# stem (degrees), segments, up attraction, and where along its parent a level's children start (level 1 starts at
# baseSize, the bare fraction of the trunk below the crown).
_PRESETS = {
    "broadleafRound": {
        "height": 12.0, "radiusRatio": 0.042, "levels": 3, "crown": "spherical", "baseSize": 0.32,
        "branches": (1, 34, 20, 14), "lengths": (1.0, 0.55, 0.40, 0.32), "down": (0, 64, 48, 52),
        "downV": (0, 12, 15, 18), "curve": (6, 36, 24, 16), "segments": (10, 8, 5, 2),
        "attractUp": (0.0, 0.2, 0.2, 0.1), "start": (0.0, 0.0, 0.3, 0.1), "ratioPower": 1.25, "baseSplits": 0,
        "curveV": (20, 60, 60, 40), "fork": (3, 0.42, 28),
        "collision": True, "leafShape": "ovate", "leafSize": 12.0, "leafCount": 100000,
        "leafColor": (0.080, 0.180, 0.030), "leafColor2": (0.160, 0.260, 0.040), "barkColor": (0.140, 0.100, 0.070)},
    "broadleafTall": {
        "height": 18.0, "radiusRatio": 0.036, "levels": 3, "crown": "tendFlame", "baseSize": 0.38,
        "branches": (1, 30, 18, 12), "lengths": (1.0, 0.40, 0.40, 0.30), "down": (0, 55, 45, 50),
        "downV": (0, 10, 14, 18), "curve": (5, 30, 22, 16), "segments": (12, 8, 5, 2),
        "attractUp": (0.0, 0.3, 0.2, 0.1), "start": (0.0, 0.0, 0.3, 0.1), "ratioPower": 1.25, "baseSplits": 0,
        "curveV": (15, 50, 50, 40), "fork": (2, 0.55, 18),
        "collision": True, "leafShape": "lanceolate", "leafSize": 11.0, "leafCount": 110000,
        "leafColor": (0.070, 0.170, 0.035), "leafColor2": (0.140, 0.240, 0.050), "barkColor": (0.160, 0.150, 0.130)},
    "conifer": {
        "height": 20.0, "radiusRatio": 0.030, "levels": 3, "crown": "conical", "baseSize": 0.18,
        "branches": (1, 130, 24, 10), "lengths": (1.0, 0.26, 0.42, 0.22), "down": (0, 92, 62, 55),
        "downV": (0, 7, 14, 15), "curve": (2, 18, 12, 10), "segments": (14, 6, 4, 2),
        "attractUp": (0.0, -0.45, 0.0, 0.0), "start": (0.0, 0.0, 0.05, 0.05), "ratioPower": 1.1, "baseSplits": 0,
        "curveV": (5, 20, 30, 30), "fork": (0, 0.0, 0),
        "collision": True, "leafShape": "needle", "leafSize": 13.0, "leafCount": 160000,
        "leafColor": (0.030, 0.090, 0.030), "leafColor2": (0.060, 0.130, 0.040), "barkColor": (0.120, 0.080, 0.060)},
    "youngTree": {
        "height": 5.0, "radiusRatio": 0.030, "levels": 3, "crown": "taperedCylindrical", "baseSize": 0.30,
        "branches": (1, 18, 12, 8), "lengths": (1.0, 0.45, 0.42, 0.34), "down": (0, 50, 45, 50),
        "downV": (0, 12, 15, 18), "curve": (8, 30, 22, 16), "segments": (8, 6, 4, 2),
        "attractUp": (0.0, 0.35, 0.25, 0.1), "start": (0.0, 0.0, 0.25, 0.1), "ratioPower": 1.3, "baseSplits": 0,
        "curveV": (15, 50, 50, 40), "fork": (0, 0.0, 0),
        "collision": True, "leafShape": "ovate", "leafSize": 8.0, "leafCount": 20000,
        "leafColor": (0.090, 0.200, 0.035), "leafColor2": (0.170, 0.280, 0.050), "barkColor": (0.150, 0.120, 0.090)},
    # A SHRUB HAS NO TRUNK HULL by default: it is walked through, not stood behind.
    "shrub": {
        "height": 2.2, "radiusRatio": 0.018, "levels": 3, "crown": "hemispherical", "baseSize": 0.05,
        "branches": (1, 120, 9, 6), "lengths": (1.0, 0.42, 0.5, 0.4), "down": (0, 45, 45, 50),
        "downV": (0, 15, 18, 20), "curve": (10, 30, 24, 18), "segments": (6, 5, 4, 2),
        "attractUp": (0.0, 0.3, 0.2, 0.1), "start": (0.0, 0.0, 0.15, 0.1), "ratioPower": 1.2, "baseSplits": 9,
        "curveV": (30, 60, 60, 40), "fork": (0, 0.0, 0),
        "collision": False, "leafShape": "ovate", "leafSize": 6.0, "leafCount": 25000,
        "leafColor": (0.070, 0.160, 0.035), "leafColor2": (0.130, 0.230, 0.045), "barkColor": (0.130, 0.100, 0.075)},
    # THE FORMS OF STREET AND PARK TREES that the five above did not cover (2026-10-05). PYRAMIDAL: one leader to the
    # top and many level limbs off it, the low ones reaching furthest and drooping at their ends, the high ones rising
    # - a pin oak's, a linden's, a sweetgum's.
    "pyramidal": {
        "height": 14.0, "radiusRatio": 0.034, "levels": 3, "crown": "tendFlame", "baseSize": 0.24,
        "branches": (1, 40, 14, 9), "lengths": (1.0, 0.40, 0.45, 0.36), "down": (0, 66, 45, 50),
        "downV": (0, 10, 15, 18), "curve": (3, 30, 22, 16), "segments": (14, 8, 5, 2),
        "attractUp": (0.0, -0.05, 0.15, 0.1), "start": (0.0, 0.0, 0.08, 0.05), "ratioPower": 1.25, "baseSplits": 0,
        "curveV": (8, 45, 50, 40), "fork": (0, 0.0, 0),
        "collision": True, "leafShape": "lobed", "leafSize": 14.0, "leafCount": 19000,
        "leafColor": (0.050, 0.140, 0.030), "leafColor2": (0.100, 0.200, 0.040), "barkColor": (0.170, 0.150, 0.130)},
    # VASE: a short trunk parting low into four rising limbs that spread out into a broad crown, open enough to see
    # sky through - a honey locust's, an elm's, a zelkova's.
    "vase": {
        "height": 13.0, "radiusRatio": 0.038, "levels": 3, "crown": "spherical", "baseSize": 0.30,
        "branches": (1, 36, 14, 8), "lengths": (1.0, 0.34, 0.45, 0.36), "down": (0, 48, 45, 50),
        "downV": (0, 12, 15, 18), "curve": (6, 30, 24, 16), "segments": (10, 8, 5, 2),
        "attractUp": (0.0, 0.10, 0.10, 0.05), "start": (0.0, 0.0, 0.1, 0.05), "ratioPower": 1.25, "baseSplits": 0,
        "curveV": (15, 55, 55, 40), "fork": (4, 0.28, 30),
        "collision": True, "leafShape": "pinnate", "leafSize": 20.0, "leafCount": 10500,
        "leafColor": (0.070, 0.170, 0.020), "leafColor2": (0.150, 0.260, 0.040), "barkColor": (0.100, 0.085, 0.075)},
    # OVAL: five steep leaders from one point and a dense, upright, egg-shaped crown - a callery pear's (the narrow
    # forks it is known for splitting at), a hornbeam's.
    "oval": {
        "height": 10.0, "radiusRatio": 0.040, "levels": 3, "crown": "spherical", "baseSize": 0.28,
        "branches": (1, 30, 16, 10), "lengths": (1.0, 0.40, 0.42, 0.34), "down": (0, 38, 40, 45),
        "downV": (0, 10, 15, 18), "curve": (4, 22, 22, 16), "segments": (10, 8, 5, 2),
        "attractUp": (0.0, 0.30, 0.25, 0.1), "start": (0.0, 0.0, 0.1, 0.05), "ratioPower": 1.25, "baseSplits": 0,
        "curveV": (12, 45, 50, 40), "fork": (5, 0.30, 16),
        "collision": True, "leafShape": "ovate", "leafSize": 7.5, "leafCount": 75000,
        "leafColor": (0.045, 0.140, 0.030), "leafColor2": (0.090, 0.200, 0.035), "barkColor": (0.140, 0.120, 0.100)},
}

_CROWNS = ("conical", "spherical", "hemispherical", "cylindrical", "taperedCylindrical", "flame", "inverseConical",
           "tendFlame")
_LEAF_SHAPES = ("ovate", "lanceolate", "round", "needle", "palmate", "lobed", "cordate", "pinnate")
_UP = np.array((0.0, 0.0, 1.0)) if np is not None else None
# Meters of bark per UV unit, round the girth and along the length alike (square texels): a trunk of 0.5 m radius
# wraps its bark texture about five times.
_BARK_TILE_M = 0.6
_MAX_LEAVES = 400000


def _crown_ratio(crown, r):
    """Weber-Penn's ShapeRatio: how long a first-level branch is at position r, where r runs from 1 at the crown's base
    to 0 at the top of the trunk."""
    r = min(1.0, max(0.0, r))
    if crown == "conical":
        return 0.2 + 0.8 * r
    if crown == "spherical":
        return 0.2 + 0.8 * math.sin(math.pi * r)
    if crown == "hemispherical":
        return 0.2 + 0.8 * math.sin(0.5 * math.pi * r)
    if crown == "cylindrical":
        return 1.0
    if crown == "taperedCylindrical":
        return 0.5 + 0.5 * r
    if crown == "flame":
        return r / 0.7 if r <= 0.7 else (1.0 - r) / 0.3
    if crown == "inverseConical":
        return 1.0 - 0.8 * r
    return 0.5 + 0.5 * r / 0.7 if r <= 0.7 else 0.5 + 0.5 * (1.0 - r) / 0.3      # tendFlame


def _rotate(v, axis, angle):
    """Rodrigues' rotation of v about a unit axis."""
    c, s = math.cos(angle), math.sin(angle)
    return v * c + np.cross(axis, v) * s + axis * float(np.dot(axis, v)) * (1.0 - c)


def _perpendicular(d):
    """A unit vector perpendicular to d, stable for any d."""
    other = _UP if abs(d[2]) < 0.9 else np.array((1.0, 0.0, 0.0))
    p = np.cross(d, other)
    return p / np.linalg.norm(p)


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else _UP.copy()


class _Stem:
    __slots__ = ("level", "points", "radii", "length", "phase", "weight0", "weight1")

    def __init__(self, level, points, radii, length, phase, weight0, weight1):
        self.level, self.points, self.radii, self.length = level, points, radii, length
        self.phase, self.weight0, self.weight1 = phase, weight0, weight1


def _grow(start, direction, length, r0, r1, level, cfg, rng, attract):
    """One stem as a polyline: it bends by the level's curvature in ONE plane, turned at random round the stem, so it
    draws a smooth arc (a random sign per segment drew a zig-zag); attraction pulls it toward up (or, negative, down)
    more strongly toward the tip, and the radius tapers linearly from r0 to r1."""
    segs = cfg["segments"][min(level, 3)]
    step = length / segs
    curve = math.radians(cfg["curve"][min(level, 3)]) / segs
    # WEBER-PENN'S CurveV: a random bend per segment about a random axis, on top of the stem's own arc. It is what
    # makes a limb read as grown rather than drawn - without it the first-level branches were straight spokes.
    curve_v = math.radians(cfg["curveV"][min(level, 3)]) / segs
    d = _unit(np.asarray(direction, dtype=float))
    bend_axis = _rotate(_perpendicular(d), d, rng.uniform(0.0, 2.0 * math.pi))
    pts, radii = [np.asarray(start, dtype=float)], [r0]
    for i in range(segs):
        if i:
            d = _unit(_rotate(d, bend_axis, curve * rng.uniform(0.5, 1.5)))
            if curve_v:
                wobble = _rotate(_perpendicular(d), d, rng.uniform(0.0, 2.0 * math.pi))
                d = _unit(_rotate(d, wobble, curve_v * rng.uniform(-1.0, 1.0)))
            if attract:
                d = _unit(d + _UP * attract * (i / segs) * 0.35)
        nxt = pts[-1] + d * step
        if level and nxt[2] < cfg["_floor"]:
            # A BRANCH DOES NOT GROW INTO THE GROUND: a shrub's low branches reached 15 cm below it before this
            d = _unit(np.array((d[0], d[1], max(d[2], 0.15))))
            nxt = pts[-1] + d * step
        pts.append(nxt)
        radii.append(r0 + (r1 - r0) * (i + 1) / segs)
    return pts, radii


def _point_at(stem, t):
    """Position, direction and radius at fraction t along a stem's polyline."""
    n = len(stem.points) - 1
    f = min(n - 1e-9, max(0.0, t * n))
    i = int(f)
    a, b = stem.points[i], stem.points[i + 1]
    u = f - i
    return a + (b - a) * u, _unit(b - a), stem.radii[i] + (stem.radii[i + 1] - stem.radii[i]) * u


def _build_stems(cfg, height, trunk_radius, rng):
    """Every stem of the tree, the trunk first - stems[0] is the one the root flare and the hull belong to. A trunk
    may FORK where the crown begins into 2-4 leaders, as a round broadleaf's does, instead of running on as one thin
    leader. Children are spaced along their parent, turned by the 137.5-degree phyllotaxis and tilted away from it by
    the down angle; a first-level child's length follows the crown envelope at the HEIGHT it starts from, so a leader
    leaning out still gets a crown, and a deeper one's shortens toward its parent's tip."""
    levels = cfg["levels"]
    splits = cfg["baseSplits"]
    root_depth = 0.03 * height
    crown_base = cfg["baseSize"] * height
    trunks = []
    if splits:
        for k in range(splits):
            # A SHRUB'S STEMS SPREAD from the ground at 15-45 degrees and vary in length; five near-upright ones of
            # nearly one height drew a broom
            ang = 2 * math.pi * k / splits + rng.uniform(-0.3, 0.3)
            # the first stem is the tallest and the one that reaches the height asked; the rest stay inside its crown
            tilt = math.radians(rng.uniform(4, 10) if k == 0 else rng.uniform(15, 45))
            d = np.array((math.cos(ang) * math.sin(tilt), math.sin(ang) * math.sin(tilt), math.cos(tilt)))
            length = height * (1.0 if k == 0 else rng.uniform(0.6, 0.92))
            pts, radii = _grow((0.0, 0.0, -root_depth * 0.3), d, length, trunk_radius * 0.6, trunk_radius * 0.08,
                               0, cfg, rng, 0.15)
            trunks.append(_Stem(0, pts, radii, length, rng.random(), 0.0, 0.15))
    elif cfg["fork"][0]:
        forks, fork_at, fork_angle = cfg["fork"]
        z_fork = fork_at * height
        base_len = z_fork + root_depth
        # THE SAME TAPER RATE AS AN UNFORKED TRUNK, and the leaders share its section: r^2 = sum of theirs
        r_fork = trunk_radius * (1.0 - 0.92 * base_len / (height + root_depth))
        pts, radii = _grow((0.0, 0.0, -root_depth), (0.0, 0.0, 1.0), base_len, trunk_radius, r_fork, 0, cfg, rng, 0.0)
        trunks.append(_Stem(0, pts, radii, base_len, rng.random(), 0.0, 0.05))
        top = pts[-1]
        az0 = rng.uniform(0.0, 2.0 * math.pi)
        for k in range(forks):
            az = az0 + 2.0 * math.pi * k / forks + rng.uniform(-0.35, 0.35)
            tilt = math.radians(fork_angle + rng.uniform(-8.0, 8.0))
            d = np.array((math.cos(az) * math.sin(tilt), math.sin(az) * math.sin(tilt), math.cos(tilt)))
            length = (height - z_fork) / math.cos(tilt) * rng.uniform(0.8, 1.0)
            r0 = r_fork / math.sqrt(forks) * 1.15
            pts, radii = _grow(top, d, length, r0, r0 * 0.12, 0, cfg, rng, 0.1)
            trunks.append(_Stem(0, pts, radii, length, rng.random(), 0.05, 0.15))
    else:
        pts, radii = _grow((0.0, 0.0, -root_depth), (0.0, 0.0, 1.0), height + root_depth, trunk_radius,
                           trunk_radius * 0.08, 0, cfg, rng, 0.0)
        trunks.append(_Stem(0, pts, radii, height + root_depth, rng.random(), 0.0, 0.1))
    stems = list(trunks)
    parents = trunks
    for level in range(1, levels + 1):
        kids = []
        if level == 1 and not splits:
            # THE FIRST LEVEL'S BRANCHES ARE SHARED OUT by each trunk stem's length inside the crown
            spans = []
            for p in parents:
                zs = np.array([q[2] for q in p.points])
                inside = np.clip((zs.max() - max(crown_base, zs.min())) / max(1e-6, zs.max() - zs.min()), 0.0, 1.0)
                spans.append(p.length * inside)
            total = max(1e-6, sum(spans))
        for pi, parent in enumerate(parents):
            n_base = cfg["branches"][level]
            if level == 1 and splits:
                t0, count = max(0.1, cfg["baseSize"]), max(3, n_base // splits)
            elif level == 1:
                zs = np.array([q[2] for q in parent.points])
                t0 = float(np.clip((crown_base - zs.min()) / max(1e-6, zs.max() - zs.min()), 0.0, 1.0))
                count = int(round(n_base * spans[pi] / total))
                if count == 0:
                    continue
            else:
                # A MAIN LIMB IS BARE NEAR THE TRUNK: foliage masses at the outside of a crown, where the light is
                t0 = cfg["start"][level]
                count = int(round(n_base * (0.25 + 0.75 * parent.length / max(1e-6, cfg["_maxlen"][level - 1]))))
            count = max(1 if level < levels else 2, count)
            angle = rng.uniform(0, 2 * math.pi)
            for c in range(count):
                t = t0 + (1.0 - t0) * (c + rng.uniform(0.15, 0.85)) / count
                pos, pdir, prad = _point_at(parent, t)
                angle += math.radians(137.5 + rng.uniform(-15, 15))
                down = math.radians(cfg["down"][level] + rng.uniform(-cfg["downV"][level], cfg["downV"][level]))
                if level == 1 and not splits:
                    r_env = (height - pos[2]) / max(1e-6, height - crown_base)
                    length = height * cfg["lengths"][1] * _crown_ratio(cfg["crown"], r_env)
                    down *= 0.75 + 0.5 * min(1.0, max(0.0, r_env))   # low branches reach out wider, high ones rise
                else:
                    length = parent.length * cfg["lengths"][level] * (1.0 - 0.55 * t)
                length *= rng.uniform(0.8, 1.15)
                if length < 0.03:
                    continue
                side = _rotate(_perpendicular(pdir), pdir, angle)
                d = _unit(_rotate(pdir, _unit(np.cross(pdir, side)), down))
                r0 = min(prad * 0.9, prad * (length / max(1e-6, parent.length)) ** cfg["ratioPower"] * 1.4)
                r0 = max(r0, 0.004)
                pts, radii = _grow(pos, d, length, r0, max(0.002, r0 * 0.25), level, cfg, rng,
                                   cfg["attractUp"][level])
                w0 = parent.weight0 + (parent.weight1 - parent.weight0) * t
                phase = rng.random() if level == 1 else parent.phase
                kids.append(_Stem(level, pts, radii, length, phase, w0, min(1.0, 0.3 + 0.8 * level / levels)))
        stems.extend(kids)
        parents = kids
        if not kids:
            break
    return stems, root_depth


def _sides(stem):
    """Sides of a stem's tube, by its base radius."""
    r = stem.radii[0]
    if stem.level == 0:
        return 24 if r > 0.15 else 16
    if r > 0.08:
        return 12
    if r > 0.03:
        return 8
    if r > 0.012:
        return 6
    return 4


def _with_base_rings(pts, radii, upto, n):
    """The same polyline with n more points between its start and `upto` meters along it, closer together toward the
    bottom - so a root flare has rings to shape rather than one long cone."""
    dist = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    upto = min(upto, 0.5 * dist[-1])
    extra = [e for e in upto * (np.arange(1, n + 1) / n) ** 1.6 if np.min(np.abs(dist - e)) > 0.01]
    at = np.sort(np.concatenate([dist, extra]))
    return (np.stack([np.interp(at, dist, pts[:, c]) for c in range(3)], axis=1), np.interp(at, dist, radii))


def _tube(pts, radii, sides, flare, root_depth, weight0, weight1, phase):
    """One stem as a closed-tip tube. Ring vertices are SHARED round the seam - the UVs are per corner, so the seam
    splits the texture and not the normals. U wraps the girth a whole number of times and V runs along the length at
    the same meters per unit (square texels, tiling). The trunk's lowest rings widen into buttress lobes when `flare`
    is set. Returns (verts, quads, quad_uv, quad_col, tris, tri_uv, tri_col), the per-corner arrays in face order."""
    n = len(pts)
    dist = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    circ = 2.0 * math.pi * max(float(radii[0]), 1e-3)
    wraps = max(1, int(round(circ / _BARK_TILE_M)))
    texel = circ / wraps
    ang = np.arange(sides) * (2.0 * math.pi / sides)
    cos_a, sin_a = np.cos(ang), np.sin(ang)
    rings = np.empty((n, sides, 3))
    x_axis = _perpendicular(_unit(pts[1] - pts[0]))
    d = _UP
    for i in range(n):
        d = _unit(pts[min(i + 1, n - 1)] - pts[max(i - 1, 0)])
        # PARALLEL TRANSPORT of the ring's frame, so the rings do not twist from one to the next
        x_axis = x_axis - d * float(np.dot(x_axis, d))
        nx = np.linalg.norm(x_axis)
        x_axis = x_axis / nx if nx > 1e-9 else _perpendicular(d)
        y_axis = np.cross(d, x_axis)
        r = np.full(sides, float(radii[i]))
        if flare is not None and pts[i][2] < flare["height"]:
            s = max(0.0, 1.0 - (pts[i][2] + root_depth) / (flare["height"] + root_depth))
            lobes = np.maximum(0.0, np.cos(flare["lobes"] * ang + flare["offset"])) ** 3
            r = r * (1.0 + flare["amount"] * s * s * (0.55 + 0.45 * lobes))
        rings[i] = pts[i] + np.outer(cos_a * r, x_axis) + np.outer(sin_a * r, y_axis)
    tip_len = max(float(radii[-1]), 0.001) * 1.5
    verts = np.concatenate([rings.reshape(-1, 3), (pts[-1] + d * tip_len)[None, :]])

    i = np.arange(n - 1)[:, None]
    k = np.arange(sides)[None, :]
    k1 = (k + 1) % sides
    quads = np.stack(np.broadcast_arrays(i * sides + k, i * sides + k1, (i + 1) * sides + k1, (i + 1) * sides + k),
                     axis=-1).reshape(-1, 4)
    u0, u1 = k / sides * wraps, (k + 1) / sides * wraps
    v0, v1 = (dist[:-1] / texel)[:, None], (dist[1:] / texel)[:, None]
    quad_uv = np.stack([np.stack(np.broadcast_arrays(u, v), axis=-1) for u, v in ((u0, v0), (u1, v0), (u1, v1),
                                                                                  (u0, v1))],
                       axis=2).reshape(-1, 4, 2)
    w = weight0 + (weight1 - weight0) * dist / max(dist[-1], 1e-9)
    quad_w = np.repeat(np.stack([w[:-1], w[:-1], w[1:], w[1:]], axis=1), sides, axis=0)
    quad_col = np.empty((len(quads), 4, 4))
    quad_col[..., 0], quad_col[..., 1], quad_col[..., 2], quad_col[..., 3] = 0.5, quad_w, phase, 1.0

    last = (n - 1) * sides
    kk = np.arange(sides)
    tris = np.stack([last + kk, last + (kk + 1) % sides, np.full(sides, n * sides)], axis=1)
    v_end = dist[-1] / texel
    tri_uv = np.stack([np.stack([kk / sides * wraps, np.full(sides, v_end)], axis=1),
                       np.stack([(kk + 1) / sides * wraps, np.full(sides, v_end)], axis=1),
                       np.stack([(kk + 0.5) / sides * wraps, np.full(sides, v_end + tip_len / texel)], axis=1)],
                      axis=1)
    tri_col = np.empty((sides, 3, 4))
    tri_col[..., 0], tri_col[..., 1], tri_col[..., 2], tri_col[..., 3] = 0.5, weight1, phase, 1.0
    return verts, quads, quad_uv, quad_col, tris, tri_uv, tri_col


def _leaf_template(shape):
    """One leaf in its own space: x along the leaf from the stalk (0) to the tip (1), y across, z its face's normal.
    Every triangle winds counter-clockwise seen from +z. A fold along the midrib and a droop toward the tip make it a
    3D leaf rather than a flat card. UVs span the whole 0-1 square. Returns (verts, tris, uvs) as arrays."""
    if shape in _OUTLINES:
        return _outline_leaf(*_OUTLINES[shape])
    if shape == "pinnate":
        return _pinnate_leaf()
    if shape == "needle":
        # A SPRAY OF THREE BLADES fanned 40 degrees each way from one base, a diamond of 2 triangles each: 12 vertices,
        # 6 triangles. Narrower blades, closer together, drew a conifer as bare branches from 25 m.
        verts, tris, uvs = [], [], []
        for k, a in enumerate((-0.7, 0.0, 0.7)):
            c, s = math.cos(a), math.sin(a)
            b = len(verts)
            for x, y, z in ((0.0, 0.0, 0.0), (0.5, 0.07, 0.02 * k), (1.0, 0.0, 0.05 * k), (0.5, -0.07, 0.02 * k)):
                verts.append((x * c - y * s, x * s + y * c, z))
            tris += [(b, b + 3, b + 2), (b, b + 2, b + 1)]
            uvs += [(0.5, 0.0), (1.0, 0.5), (0.5, 1.0), (0.0, 0.5)]
        return np.array(verts, float), np.array(tris, np.int64), np.array(uvs, float)
    # SEVEN POINTS, SIX TRIANGLES: stalk, two pairs of edge points, a midrib point between them, the tip
    w = {"ovate": 0.55, "lanceolate": 0.30, "round": 0.85}[shape]
    x1, xm, x2, h1, h2 = 0.35, 0.55, 0.75, w * 0.5, w * 0.4
    if shape == "round":
        x1, xm, x2, h2 = 0.30, 0.50, 0.72, w * 0.45
    elif shape == "lanceolate":
        x1, x2, h2 = 0.30, 0.70, w * 0.36
    flat =[(0.0, 0.0), (x1, h1), (x1, -h1), (xm, 0.0), (x2, h2), (x2, -h2), (1.0, 0.0)]
    tris = [(0, 3, 1), (0, 2, 3), (1, 3, 4), (3, 6, 4), (3, 5, 6), (2, 5, 3)]
    fold, droop = 0.30, 0.12
    verts = [(x, y, fold * abs(y) - droop * x * x) for x, y in flat]
    uvs = [(0.5 + y / w, x) for x, y in flat]
    return np.array(verts, float), np.array(tris, np.int64), np.array(uvs, float)


# THE OUTLINED LEAVES of street and park trees, each the upper half of its outline from beside the tip back toward the
# stalk (the lower half mirrors it), the points on the midrib between stalk and tip, then fold and droop. Drawn from
# the leaves' botany, measured on nothing: palmate is a plane's or a maple's (a broad middle lobe, a side lobe reaching
# forward and out, a small basal one, shallow sinuses), lobed an oak's of the red-oak kind (narrow pointed lobes, the
# middle pair longest, sinuses cut almost to the midrib, a wedge base), cordate a linden's (a heart, its rounded
# basal lobes reaching back past the stalk, a drawn-out tip). Teeth smaller than a few percent of the leaf are left
# out: they cost triangles on every one of tens of thousands of leaves and are invisible past arm's length.
_OUTLINES = {
    "palmate": (((0.80, 0.13), (0.70, 0.17), (0.58, 0.22), (0.64, 0.38), (0.70, 0.58), (0.54, 0.53), (0.38, 0.46),
                 (0.31, 0.51), (0.14, 0.43), (0.05, 0.27), (0.02, 0.12)), (0.25, 0.55), 0.10, 0.08),
    "lobed": (((0.90, 0.08), (0.80, 0.08), (0.82, 0.19), (0.89, 0.40), (0.68, 0.18), (0.62, 0.09), (0.62, 0.22),
               (0.57, 0.48), (0.44, 0.22), (0.38, 0.09), (0.34, 0.21), (0.27, 0.32), (0.18, 0.18), (0.08, 0.06)),
              (0.40, 0.70), 0.22, 0.08),
    "cordate": (((0.80, 0.14), (0.52, 0.35), (0.22, 0.42), (-0.04, 0.33), (-0.08, 0.12)), (0.50,), 0.18, 0.14),
}


def _cross2(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _ear_clip(poly):
    """Triangles of a simple counter-clockwise polygon, as index triples into it, by ear clipping - enough for an
    outline of a few dozen points, and it handles the deep sinuses of a lobed leaf. A point lying ON a candidate ear
    first blocks it (a vertex on a triangle's edge would leave a crack); only when no ear is clear that way is it let
    through. Collinear runs (the midrib) are never an ear, so no zero-area triangle is made."""
    def inside(p, a, b, c, strict):
        d = (_cross2(a, b, p), _cross2(b, c, p), _cross2(c, a, p))
        return all(x > 1e-12 for x in d) if strict else all(x >= -1e-12 for x in d)

    idx = list(range(len(poly)))
    tris = []
    while len(idx) > 3:
        n, hit = len(idx), None
        for strict in (False, True):
            for k in range(n):
                i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % n]
                a, b, c = poly[i0], poly[i1], poly[i2]
                if _cross2(a, b, c) <= 1e-12:
                    continue
                if any(inside(poly[j], a, b, c, strict) for j in idx if j not in (i0, i1, i2)):
                    continue
                hit = k
                break
            if hit is not None:
                break
        if hit is None:
            raise MifOpError("a leaf outline in build_tree is not a simple counter-clockwise polygon - a defect in this "
                             "op, not in the request. NOTHING was created.")
        tris.append((idx[hit - 1], idx[hit], idx[(hit + 1) % n]))
        del idx[hit]
    if _cross2(*(poly[i] for i in idx)) > 1e-12:
        tris.append(tuple(idx))
    return tris


def _outline_leaf(upper, midrib, fold, droop):
    """A leaf from its upper half-outline: the stalk at (0, 0), the tip at (1, 0), the lower half the mirror. Each half
    is triangulated ON ITS OWN (stalk, midrib, tip, outline), so the midrib is an edge of every triangle along it and
    the fold creases there. UVs span the outline's own box, 0-1 both ways - a cordate leaf's basal lobes reach behind
    the stalk and still map inside the square."""
    mids = [(x, 0.0) for x in midrib]
    lower = [(x, -y) for x, y in reversed(upper)]
    flat = [(0.0, 0.0)] + mids + [(1.0, 0.0)] + list(upper) + lower
    axis = list(range(len(mids) + 2))
    iu = list(range(len(axis), len(axis) + len(upper)))
    il = list(range(len(axis) + len(upper), len(flat)))
    tris = []
    for poly in (axis + iu, [0] + il + axis[:0:-1]):
        tris += [tuple(poly[i] for i in t) for t in _ear_clip([flat[i] for i in poly])]
    xs = [p[0] for p in flat]
    x0, span = min(xs), max(xs) - min(xs)
    ymax = max(abs(p[1]) for p in flat)
    verts = [(x, y, fold * abs(y) - droop * max(0.0, x) ** 2) for x, y in flat]
    uvs = [(0.5 + y / (2.0 * ymax), (x - x0) / span) for x, y in flat]
    return np.array(verts, float), np.array(tris, np.int64), np.array(uvs, float)


def _pinnate_leaf(pairs=7):
    """A compound leaf, a honey locust's or an ash's: a rachis (a thin strip, 2 triangles) carrying `pairs` opposite
    pairs of leaflets, each a six-point ellipse of 4 triangles folded along its own midrib, set out at 72 degrees near
    the stalk closing to 42 at the tip, largest in the middle. Each leaflet maps the whole 0-1 square, so a leaf
    texture drawn for one leaflet serves them all; the rachis maps a sliver at the square's stalk end. The leaflets
    tilt up off the rachis and the rachis arches down toward its tip."""
    w = 0.008
    verts = [(0.0, w), (0.0, -w), (1.0, -0.5 * w), (1.0, 0.5 * w)]
    uvs = [(0.5 + y, 0.02 * x) for x, y in verts]
    tris = [(0, 1, 2), (0, 2, 3)]
    shape = ((0.0, 0.0), (0.30, -0.5), (0.75, -0.42), (1.0, 0.0), (0.75, 0.42), (0.30, 0.5))
    fan = ((0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 5))
    for k in range(pairs):
        t = (k + 0.5) / pairs
        x0 = 0.12 + 0.84 * t
        ang = math.radians(72.0 - 30.0 * t)
        size = 0.24 * (0.8 + 0.35 * math.sin(math.pi * min(1.0, 1.1 * t)))
        for side in (1.0, -1.0):
            c, s = math.cos(ang), math.sin(ang) * side
            b = len(verts)
            for lx, ly in shape:
                px, py = lx * size, ly * size * 0.42
                verts.append((x0 + px * c - py * s, side * 0.5 * w + px * s + py * c))
                uvs.append((0.5 + ly, lx))
            tris += [(b + i, b + j, b + m) for i, j, m in fan]
    verts = [(x, y, 0.06 * abs(y) - 0.10 * x * x) for x, y in verts]
    return np.array(verts, float), np.array(tris, np.int64), np.array(uvs, float)


# A LEAF CLUSTER, the design's lighter tree: one twig carrying a spray of leaves, built ONCE as a template and placed
# like a single leaf - fewer, richer parts. The budget is about 80 triangles a cluster, so a cluster carries as many
# leaves as fit beside its twig: twelve ovate or needle sprays, six cordate, three palmate, two lobed, one pinnate.
_CLUSTER_TRIS = 80
_CLUSTER_TWIG_TRIS = 6
_CLUSTER_SPAN = 3.0                     # a cluster is this many leaf lengths from its twig's base to its outermost tip


def _cluster_per(shape):
    """How many leaves of `shape` a cluster of about _CLUSTER_TRIS triangles carries beside its twig."""
    per_leaf = len(_leaf_template(shape)[1])
    return max(1, min(16, int(round((_CLUSTER_TRIS - _CLUSTER_TWIG_TRIS) / float(per_leaf)))))


def _cluster_template(shape, per):
    """One cluster in its own space, in units of its span: x along the twig from its base (0) toward the tip (1), z the
    cluster's face. The twig is an open three-sided prism (6 triangles) to x 0.8; `per` leaves of 1/_CLUSTER_SPAN
    length leave it from 0.15 to its end at a golden-angle turn and 50 degrees forward, each turned to face +z, the
    last one carrying on from the tip. Every leaf keeps its own whole 0-1 UV square; the twig maps a sliver at the
    square's stalk end, as the compound leaf's rachis does. Returns (verts, tris, uvs, part) - part is the leaf
    index (the twig -1) of each triangle, so a per-leaf color value varies inside one cluster."""
    lv, lt, luv = _leaf_template(shape)
    leaf_len = 1.0 / _CLUSTER_SPAN
    twig_end, tw = 0.8, 0.006
    verts, tris, uvs, part = [], [], [], []
    for x in (0.0, twig_end):
        for k in range(3):
            a = 2.0 * math.pi * k / 3.0
            verts.append((x, tw * math.cos(a), tw * math.sin(a)))
            uvs.append((0.49 + 0.01 * k, 0.02 * x))
    for k in range(3):
        j = (k + 1) % 3
        tris += [(k, j, 3 + j), (k, 3 + j, 3 + k)]
        part += [-1, -1]
    rng = random.Random(per * 7919 + len(lv))
    for i in range(per):
        last = i == per - 1
        t = twig_end if last else 0.15 + (twig_end - 0.15) * i / max(1, per - 1)
        phi = i * math.radians(137.5)
        lean = 0.0 if last else math.radians(50.0 + rng.uniform(-8.0, 8.0))
        d = (math.cos(lean), math.sin(lean) * math.cos(phi), math.sin(lean) * math.sin(phi))
        # THE LEAF FACES THE CLUSTER'S +Z, tilted off it as far as its direction makes it: z' = +z less its part along d
        zl = (-d[2] * d[0], -d[2] * d[1], 1.0 - d[2] * d[2])
        zn = math.sqrt(sum(c * c for c in zl))
        if zn < 1e-6:
            zl, zn = (0.0, -1.0, 0.0), 1.0
        zl = tuple(c / zn for c in zl)
        yl = (zl[1] * d[2] - zl[2] * d[1], zl[2] * d[0] - zl[0] * d[2], zl[0] * d[1] - zl[1] * d[0])
        s = leaf_len * rng.uniform(0.85, 1.15)
        base = len(verts)
        for x, y, z in lv:
            verts.append(tuple(t * (axis == 0) + s * (x * d[axis] + y * yl[axis] + z * zl[axis])
                               for axis in range(3)))
        uvs += [tuple(u) for u in luv]
        tris += [tuple(int(c) + base for c in tri) for tri in lt]
        part += [i] * len(lt)
    return np.array(verts, float), np.array(tris, np.int64), np.array(uvs, float), np.array(part, np.int64)


def _bearers(stems, twig_level):
    """Which stems carry leaves, from how far along, and with what share: every twig from 12% of its length, the outer
    45% of the branches the twigs grow on, and the top 10% of each trunk stem. A crown is foliage out to its tips -
    leaves on the twigs alone left sky between the sprays and a bare spike on top of a conifer."""
    out = []
    for s in stems:
        if s.level == twig_level:
            out.append((s, 0.12, 1.0))
        elif twig_level >= 2 and s.level == twig_level - 1:
            out.append((s, 0.55, 0.35))
        elif twig_level >= 2 and s.level == 0:
            out.append((s, 0.9, 0.3))
    return out


def _leaves(bearers, count, template, size_m, np_rng):
    """`count` leaves (or clusters - `template` is either) spread over the bearing stems in proportion to length times
    share, each set on its stem, leaning 35-65 degrees out from it at a golden-angle turn from its neighbor, its face
    turned up and out toward the light with a random tilt, and sized +-25%. One template, one transform per leaf, all
    in numpy. `template` is (verts, tris, uvs) or, for a cluster, (verts, tris, uvs, part): R then differs per leaf
    inside a cluster. Returns (verts, tris, loop_uv, loop_col, placed)."""
    tv, tt, tuv = template[:3]
    tpart = template[3] if len(template) > 3 else np.zeros(len(tt), np.int64)
    k = len(tv)
    if count <= 0 or not bearers:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64), np.zeros((0, 3, 2)), np.zeros((0, 3, 4)), 0
    # EACH BEARING SPAN RESAMPLED TO THE SAME NUMBER OF POINTS, so stems of every level stack into one array
    span_pts = 6
    pts = np.empty((len(bearers), span_pts, 3))
    share = np.empty(len(bearers))
    w_from, w_to, phases = np.empty(len(bearers)), np.empty(len(bearers)), np.empty(len(bearers))
    for j, (s, t0, weight) in enumerate(bearers):
        p = np.asarray(s.points, dtype=float)
        dist = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
        at = np.linspace(t0 * dist[-1], dist[-1], span_pts)
        pts[j] = np.stack([np.interp(at, dist, p[:, c]) for c in range(3)], axis=1)
        share[j] = dist[-1] * (1.0 - t0) * weight
        w_from[j] = s.weight0 + (s.weight1 - s.weight0) * t0
        w_to[j], phases[j] = s.weight1, s.phase
    owner = np.repeat(np.arange(len(bearers)), np_rng.multinomial(count, share / share.sum()))
    m = len(owner)
    segs = span_pts - 1
    t = np_rng.random(m)
    f = np.minimum(segs - 1e-9, t * segs)
    seg = f.astype(np.int64)
    a, b = pts[owner, seg], pts[owner, seg + 1]
    pos = a + (b - a) * (f - seg)[:, None]
    dirs = (b - a) / np.maximum(1e-12, np.linalg.norm(b - a, axis=1, keepdims=True))
    ref = np.where(np.abs(dirs[:, 2:3]) < 0.9, _UP, np.array((1.0, 0.0, 0.0)))
    perp = np.cross(dirs, ref)
    perp /= np.linalg.norm(perp, axis=1, keepdims=True)
    other = np.cross(dirs, perp)
    phi = np.arange(m) * math.radians(137.5) + np_rng.uniform(-0.4, 0.4, m)
    radial = perp * np.cos(phi)[:, None] + other * np.sin(phi)[:, None]
    lean = np.radians(np_rng.uniform(35, 65, m))
    axis_x = dirs * np.cos(lean)[:, None] + radial * np.sin(lean)[:, None]
    axis_x /= np.linalg.norm(axis_x, axis=1, keepdims=True)
    # THE FACE LOOKS UP AND OUT, toward the light a crown actually gets: straight up alone put every leaf edge-on to a
    # camera at eye level, and the first render read as a sparse crown for that reason alone
    outward = np.concatenate([pos[:, :2], np.zeros((m, 1))], axis=1)
    outward /= np.maximum(1e-9, np.linalg.norm(outward, axis=1, keepdims=True))
    bend = np_rng.uniform(0.3, 0.8, m)[:, None]
    up = _UP * (1.0 - bend) + outward * bend + np_rng.normal(0.0, 0.3, (m, 3))
    axis_z = up - axis_x * np.sum(up * axis_x, axis=1, keepdims=True)
    axis_z /= np.maximum(1e-9, np.linalg.norm(axis_z, axis=1, keepdims=True))
    axis_y = np.cross(axis_z, axis_x)
    local = tv[None, :, :] * (size_m * np_rng.uniform(0.75, 1.25, m))[:, None, None]
    verts = (pos[:, None, :] + local[:, :, 0:1] * axis_x[:, None, :] + local[:, :, 1:2] * axis_y[:, None, :]
             + local[:, :, 2:3] * axis_z[:, None, :]).reshape(-1, 3)
    tris = (tt[None, :, :] + (np.arange(m) * k)[:, None, None]).reshape(-1, 3)
    loop_uv = np.tile(tuv[tt], (m, 1, 1))
    # G: the stem's wind weight where the leaf sits, rising to 1 at the leaf's tip; B: the stem's (its limb's) phase
    w_at = w_from[owner] + (w_to[owner] - w_from[owner]) * t
    x_local = np.clip(tv[:, 0], 0.0, 1.0)[tt]                                  # (tris, 3)
    loop_col = np.empty((m, len(tt), 3, 4))
    loop_col[..., 0] = np.mod(np_rng.random(m)[:, None, None] + 0.618034 * np.maximum(tpart, 0)[None, :, None], 1.0)
    loop_col[..., 1] = w_at[:, None, None] + (1.0 - w_at)[:, None, None] * x_local[None, :, :]
    loop_col[..., 2] = phases[owner][:, None, None]
    loop_col[..., 3] = 1.0
    return verts, tris, loop_uv, loop_col.reshape(-1, 3, 4), m


def _find_type(nodes, type_name):
    return next((n for n in nodes if n.type == type_name), None)


def _sock(sockets, *names):
    """A socket by the first of `names` this Blender has - socket names move between releases (Noise's Fac became
    Factor at 5.0), and a hard-coded one fails on the build that renamed it."""
    for name in names:
        s = sockets.get(name)
        if s is not None:
            return s
    raise MifOpError("a shader socket build_tree needs is missing on this Blender: wanted %s, the node has %s."
                     % (" or ".join(names), ", ".join(s.name for s in sockets)))


def _materials(base, bark_color, leaf_color, leaf_color2):
    """The two materials, nodes found by TYPE, never by their localized names. Bark: a noise stretched along the stem
    (furrows run lengthwise) through a two-tone ramp, with the same noise as bump. Leaves: vertex color R picks
    between the two greens, mixed 30% with translucency for the light through them. Both are two-sided, as every
    Blender material is."""
    bark = bpy.data.materials.new(base + "_Bark")
    bark.use_nodes = True
    nt = bark.node_tree
    bsdf = _find_type(nt.nodes, "BSDF_PRINCIPLED")
    coord = nt.nodes.new("ShaderNodeTexCoord")
    mapping = nt.nodes.new("ShaderNodeMapping")
    _sock(mapping.inputs, "Scale").default_value = (8.0, 1.0, 1.0)
    noise = nt.nodes.new("ShaderNodeTexNoise")
    _sock(noise.inputs, "Scale").default_value = 2.5
    _sock(noise.inputs, "Detail").default_value = 10.0
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.35
    ramp.color_ramp.elements[0].color = tuple(c * 0.45 for c in bark_color) + (1.0,)
    ramp.color_ramp.elements[1].position = 0.65
    ramp.color_ramp.elements[1].color = tuple(bark_color) + (1.0,)
    bump = nt.nodes.new("ShaderNodeBump")
    _sock(bump.inputs, "Strength").default_value = 0.8
    fac = _sock(noise.outputs, "Fac", "Factor")
    nt.links.new(_sock(coord.outputs, "UV"), _sock(mapping.inputs, "Vector"))
    nt.links.new(_sock(mapping.outputs, "Vector"), _sock(noise.inputs, "Vector"))
    nt.links.new(fac, ramp.inputs[0])
    nt.links.new(ramp.outputs[0], _sock(bsdf.inputs, "Base Color"))
    nt.links.new(fac, _sock(bump.inputs, "Height"))
    nt.links.new(_sock(bump.outputs, "Normal"), _sock(bsdf.inputs, "Normal"))
    _sock(bsdf.inputs, "Roughness").default_value = 0.9

    leaves = bpy.data.materials.new(base + "_Leaves")
    leaves.use_nodes = True
    nt = leaves.node_tree
    bsdf = _find_type(nt.nodes, "BSDF_PRINCIPLED")
    out = _find_type(nt.nodes, "OUTPUT_MATERIAL")
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "Col"
    # THE DATA IS STORED IN sRGB SPACE (see op_build_tree) and the Attribute node hands it over linearized; a gamma of
    # 1/2.2 brings the per-leaf value back near what was written, so the two greens mix as they did before.
    unlin = nt.nodes.new("ShaderNodeGamma")
    _sock(unlin.inputs, "Gamma").default_value = 1.0 / 2.2
    sep = nt.nodes.new("ShaderNodeSeparateColor")
    pick = nt.nodes.new("ShaderNodeValToRGB")
    pick.color_ramp.elements[0].color = tuple(leaf_color) + (1.0,)
    pick.color_ramp.elements[1].color = tuple(leaf_color2) + (1.0,)
    nt.links.new(_sock(attr.outputs, "Color"), _sock(unlin.inputs, "Color"))
    nt.links.new(_sock(unlin.outputs, "Color"), sep.inputs[0])
    nt.links.new(sep.outputs[0], pick.inputs[0])
    nt.links.new(pick.outputs[0], _sock(bsdf.inputs, "Base Color"))
    _sock(bsdf.inputs, "Roughness").default_value = 0.6
    trans = nt.nodes.new("ShaderNodeBsdfTranslucent")
    nt.links.new(pick.outputs[0], _sock(trans.inputs, "Color"))
    mix = nt.nodes.new("ShaderNodeMixShader")
    mix.inputs[0].default_value = 0.3
    nt.links.new(bsdf.outputs[0], mix.inputs[1])
    nt.links.new(trans.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], _sock(out.inputs, "Surface"))
    return bark, leaves


def _color3(params, key, default):
    v = params.get(key)
    if v is None:
        return tuple(default)
    if not isinstance(v, (list, tuple)) or len(v) != 3:
        raise MifOpError("'%s' takes [r, g, b] (linear, 0-1), got %r. NOTHING was created." % (key, v))
    for c in v:
        if isinstance(c, bool) or not isinstance(c, (int, float)) or not math.isfinite(c) or not 0.0 <= c <= 1.0:
            raise MifOpError("'%s' components must be numbers from 0 to 1, got %r. NOTHING was created." % (key, v))
    return tuple(float(c) for c in v)


def _hull2d(xy, eps):
    """Indices of the 2D convex hull of xy (Andrew's monotone chain), dropping points within eps of collinear."""
    order = sorted(range(len(xy)), key=lambda i: (xy[i][0], xy[i][1]))

    def cross(o, a, b):
        return (xy[a][0] - xy[o][0]) * (xy[b][1] - xy[o][1]) - (xy[a][1] - xy[o][1]) * (xy[b][0] - xy[o][0])

    lower, upper = [], []
    for i in order:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], i) <= eps:
            lower.pop()
        lower.append(i)
    for i in reversed(order):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], i) <= eps:
            upper.pop()
        upper.append(i)
    return lower[:-1] + upper[:-1]


def _trunk_hull(obj, coll, trunk_parts, cut_z, radius):
    """A UCX_<tree>_00 convex hull round the trunk below cut_z, audited the way create_collision_hull audits.
    Returns (hull dict or None, note or None).

    TWO RINGS, NOT ALL OF THEM. Handed every ring of the trunk, bmesh's convex_hull kept points that lie within ITS
    tolerance of a face plane and outside the audit's (1e-5 of the span): 1 convexity violation on the default tree,
    38 once a jitter was added to break the rings' coplanarity. So it gets input with nothing near-degenerate in it:
    the lowest ring at or above the ground - the flare's widest where anything can touch it; the one below ground
    made the hull 0.91 m round a 0.5 m trunk - cut down to its own 2D convex outline, and the ring at the cut taken on
    the OTHER half of its vertices, so no side face is a near-flat quad. The flare only narrows upward, so the frustum
    between them holds the trunk."""
    name = "UCX_%s_00" % obj.name
    if name in bpy.data.objects:
        return None, ("an object named '%s' already exists, so no hull was made: Blender would name this one '%s.001', "
                      "and that suffix survives FBX export, attaching the hull to nothing." % (name, name))
    points = []
    for verts, sides in trunk_parts:
        rings = verts[:-1].reshape(-1, sides, 3)
        below_at = [i for i, r in enumerate(rings) if r[:, 2].mean() <= cut_z]
        below = [rings[i] for i in below_at]
        if len(below) < 2:
            continue
        low = next((r for r in below[:-1] if r[:, 2].mean() >= 0.0), below[0])
        points.extend(low[i] for i in _hull2d(low[:, :2].tolist(), (radius * 1e-3) ** 2))
        # THE TOP RING IS CUT AT cut_z, NOT THE LAST RING BELOW IT. Rings sit a segment apart, and on a youngTree of
        # 5 m that is about a meter there: the hull topped out at 0.97 m under a cut of 1.5 m, so half the bare trunk
        # had no collision while topM said 1.5 (benchmark round ten, 5.2.1). The ring at the cut is interpolated vertex
        # for vertex between the rings either side - the bark between two rings is exactly that ruled surface - and
        # only when no ring rises past the cut is the last one below it the top.
        top, k = below[-1], below_at[-1]
        if k + 1 < len(rings):
            above = rings[k + 1]
            z0, z1 = top[:, 2].mean(), above[:, 2].mean()
            if z1 > cut_z >= z0:
                top = top + (above - top) * ((cut_z - z0) / (z1 - z0))
        points.extend(top[1::2])
    if len(points) < 4:
        return None, "the trunk had fewer than two rings below %.2f m, so no hull was made." % cut_z
    bm = _build_hull([tuple(p) for p in points], radius * 1e-4)
    try:
        mesh = bpy.data.meshes.new(name)
        bm.to_mesh(mesh)
    finally:
        bm.free()
    audit = _hull_audit(mesh)
    bad = [k for k in ("boundaryEdges", "nonManifoldEdges", "looseVertices", "convexityViolations") if audit[k]]
    if audit["eulerCharacteristic"] != 2:
        bad.append("eulerCharacteristic")
    if audit["volume"] <= 1e-9:
        bad.append("volume")
    if bad or audit["vertices"] > _UE_HULL_VERT_LIMIT:
        bpy.data.meshes.remove(mesh)
        return None, ("the trunk hull failed its audit (%s) and was removed rather than kept: a broken hull imports, "
                      "looks like collision, and things fall through it. Measurements: %s"
                      % (", ".join(bad) or "more than %d vertices" % _UE_HULL_VERT_LIMIT, audit))
    hull = bpy.data.objects.new(name, mesh)
    # A COLLISION MESH IS NOT SCENERY: drawn as a wire, and left out of renders - the first render of a tree with its
    # hull showed a solid white frustum where the trunk should be. FBX export takes it either way.
    hull.display_type = "WIRE"
    hull.hide_render = True
    coll.objects.link(hull)
    hull.location = obj.location.copy()
    hull.rotation_euler = obj.rotation_euler.copy()
    hull.scale = obj.scale.copy()
    return {"object": hull.name, "topM": round(float(cut_z), 3), "vertices": audit["vertices"],
            "volume": audit["volume"], "audit": "closed, manifold, convex"}, None


# ---- output gameReady: leaf cards baked from real leaves, a lighter bark, LODs -----------------------------------
# WHY: real leaf geometry cannot fit a game budget - a default broadleaf is 747,610 triangles and a
# conifer 1,328,008 against the 36,014 mesh_quality gameReady holds a large piece to, and a broadleaf cut to 4,000
# leaves is still 65,560 and reads as paddles. A game tree carries its leaves as CARDS: a quad textured with a cluster
# of leaves. Here the cluster is the same twig-and-leaf unit leafClusters builds, baked by bake_texture_set from the
# real geometry onto the quad, so the card shows the leaves the dense tree has; the cards stand where the dense tree's
# clusters stand, on the same skeleton (the same seed grows the same stems).
_GAME_OUTPUTS = ("geometry", "gameReady")
_GAME_BARK_SHARE = 0.45                 # the most of the budget the bark may take; the cards get the rest
_GAME_CARD_TRIS = 2                     # a flat quad
_GAME_SPRAY_STEP = 0.55                 # clusters along a spray's twig, in cluster spans
_GAME_DEFAULT_LODS = (0.5, 0.25)
# MEASURED 2026-11-26 (5.0.1, Cycles, the default broadleafRound against its dense tree from one camera): a card per
# cluster covered 1.00 of the dense tree's pixels at IoU 0.81; 0.65 of a card per cluster thinned the crown to 0.90 at
# 0.77. The cards' translucency at the dense leaves' 0.3 put the mean color 15.2 of 255 off theirs, none 18.1.
_GAME_CARD_COVER = 1.0                  # cards per cluster of the dense tree in clusters
_GAME_TRANSLUCENT = 0.3                 # the dense leaves' share of translucency, given to the card material too


def _game_sides(stem):
    """Sides of a stem's tube in the game-ready bark: about half the dense count, 3 on the thinnest."""
    r = stem.radii[0]
    if stem.level == 0:
        return 10 if r > 0.15 else 8
    if r > 0.08:
        return 6
    if r > 0.02:
        return 4
    return 3


def _thin_points(stem):
    """A branch's polyline with every other point (the ends kept): half the rings. The trunk keeps all of its."""
    if stem.level == 0 or len(stem.points) <= 3:
        return np.asarray(stem.points, dtype=float), np.asarray(stem.radii, dtype=float)
    keep = list(range(0, len(stem.points) - 1, 2)) + [len(stem.points) - 1]
    return np.asarray(stem.points, dtype=float)[keep], np.asarray(stem.radii, dtype=float)[keep]


def _game_bark_tris(stem):
    pts, _r = _thin_points(stem)
    sides = _game_sides(stem)
    return 2 * sides * (len(pts) - 1) + sides


def _game_stems(stems, twig_level, bark_budget):
    """Which stems the game-ready bark builds: every one but the twigs (the cards carry those), then the shortest of the
    deepest level dropped until the bark fits bark_budget triangles. Returns (kept, dropped count). The trunk stays."""
    keep = [s for s in stems if s.level < twig_level or twig_level == 0 or s.level == 0]
    total = sum(_game_bark_tris(s) for s in keep)
    dropped = len(stems) - len(keep)
    while total > bark_budget:
        deepest = max(s.level for s in keep)
        if deepest == 0:
            break
        gone = set()
        for s in sorted((s for s in keep if s.level == deepest), key=lambda s: s.length):
            if total <= bark_budget:
                break
            gone.add(id(s))
            total -= _game_bark_tris(s)
            dropped += 1
        keep = [s for s in keep if id(s) not in gone]
    return keep, dropped


def _spray_template(shape, per, k):
    """k clusters (each _cluster_template's) on one twig in the plane of the card, alternating 35 degrees either side and
    the last one carrying on from the tip; in units of the spray's span along x (0 at its base, 1 at its tip). Returns
    (verts, tris, part, span) - part the leaf index per triangle (-1 a twig), span the spray's length in cluster spans."""
    cv, ct, _uv, cpart = _cluster_template(shape, per)
    if k == 1:
        return cv, ct, cpart, 1.0
    verts, tris, part = [], [], []
    tw = 0.006
    end = _GAME_SPRAY_STEP * (k - 1)
    for x in (0.0, end):
        for j in range(3):
            a = 2.0 * math.pi * j / 3.0
            verts.append((x, tw * math.cos(a), tw * math.sin(a)))
    for j in range(3):
        m = (j + 1) % 3
        tris += [(j, m, 3 + m), (j, 3 + m, 3 + j)]
        part += [-1, -1]
    for i in range(k):
        ang = 0.0 if i == k - 1 else math.radians(35.0) * (1 if i % 2 == 0 else -1)
        c, s = math.cos(ang), math.sin(ang)
        rot = np.array(((c, s, 0.0), (-s, c, 0.0), (0.0, 0.0, 1.0)))
        placed = cv @ rot
        placed[:, 0] += _GAME_SPRAY_STEP * i
        base = len(verts)
        verts.extend(map(tuple, placed.tolist()))
        tris.extend((a + base, b + base, d + base) for a, b, d in ct.tolist())
        part.extend((p + i * per) if p >= 0 else -1 for p in cpart.tolist())
    verts = np.array(verts, float)
    span = float(verts[:, 0].max())
    return verts / span, np.array(tris, np.int64), np.array(part, np.int64), span


def _card_template(spray_verts):
    """The card: a quad round the spray's outline seen from +z, 2% over it, at the spray's middle depth; UV the whole
    0-1 square, u along the spray. Returns ((verts, tris, uvs), aspect height/width, z extent below and above)."""
    lo, hi = spray_verts.min(axis=0), spray_verts.max(axis=0)
    pad = 0.02 * float(hi[0] - lo[0])
    x0, x1, y0, y1 = 0.0, float(hi[0]) + pad, float(lo[1]) - pad, float(hi[1]) + pad
    zc = 0.5 * float(lo[2] + hi[2])
    verts = np.array(((x0, y0, zc), (x1, y0, zc), (x1, y1, zc), (x0, y1, zc)), float)
    tris = np.array(((0, 1, 2), (0, 2, 3)), np.int64)
    uvs = np.array(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), float)
    return (verts, tris, uvs), (y1 - y0) / (x1 - x0), (zc - float(lo[2]), float(hi[2]) - zc)


def _bake_leaf_cards(base, spray, card, depth, unit_m, leaf_color, leaf_color2, folder, size, aspect):
    """Bake the spray's real leaves onto the card (bake_card_material). The spray and the bake card are temporary and
    removed. Returns (material, maps {type: file}, (width, height))."""
    sv, st, spart = spray
    hi_me = bpy.data.meshes.new(base + "_CardSource")
    hi_me.from_pydata((sv * unit_m).tolist(), [], st.tolist())
    col = hi_me.color_attributes.new(name="Col", type="BYTE_COLOR", domain="CORNER")
    rng = np.random.default_rng(len(st))
    per_leaf = rng.random(int(spart.max()) + 2)
    corner = np.empty((len(st), 3, 4))
    corner[..., 0] = np.where(spart >= 0, per_leaf[np.maximum(spart, 0)], 0.5)[:, None]
    corner[..., 1], corner[..., 2], corner[..., 3] = 1.0, 0.0, 1.0
    col.data.foreach_set("color_srgb", corner.astype(np.float32).ravel())
    hi_me.update()
    cv, ct, cuv = card
    lo_me = bpy.data.meshes.new(base + "_CardBake")
    lo_me.from_pydata((cv * unit_m).tolist(), [], ct.tolist())
    lo_me.uv_layers.new(name="UVMap").data.foreach_set("uv", cuv[ct].astype(np.float32).ravel())
    lo_me.update()
    hi = bpy.data.objects.new(base + "_CardSource", hi_me)
    lo = bpy.data.objects.new(base + "_CardBake", lo_me)
    scene_coll = bpy.context.scene.collection
    scene_coll.objects.link(hi)
    scene_coll.objects.link(lo)
    bark_tmp, leaf_tmp = _materials(base + "_CardSource", (0.5, 0.5, 0.5), leaf_color, leaf_color2)
    hi_me.materials.append(leaf_tmp)
    width = int(size)
    height = finite_int(max(64, 2 ** round(math.log2(max(1.0, size * aspect)))), "the card texture's height")
    below, above = depth
    reach = (below + above) * unit_m
    try:
        mat, files = bake_card_material(lo, [hi], folder, "T_%s_LeafCard_{suffix}" % base, width, height,
                                        above * unit_m + 0.01 * reach + 1e-4, reach * 1.04 + 2e-4, base + "_LeafCards")
    finally:
        remove_objects([hi, lo])
        bpy.data.meshes.remove(hi_me)
        bpy.data.meshes.remove(lo_me)
        bpy.data.materials.remove(bark_tmp)
        bpy.data.materials.remove(leaf_tmp)
    return mat, files, (width, height)


def bake_card_material(lo, sources, folder, pattern, width, height, extrusion, max_ray, mat_name, margin=16,
                       normal=True, types=None, alpha=True, uv_layer=None):
    """Bake `sources` (real leaves, blades) onto the card mesh `lo` through bake_texture_set, in this Blender, high to
    low: base color and normal with a margin, then alpha with none - a margin would grow the leaves' outline into the
    empty card. Then the material, through the helper bake_texture_set's bakedMaterial uses, cut by its alpha in EEVEE
    and given the dense leaves' translucency. build_tree's and build_plant's game-ready outputs share it. normal false
    bakes no normal map: on build_plant's upright slices one darkened the render (a fern's mean color 48.6 off the
    geometry's with it, 18.5 without), where a tree's cards, facing the way their leaves do, came 1.5 nearer with it.
    build_rock's game-ready output bakes a solid mesh with it: `types` replaces the first pass's maps, alpha false bakes
    no alpha and leaves the material opaque (no cut, no translucency), and uv_layer is the layer baked through and read
    by the material (None: the active one).
    Returns (material, maps {type: file})."""
    from .ops_gamebake import _wire_baked_material, op_bake_texture_set   # noqa: PLC0415
    from .ops_material import op_set_material_settings                   # noqa: PLC0415
    files = {}
    had = [m for m in lo.data.materials]
    first = list(types) if types else (["BASE_COLOR", "NORMAL"] if normal else ["BASE_COLOR"])
    for maps, mg in ((first, margin),) + (((["ALPHA"], 0),) if alpha else ()):
        req = {"object": lo.name, "source": [o.name for o in sources], "types": maps, "folder": folder,
               "pattern": pattern, "width": width, "height": height, "samples": 8, "margin": mg,
               "extrusion": extrusion, "maxRayDistance": max_ray, "outOfProcess": False}
        if uv_layer:
            req["uvLayer"] = uv_layer
        r = op_bake_texture_set(req)
        for row in r.get("maps") or []:
            files[row["type"]] = row["file"]
    # bake_texture gives a card with no material one to hold its image node (<object>_BakeMat): taken off again, so
    # the card's only slot is the baked material
    added = [m for m in lo.data.materials if m not in had]
    while len(lo.data.materials) > len(had):
        lo.data.materials.pop(index=len(lo.data.materials) - 1)
    for m in added:
        if m is not None and m.users == 0:
            bpy.data.materials.remove(m)
    plan = {"name": mat_name, "object": lo.name, "assign": False, "single": True, "green": "OPENGL",
            "uvMap": uv_layer or None}
    wired = _wire_baked_material(plan, files)
    mat = bpy.data.materials[wired["name"]]
    if not alpha:
        return mat, files
    # EEVEE CUTS THE CARD BY ITS ALPHA only when told to (Cycles reads the shader); from 4.2 the setting stores as
    # HASHED, which cuts too
    op_set_material_settings({"material": mat.name, "blendMethod": "CLIP", "alphaThreshold": 0.5})
    if _GAME_TRANSLUCENT > 0:
        _card_translucency(mat)
    return mat, files


def _card_translucency(mat):
    """The dense leaves' look on the cards: the baked color also through a Translucent BSDF, mixed in at the same share,
    and the whole cut by the baked alpha (a Mix with a Transparent BSDF) - the translucent half would otherwise draw the
    card's empty corners. Blender's preview only: an engine reads the textures and makes its own foliage shading."""
    nt = mat.node_tree
    bsdf = _find_type(nt.nodes, "BSDF_PRINCIPLED")
    out = _find_type(nt.nodes, "OUTPUT_MATERIAL")
    if bsdf is None or out is None:
        return
    color_in, alpha_in = _sock(bsdf.inputs, "Base Color"), _sock(bsdf.inputs, "Alpha")
    if not color_in.is_linked or not alpha_in.is_linked:
        return
    trans = nt.nodes.new("ShaderNodeBsdfTranslucent")
    nt.links.new(color_in.links[0].from_socket, _sock(trans.inputs, "Color"))
    mix = nt.nodes.new("ShaderNodeMixShader")
    mix.inputs[0].default_value = _GAME_TRANSLUCENT
    nt.links.new(bsdf.outputs[0], mix.inputs[1])
    nt.links.new(trans.outputs[0], mix.inputs[2])
    cut = nt.nodes.new("ShaderNodeMixShader")
    clear = nt.nodes.new("ShaderNodeBsdfTransparent")
    nt.links.new(alpha_in.links[0].from_socket, cut.inputs[0])
    nt.links.new(clear.outputs[0], cut.inputs[1])
    nt.links.new(mix.outputs[0], cut.inputs[2])
    nt.links.new(cut.outputs[0], _sock(out.inputs, "Surface"))


def game_lods(params, default):
    """lods, checked before anything is made by build_lod_chain's own rules (it would refuse them after the build):
    up to _MAX_LODS - 1 falling fractions, each above 0 and below 1; [] for none."""
    from .ops_kit import _MAX_LODS                                          # noqa: PLC0415
    raw = params.get("lods")
    lods = list(default) if raw is None else raw
    if (not isinstance(lods, (list, tuple)) or any(isinstance(r, bool) or not isinstance(r, (int, float))
                                                    or not math.isfinite(r) or not 0.0 < r < 1.0 for r in lods)
            or any(b >= a for a, b in zip(lods, lods[1:])) or len(lods) > _MAX_LODS - 1):
        raise MifOpError("lods takes build_lod_chain's ratios: up to %d fractions of LOD0's triangles, each above 0 and "
                         "below 1, falling - e.g. [0.5, 0.25], or [] for none; got %r. NOTHING was created."
                         % (_MAX_LODS - 1, raw))
    return [float(r) for r in lods]


def _budget_for(verts, given):
    """(budget, how it was chosen): the one given, else the top of the usual range for the size class of these
    vertices' bounds - the _CONV_TRIS row mesh_quality gameReady holds a mesh to when no budget is given."""
    from .ops_mesh import _CONV_TRIS, _conv_size_class                   # noqa: PLC0415
    if given is not None:
        return given, {"from": "given"}
    longest = float(np.max(verts.max(axis=0) - verts.min(axis=0))) * 100.0
    size_class = _conv_size_class(longest)
    return _CONV_TRIS[size_class][1], {"from": "sizeClass", "sizeClass": size_class, "longestSideCm": round(longest, 1)}


def _stump_spec(params):
    """The stump's numbers, every one checked before anything exists."""
    tail = "NOTHING was created."
    sp = params.get("stump")
    if sp is True:
        sp = {}
    if not isinstance(sp, dict) or set(sp) - _STUMP_KEYS:
        raise MifOpError("stump is true or {%s}, got %r. %s" % (", ".join(sorted(_STUMP_KEYS)), sp, tail))
    # output gameReady swaps a tree's leaves for baked cards and thins its branches; a stump has neither, so the game
    # keys are refused by name here rather than as "shapes a whole tree". output "geometry" is what a stump is.
    game = sorted(k for k in _STUMP_GAME_KEYS if params.get(k) is not None)
    if params.get("output") in (None, "geometry"):
        game = [k for k in game if k != "output"]
    if game:
        named = ", ".join("output %r" % params.get("output") if k == "output" else k for k in game)
        raise MifOpError("%s with stump: output gameReady makes leaf cards and a thinned bark for a whole tree, and a "
                         "stump has neither - it is one closed mesh already. Build the stump with output geometry (the "
                         "default) and without textureFolder, textureSize, lods or triangleBudget, then hold it to a "
                         "budget with mesh_quality and make LODs with build_lod_chain. %s" % (named, tail))
    extra = sorted(k for k in params if params[k] is not None and k not in _STUMP_TREE_KEYS | _STUMP_GAME_KEYS)
    if extra:
        raise MifOpError("%s shape%s a whole tree and %s not read for a stump. %s" % (
            ", ".join(extra), "" if len(extra) > 1 else "s", "are" if len(extra) > 1 else "is", tail))

    def num(d, key, default, lo, hi, label):
        v = d.get(key, default)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not lo <= v <= hi:
            raise MifOpError("%s is %g to %g, got %r. %s" % (label, lo, hi, v, tail))
        return float(v)
    r0 = params.get("trunkRadius")
    radius = num(sp, "radius", r0 if r0 is not None else 0.25, 0.02, 5.0, "stump.radius (m)")
    if r0 is not None and sp.get("radius") is not None and abs(finite_float(r0, "trunkRadius") - radius) > 1e-9:
        raise MifOpError("trunkRadius and stump.radius say different radii - pass one. %s" % tail)
    out = {"radius": radius,
           "height": num(sp, "height", max(0.3, 2.4 * radius), 0.05, 20.0, "stump.height (m, the cut)"),
           "flare": num(sp, "flare", 1.35, 1.0, 4.0, "stump.flare (the base radius over the trunk's)"),
           "flareHeight": num(sp, "flareHeight", radius, 0.01, 20.0, "stump.flareHeight (m)"),
           "sink": num(sp, "sink", 0.05 * radius * 4, 0.0, 5.0, "stump.sink (m below the ground plane)"),
           "segments": int(num(sp, "segments", 192, 24, 1024, "stump.segments"))}
    rt = sp.get("roots", {})
    if rt is False:
        rt = {"count": 0}
    if not isinstance(rt, dict) or set(rt) - _ROOT_KEYS:
        raise MifOpError("stump.roots is {%s}, got %r. %s" % (", ".join(sorted(_ROOT_KEYS)), rt, tail))
    if rt.get("spread") is not None and rt.get("divePoint") is not None and rt["spread"] != rt["divePoint"]:
        raise MifOpError("stump.roots spread and divePoint are one value (where a root dives into the ground, from the "
                         "axis) - pass one. %s" % tail)
    count = rt.get("count", 5)
    if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= 16:
        raise MifOpError("stump.roots.count is 0-16, got %r. %s" % (count, tail))
    base = radius * out["flare"]
    spread = num(rt, "spread", rt.get("divePoint", 2.4 * radius), 0.0, 50.0, "stump.roots.spread (m)")
    if count and spread <= base:
        raise MifOpError("stump.roots.spread %g m is inside the flared base (%g m) - a root would not leave the trunk. "
                         "%s" % (spread, base, tail))
    seed = rt.get("seed", params.get("seed", 1))
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise MifOpError("stump.roots.seed is a whole number. %s" % tail)
    out["roots"] = {"count": count, "spread": spread, "seed": seed,
                    "archHeight": num(rt, "archHeight", 1.1 * radius, 0.01, 20.0, "stump.roots.archHeight (m)"),
                    "thickness": num(rt, "thickness", 0.5 * radius, 0.005, 20.0, "stump.roots.thickness (m)")}
    ct = sp.get("cut", {})
    if not isinstance(ct, dict) or set(ct) - _CUT_KEYS:
        raise MifOpError("stump.cut is {%s}, got %r. %s" % (", ".join(sorted(_CUT_KEYS)), ct, tail))
    kind = str(ct.get("kind", "SAW")).upper()
    if kind not in _CUT_KINDS:
        raise MifOpError("stump.cut.kind is SAW, NOTCH (a felling notch with its hinge) or BROKEN, got %r. %s"
                         % (ct.get("kind"), tail))
    d = ct.get("direction", [1.0, 0.0])
    if not isinstance(d, (list, tuple)) or len(d) not in (2, 3) or not all(
            isinstance(c, (int, float)) and not isinstance(c, bool) and math.isfinite(c) for c in d) \
            or math.hypot(d[0], d[1]) < 1e-9:
        raise MifOpError("stump.cut.direction is the fall direction [x, y] (not zero), got %r. %s" % (d, tail))
    for key in ("notchStep", "hinge", "hingeHeight", "breakHeight", "angle"):
        if ct.get(key) is not None and {"notchStep": "NOTCH", "hinge": "NOTCH", "hingeHeight": "NOTCH",
                                        "breakHeight": "BROKEN", "angle": "SAW"}[key] != kind:
            raise MifOpError("stump.cut.%s is read by kind %s only; kind is %s. %s" % (
                key, {"notchStep": "NOTCH", "hinge": "NOTCH", "hingeHeight": "NOTCH", "breakHeight": "BROKEN",
                      "angle": "SAW"}[key], kind, tail))
    out["cut"] = {"kind": kind, "direction": [d[0] / math.hypot(d[0], d[1]), d[1] / math.hypot(d[0], d[1])],
                  "angle": num(ct, "angle", 0.0, 0.0, 45.0, "stump.cut.angle (degrees)"),
                  "notchStep": num(ct, "notchStep", 0.12 * radius, 0.0, 20.0, "stump.cut.notchStep (m)"),
                  "hinge": num(ct, "hinge", 0.12 * radius, 0.0, 20.0, "stump.cut.hinge (m wide)"),
                  "hingeHeight": num(ct, "hingeHeight", 0.15 * radius, 0.0, 20.0, "stump.cut.hingeHeight (m)"),
                  "breakHeight": num(ct, "breakHeight", 0.5 * radius, 0.0, 20.0, "stump.cut.breakHeight (m)")}
    if kind == "SAW":
        drop = math.tan(math.radians(out["cut"]["angle"])) * radius * out["flare"]
    elif kind == "NOTCH":
        drop = out["cut"]["notchStep"]
    else:
        drop = 0.0
    if drop >= 0.8 * out["height"]:
        raise MifOpError("the cut drops %g m below its height %g m - it would reach the ground. %s"
                         % (drop, out["height"], tail))
    out["cut"]["drop"] = drop
    return out


def _stump_top(cut, x, y, rtop, nz):
    """The cut face's height offset at (x, y): SAW a plane tilted along the fall direction; NOTCH the back cut at 0,
    a hinge strip of torn fibres standing up, the notch's floor a step below; BROKEN ridged splinters 0 to breakHeight."""
    dx, dy = cut["direction"]
    u = (x * dx + y * dy) / max(rtop, 1e-9)
    if cut["kind"] == "SAW":
        return -math.tan(math.radians(cut["angle"])) * (u + 1.0) * rtop * 0.5
    if cut["kind"] == "NOTCH":
        h0, hw = 0.2, cut["hinge"] / max(rtop, 1e-9)
        if u < h0:
            return 0.0
        if u < h0 + hw:
            t = (u - h0) / max(hw, 1e-9)
            fib = 0.6 + 0.4 * abs(nz(x * 40.0 / rtop, y * 4.0 / rtop))
            return cut["hingeHeight"] * math.sin(math.pi * t) * fib
        return -cut["notchStep"]
    a = 1.0 - abs(nz(x * 2.5 / rtop, y * 2.5 / rtop))
    b = abs(nz(x * 9.0 / rtop + 7.3, y * 9.0 / rtop))
    return cut["breakHeight"] * min(1.0, 0.75 * a * a + 0.5 * b ** 3)


def _build_stump(params, started):
    """build_tree stump (2026-11-25): a tree stump - flared trunk, buttress roots and a cut - as ONE closed manifold
    mesh: rings round the axis whose radius rises into a lobe at each root, from a flat ground-plane cap `sink` below
    z 0 up to the cut. Slot 0 bark (wall and ground cap), slot 1 wood (the cut face)."""
    from mathutils import Vector, noise as mnoise   # noqa: PLC0415
    import bmesh                                     # noqa: PLC0415
    tail = "NOTHING was created."
    spec = _stump_spec(params)
    name = take(params, "name", default="Stump", kind=str)
    if not name.strip() or any(ord(ch) < 32 for ch in name):
        raise MifOpError("name cannot be empty or hold a control character. %s" % tail)
    bark_color = _color3(params, "barkColor", (0.24, 0.17, 0.11))
    placed = _place_values(params)
    coll = _resolve_collection(params)
    R, H, segs = spec["radius"], spec["height"], spec["segments"]
    f, fh, sink = spec["flare"], spec["flareHeight"], spec["sink"]
    rs, cut = spec["roots"], spec["cut"]
    rng = random.Random(rs["seed"])
    off = Vector((rng.uniform(-300, 300), rng.uniform(-300, 300), rng.uniform(-300, 300)))

    def nz(a, b):
        return mnoise.noise(Vector((a, b, 0.0)) + off, noise_basis="PERLIN_ORIGINAL")
    roots = []
    for k in range(rs["count"]):
        ang = 2.0 * math.pi * (k + rng.uniform(-0.3, 0.3)) / rs["count"]
        roots.append((ang, rs["spread"] * rng.uniform(0.8, 1.1), rs["archHeight"] * rng.uniform(0.75, 1.2)))

    def base(z):
        return R * (1.0 + (f - 1.0) * math.exp(-max(z, 0.0) / fh))

    def radius_at(th, z):
        b = base(z)
        r = b
        for ang, spread, arch in roots:
            zz = max(z, 0.0)
            if zz >= arch or spread <= b:
                continue
            ext = (spread - b) * (1.0 - zz / arch) ** 2
            sigma = rs["thickness"] / (b + 0.5 * ext)
            da = (th - ang + math.pi) % (2.0 * math.pi) - math.pi
            r += ext * math.exp(-(da / sigma) ** 2)
        return r
    top_low = H - cut["drop"] - 0.02 * H
    n_rows = max(16, min(200, int(H / (2.0 * math.pi * R / segs * 1.5))))
    zs = [-sink] if sink > 0 else []
    zs += [top_low * (k / (n_rows - 1)) ** 1.6 for k in range(n_rows)]
    thetas = [2.0 * math.pi * k / segs for k in range(segs)]
    verts, faces, mats, uvs = [], [], [], []
    rows = []
    tile = 1.0
    for z in zs:
        row = []
        for th in thetas:
            r = radius_at(th, z)
            row.append(len(verts))
            verts.append((r * math.cos(th), r * math.sin(th), z))
        rows.append(row)
    rtop = base(H)
    top = []
    for th in thetas:
        r = radius_at(th, H)
        x, y = r * math.cos(th), r * math.sin(th)
        top.append(len(verts))
        verts.append((x, y, H + _stump_top(cut, x, y, rtop, nz)))
    rows.append(top)
    zrows = zs + [H]
    for j in range(len(rows) - 1):
        a, b = rows[j], rows[j + 1]
        for k in range(segs):
            k1 = (k + 1) % segs
            faces.append([a[k], a[k1], b[k1], b[k]])
            mats.append(0)
            u0, u1 = 2.0 * math.pi * R * k / segs / tile, 2.0 * math.pi * R * (k + 1) / segs / tile
            uvs.append([(u0, zrows[j] / tile), (u1, zrows[j] / tile), (u1, zrows[j + 1] / tile),
                        (u0, zrows[j + 1] / tile)])

    def cap(ring, z_of, mat, flip, uv_off):
        """A disc of concentric rings into the centre - quads, then a fan."""
        steps = 10
        prev = ring
        span = max(max(abs(verts[i][0]), abs(verts[i][1])) for i in ring) or 1.0
        for st in range(1, steps):
            s_ = 1.0 - st / steps
            cur = []
            for i in ring:
                x, y = verts[i][0] * s_, verts[i][1] * s_
                cur.append(len(verts))
                verts.append((x, y, z_of(x, y)))
            for k in range(segs):
                k1 = (k + 1) % segs
                quad = [prev[k], prev[k1], cur[k1], cur[k]]
                faces.append(list(reversed(quad)) if flip else quad)
                mats.append(mat)
                uvs.append([(uv_off + 0.5 + 0.5 * verts[i][0] / span, 0.5 + 0.5 * verts[i][1] / span)
                            for i in faces[-1]])
            prev = cur
        mid = len(verts)
        verts.append((0.0, 0.0, z_of(0.0, 0.0)))
        for k in range(segs):
            k1 = (k + 1) % segs
            tri = [prev[k], prev[k1], mid]
            faces.append(list(reversed(tri)) if flip else tri)
            mats.append(mat)
            uvs.append([(uv_off + 0.5 + 0.5 * verts[i][0] / span, 0.5 + 0.5 * verts[i][1] / span) for i in faces[-1]])
    cap(top, lambda x, y: H + _stump_top(cut, x, y, rtop, nz), 1, False, 2.0)
    cap(rows[0], lambda x, y: zs[0], 0, True, 4.0)
    bm = bmesh.new()
    try:
        bv = [bm.verts.new(v) for v in verts]
        for fc in faces:
            bm.faces.new([bv[i] for i in fc])
        bm.normal_update()
        vol = sum(f_.calc_center_median().dot(f_.normal) * f_.calc_area() for f_ in bm.faces) / 3.0
        boundary = sum(1 for e in bm.edges if len(e.link_faces) == 1)
        nonman = sum(1 for e in bm.edges if len(e.link_faces) > 2)
    finally:
        bm.free()
    if vol < 0:
        faces = [list(reversed(fc)) for fc in faces]
        uvs = [list(reversed(u)) for u in uvs]
        vol = -vol
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    uv = mesh.uv_layers.new(name="UVMap")
    flat = [c for face in uvs for corner in face for c in corner]
    uv.data.foreach_set("uv", flat)
    mesh.polygons.foreach_set("material_index", mats)
    mesh.polygons.foreach_set("use_smooth", [True] * len(faces))
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    _link_new(obj, coll)
    _place(obj, placed)
    bark_m, leaf_m = _materials(obj.name, bark_color, (0.2, 0.4, 0.1), (0.3, 0.5, 0.1))
    bpy.data.materials.remove(leaf_m)
    wood = bpy.data.materials.new(obj.name + "_Wood")
    wood.use_nodes = True
    bsdf = _find_type(wood.node_tree.nodes, "BSDF_PRINCIPLED")
    _sock(bsdf.inputs, "Base Color").default_value = (0.55, 0.40, 0.25, 1.0)
    _sock(bsdf.inputs, "Roughness").default_value = 0.8
    mesh.materials.append(bark_m)
    mesh.materials.append(wood)
    out = _created(obj, coll, "MESH", "build_tree", name)
    reach = max(math.hypot(v[0], v[1]) for v in verts)
    out.update({
        "stump": {"radius": R, "height": H, "flare": f, "flareHeight": fh, "sink": sink, "segments": segs,
                  "rings": len(rows), "reachM": round(reach, 4),
                  "roots": [{"angleDeg": round(math.degrees(a), 2), "spread": round(s_, 4), "archHeight": round(h_, 4)}
                            for a, s_, h_ in roots],
                  "cut": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in cut.items()},
                  "topZ": [round(min(verts[i][2] for i in top), 5), round(max(verts[i][2] for i in top), 5)]},
        "closedCheck": {"boundaryEdges": boundary, "nonManifoldEdges": nonman, "volumeM3": round(vol, 6),
                        "groundCapZ": round(zs[0], 6)},
        "vertices": len(verts), "faces": len(faces),
        "materialSlots": [{"index": 0, "part": "bark (wall and ground cap)", "material": bark_m.name},
                          {"index": 1, "part": "wood (the cut)", "material": wood.name}],
        "pivot": "the trunk's axis at ground level; the ground cap sits sink below it",
        "elapsedSeconds": round(time.perf_counter() - started, 2),
    })
    return out


def op_build_tree(params):
    """Build one tree with real geometry leaves: ONE object with two material slots (0 bark, 1 leaves), its pivot at
    the trunk's base, Z up, in meters at its real size. The same seed and parameters give the same tree on a given
    Blender.

    params:
      name (str)            default "Tree"
      preset (str)          broadleafRound (default), broadleafTall, conifer, youngTree, shrub, pyramidal, vase,
                            oval - tree forms
      seed (int)            0 or more, default 1
      height (m)            0.5 to 60; the preset's by default
      trunkRadius (m)       the base radius before the root flare; default the preset's share of the height
      levels (int)          branch levels below the trunk, 1 to 3
      crownShape (str)      conical, spherical, hemispherical, cylindrical, taperedCylindrical, flame, inverseConical,
                            tendFlame - the envelope the first-level branch lengths follow
      leafShape (str)       ovate, lanceolate, round (6 triangles), needle (a bundle of three, 6 triangles),
                            palmate (26), lobed (32), cordate (12), pinnate (a compound leaf of 14 leaflets, 58)
      leafSize (cm)         the leaf's length, 0.5 to 60; each leaf varies +-25%
      leafCount (int)       0 (a bare tree) to 400000
      leafColor, leafColor2 ([r,g,b] linear 0-1)   the two greens vertex color R picks between, per leaf
      barkColor ([r,g,b] linear 0-1)
      leafClusters (bool)   false (default): every leaf placed singly. true: twig-and-leaf clusters of about 80
                            triangles - a twig (6 triangles) carrying as many leaves of leafShape as fit, placed along
                            the branches in place of single leaves; leafCount stays the number of leaves, so the tree
                            has leafCount / leavesPerCluster clusters (rounded up)
      leavesPerCluster (int) 1 to 16, only with leafClusters; default the number that fits about 80 triangles
      collision (bool)      a UCX_<name>_00 convex hull round the trunk below the crown; default on, off for shrub
      location, rotation ([x,y,z] or {x,y,z}), collection
      output (str)          geometry (default): real geometry leaves, as above. gameReady: a tree inside a game
                            triangle budget - leaf cards (quads) carrying a cluster of real leaves baked by
                            bake_texture_set to base color, normal and alpha PNGs, a lighter bark (no twigs, fewer
                            sides and rings, the shortest branches dropped until the bark fits 45% of the budget), and
                            LODs from build_lod_chain. The cards stand where the clusters of the same tree would.
      textureFolder (str)   gameReady only, required there: where the card textures are written,
                            T_<name>_LeafCard_BC / _N / _A.png
      textureSize (int)     gameReady only: the card texture's width, 256 to 4096, a power of two, default 1024 (its
                            height follows the card's shape)
      lods (list)           gameReady only: build_lod_chain's ratios, default [0.5, 0.25]; [] for none
      triangleBudget (int)  gameReady only: the triangles LOD0 may have; default the top of the usual range for the
                            tree's size class, the number mesh_quality gameReady holds it to when given none
      stump (true or dict)  (2026-11-25) a stump instead of a tree: {height (the cut, m), radius, flare (base radius
                            over the trunk's, default 1.35), flareHeight, sink (the ground cap below z 0), segments,
                            roots {count (0-16, default 5), spread (alias divePoint: where a root dives into the ground,
                            from the axis), archHeight, thickness, seed}, cut {kind SAW (angle) | NOTCH (direction,
                            notchStep, hinge, hingeHeight: a felling notch with its hinge) | BROKEN (breakHeight)}} -
                            ONE closed manifold mesh, buttress roots as lobes of the trunk's rings, slot 0 bark, slot 1
                            wood on the cut; refused with any key that shapes a whole tree, and with output gameReady
                            or its keys (a stump has no leaves to card; mesh_quality and build_lod_chain take it)
    """
    reject_unknown(params, BUILD_TREE_PARAMS, "build_tree", _NOTES)
    started = time.perf_counter()
    if params.get("stump") not in (None, False):
        return _build_stump(params, started)
    # EVERY VALUE IS READ AND CHECKED BEFORE ANYTHING IS CREATED
    preset_name = take(params, "preset", default="broadleafRound", kind=str)
    if preset_name not in _PRESETS:
        raise MifOpError("preset must be one of %s, got '%s'. NOTHING was created."
                         % (", ".join(sorted(_PRESETS)), preset_name))
    cfg = dict(_PRESETS[preset_name])
    name = take(params, "name", default="Tree", kind=str)
    if not name.strip():
        raise MifOpError("name cannot be empty. NOTHING was created.")
    if any(ord(ch) < 32 for ch in name):
        raise MifOpError("'name' contains a control character, which Blender's names cannot hold - got %r. NOTHING "
                         "was created." % name)
    seed = take_int(params, "seed", default=1)
    if seed < 0:
        raise MifOpError("seed must be 0 or more, got %d. NOTHING was created." % seed)
    height = take_float(params, "height", default=cfg["height"])
    if not 0.5 <= height <= 60.0:
        raise MifOpError("height is %g m; a tree here is 0.5 to 60 m. NOTHING was created." % height)
    radius = take_float(params, "trunkRadius", default=None)
    if radius is None:
        radius = height * cfg["radiusRatio"]
    elif not 0.005 <= radius <= height * 0.25:
        raise MifOpError("trunkRadius is %g m; it must be 0.005 m to a quarter of the height (%g m). NOTHING was "
                         "created." % (radius, height * 0.25))
    levels = take_int(params, "levels", default=cfg["levels"])
    if not 1 <= levels <= 3:
        raise MifOpError("levels is %d; it must be 1, 2 or 3 (branch levels below the trunk). NOTHING was created."
                         % levels)
    cfg["levels"] = levels
    crown = take(params, "crownShape", default=cfg["crown"], kind=str)
    if crown not in _CROWNS:
        raise MifOpError("crownShape must be one of %s, got '%s'. NOTHING was created." % (", ".join(_CROWNS), crown))
    cfg["crown"] = crown
    leaf_shape = take(params, "leafShape", default=cfg["leafShape"], kind=str)
    if leaf_shape not in _LEAF_SHAPES:
        raise MifOpError("leafShape must be one of %s, got '%s'. NOTHING was created."
                         % (", ".join(_LEAF_SHAPES), leaf_shape))
    leaf_size = take_float(params, "leafSize", default=cfg["leafSize"])
    if not 0.5 <= leaf_size <= 60.0:
        raise MifOpError("leafSize is %g cm; it must be 0.5 to 60 cm. NOTHING was created." % leaf_size)
    leaf_count = take_int(params, "leafCount", default=cfg["leafCount"])
    if not 0 <= leaf_count <= _MAX_LEAVES:
        raise MifOpError("leafCount is %d; it must be 0 to %d. NOTHING was created." % (leaf_count, _MAX_LEAVES))
    clusters = take_bool(params, "leafClusters", default=False)
    per_cluster = take_int(params, "leavesPerCluster", default=None)
    if per_cluster is not None and not clusters and params.get("output") != "gameReady":
        raise MifOpError("leavesPerCluster is %d but leafClusters is not true - it sets how many leaves each cluster "
                         "carries, so pass leafClusters true with it, or leave it out. NOTHING was created." % per_cluster)
    if per_cluster is not None and not 1 <= per_cluster <= 16:
        raise MifOpError("leavesPerCluster is %d; it must be 1 to 16. NOTHING was created." % per_cluster)
    output = take(params, "output", default="geometry", kind=str)
    if output not in _GAME_OUTPUTS:
        raise MifOpError("output must be one of %s, got '%s'. NOTHING was created." % (", ".join(_GAME_OUTPUTS), output))
    game = output == "gameReady"
    game_only = [k for k in ("textureFolder", "textureSize", "lods", "triangleBudget") if params.get(k) is not None]
    if game_only and not game:
        raise MifOpError("%s %s output gameReady's; pass output gameReady with %s, or leave %s out. NOTHING was created."
                         % (", ".join(game_only), "is" if len(game_only) == 1 else "are",
                            "it" if len(game_only) == 1 else "them", "it" if len(game_only) == 1 else "them"))
    tex_folder, tex_size, lods, budget_given = None, 1024, [], None
    if game:
        if params.get("leafClusters") is not None:
            raise MifOpError("output gameReady always bakes leaf clusters onto its cards, so leafClusters has nothing to "
                             "choose there (leavesPerCluster still sets each cluster's leaves). NOTHING was created.")
        from .ops_gamebake import _folder                                   # noqa: PLC0415
        if params.get("textureFolder") is None:
            raise MifOpError("output gameReady writes its card textures to textureFolder - pass a folder. NOTHING was "
                             "created.")
        tex_folder = _folder({"folder": take(params, "textureFolder", kind=str)}, "NOTHING was created.")
        tex_size = take_int(params, "textureSize", default=1024)
        if tex_size not in (256, 512, 1024, 2048, 4096):
            raise MifOpError("textureSize is %d; it is the card texture's width, a power of two from 256 to 4096. "
                             "NOTHING was created." % tex_size)
        lods = game_lods(params, _GAME_DEFAULT_LODS)
        budget_given = take_int(params, "triangleBudget", default=None)
        if budget_given is not None and budget_given < 200:
            raise MifOpError("triangleBudget is %d; a game-ready tree needs at least 200 triangles. NOTHING was "
                             "created." % budget_given)
        if len(name) > 54:
            raise MifOpError("name is %d characters; output gameReady makes <name>_LODGroup, so it can be at most 54 "
                             "(Blender keeps 63). NOTHING was created." % len(name))
        if bpy.data.materials.get(name + "_LeafCards") is not None:
            raise MifOpError("a material named '%s' already exists; the card material takes that name - pick another "
                             "name for the tree. NOTHING was created." % (name + "_LeafCards"))
        clusters = True
    leaf_color = _color3(params, "leafColor", cfg["leafColor"])
    leaf_color2 = _color3(params, "leafColor2", cfg["leafColor2"])
    bark_color = _color3(params, "barkColor", cfg["barkColor"])
    want_hull = take_bool(params, "collision", default=cfg["collision"])
    if want_hull and len(name) > 56:
        raise MifOpError("name is %d characters; with collision on it can be at most 56, so that UCX_<name>_00 fits the "
                         "63 characters Blender 3.6-4.4 keep in a name - a truncated hull name attaches to nothing. "
                         "Pass a shorter name, or collision false. NOTHING was created." % len(name))
    try:
        placed = _place_values(params)
    except MifOpError as exc:
        raise MifOpError("%s. NOTHING was created." % str(exc).rstrip("."))
    coll = _resolve_collection(params)
    if np is None:
        raise MifOpError("this Blender's Python has no numpy, which build_tree builds with. NOTHING was created.")

    # ---- the branching ----------------------------------------------------------------------------------------
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    # the longest stem each level can grow, which scales how many children a shorter parent carries
    cfg["_maxlen"] = (height, height * cfg["lengths"][1], height * cfg["lengths"][1] * cfg["lengths"][2])
    cfg["_floor"] = 0.03 * height
    stems, root_depth = _build_stems(cfg, height, radius, rng)
    # THE HEIGHT ASKED FOR IS THE WHOLE TREE'S, not the trunk's: the top branches arch above the trunk's tip - by 14%
    # on the default tree, 13.67 m for 12 asked, the first time this ran. So the skeleton is scaled until its highest
    # point, plus the lean of a leaf above it, sits at `height`. Radii are left alone: trunkRadius is what was asked.
    top = max(float(np.max(np.asarray(s.points)[:, 2])) for s in stems)
    if clusters and per_cluster is None:
        per_cluster = _cluster_per(leaf_shape)
    reach = _CLUSTER_SPAN * 0.8 if clusters else 0.6       # how far a leaf, or a cluster, stands above its stem
    fit = (height - reach * leaf_size / 100.0 * (leaf_count > 0)) / top
    if fit < 1.0:
        # SCALED COPIES, the list rebound: nothing built so far is written to, so a refusal below still holds
        stems = [_Stem(s.level, [p * fit for p in s.points], s.radii, s.length * fit, s.phase, s.weight0, s.weight1)
                 for s in stems]
        root_depth *= fit
    twig_level = max(s.level for s in stems)

    # ---- bark: one closed-tip tube per stem -------------------------------------------------------------------
    flare = None if cfg["baseSplits"] else {"height": 0.06 * height, "amount": 0.9, "lobes": rng.choice((4, 5, 6)),
                                            "offset": rng.uniform(0.0, math.pi)}
    v_parts, q_parts, quv_parts, qcol_parts, t_parts, tuv_parts, tcol_parts = [], [], [], [], [], [], []
    trunk_parts = []
    base = 0
    built, budget, budget_from, dropped = stems, None, None, 0
    if game:
        # THE BUDGET COMES FIRST: it splits between the bark and the cards. Judged on the whole skeleton's bounds; the
        # finished tree's bounds are judged again below, and the cards cut if its size class gives fewer triangles
        all_pts = np.concatenate([np.asarray(s.points, dtype=float) for s in stems])
        budget, budget_from = _budget_for(all_pts, budget_given)
        built, dropped = _game_stems(stems, twig_level, _GAME_BARK_SHARE * budget)
    for s in built:
        pts, radii = (_thin_points(s) if game else
                      (np.asarray(s.points, dtype=float), np.asarray(s.radii, dtype=float)))
        # THE FLARE AND THE HULL BELONG TO THE STEM THAT MEETS THE GROUND: stems[0], or every stem of a shrub
        grounded = s is stems[0] or (cfg["baseSplits"] and s.level == 0)
        s_flare = flare if s is stems[0] else None
        if s_flare is not None:
            pts, radii = _with_base_rings(pts, radii, (s_flare["height"] + root_depth) * 1.3, 4 if game else 8)
        sides = _game_sides(s) if game else _sides(s)
        v, q, quv, qcol, t, tuv, tcol = _tube(pts, radii, sides, s_flare, root_depth, s.weight0, s.weight1, s.phase)
        v_parts.append(v)
        q_parts.append(q + base)
        quv_parts.append(quv)
        qcol_parts.append(qcol)
        t_parts.append(t + base)
        tuv_parts.append(tuv)
        tcol_parts.append(tcol)
        if grounded:
            trunk_parts.append((v, sides))
        base += len(v)
    bark_v = np.concatenate(v_parts)
    quads, caps = np.concatenate(q_parts), np.concatenate(t_parts)

    # ---- leaves -----------------------------------------------------------------------------------------------
    bark_tris = 2 * len(quads) + len(caps)
    game_cards = None
    if game:
        # AS MANY CLUSTERS AS THE DENSE TREE IN CLUSTERS WOULD CARRY, k to a card, k as small as the budget allows
        units = int(math.ceil(_GAME_CARD_COVER * -(-leaf_count // per_cluster)))
        start_rng = copy.deepcopy(np_rng)       # each attempt places its cards from the same draw
        for _attempt in range(2):
            room = max(0, budget - bark_tris) // _GAME_CARD_TRIS
            if units and not room:
                raise MifOpError("the bark alone takes %d of the %d-triangle budget, leaving no room for a single leaf "
                                 "card - raise triangleBudget, or lower levels. NOTHING was created."
                                 % (bark_tris, budget))
            per_card = max(1, -(-units // room)) if units else 1
            n_cards = -(-units // per_card) if units else 0
            spray_v, spray_t, spray_part, span_k = _spray_template(leaf_shape, per_cluster, per_card)
            card, aspect, depth = _card_template(spray_v)
            unit_m = span_k * _CLUSTER_SPAN * leaf_size / 100.0
            leaf_v, leaf_t, leaf_uv, leaf_col, n_units = _leaves(_bearers(stems, twig_level), n_cards, card, unit_m,
                                                                 copy.deepcopy(start_rng))
            final, final_from = _budget_for(np.concatenate([bark_v, leaf_v]), budget_given)
            if final >= budget or bark_tris + len(leaf_t) <= final:
                break
            budget, budget_from = final, final_from
        n_leaves = n_units * per_card * per_cluster
        game_cards = {"count": int(n_units), "clustersPerCard": per_card, "leavesPerCard": per_card * per_cluster,
                      "trianglesPerCard": _GAME_CARD_TRIS, "spanM": round(unit_m, 3),
                      "widthM": round(unit_m * float(card[0][2][1] - card[0][0][1]), 3)}
    else:
        if clusters:
            template = _cluster_template(leaf_shape, per_cluster)
            units, unit_m = -(-leaf_count // per_cluster), _CLUSTER_SPAN * leaf_size / 100.0
        else:
            template, units, unit_m = _leaf_template(leaf_shape), leaf_count, leaf_size / 100.0
        leaf_v, leaf_t, leaf_uv, leaf_col, n_units = _leaves(_bearers(stems, twig_level), units, template, unit_m,
                                                             np_rng)
        n_leaves = n_units * per_cluster if clusters else n_units
    baked = None
    if game and n_units:
        # BAKED BEFORE THE TREE IS MADE: a bake that fails leaves no half-built tree behind
        baked = _bake_leaf_cards(name, (spray_v, spray_t, spray_part), card, depth, unit_m, leaf_color, leaf_color2,
                                 tex_folder, tex_size, aspect)
    verts = np.concatenate([bark_v, leaf_v])
    leaf_t = leaf_t + len(bark_v)

    # ---- the mesh: faces in the order quads, caps, leaves - and every per-corner array in that same order ----
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts.tolist(), [], quads.tolist() + caps.tolist() + leaf_t.tolist())
    loop_uv = np.concatenate([np.concatenate(quv_parts).reshape(-1, 2), np.concatenate(tuv_parts).reshape(-1, 2),
                              leaf_uv.reshape(-1, 2)])
    loop_col = np.concatenate([np.concatenate(qcol_parts).reshape(-1, 4), np.concatenate(tcol_parts).reshape(-1, 4),
                               leaf_col.reshape(-1, 4)])
    if len(mesh.loops) != len(loop_uv):
        bpy.data.meshes.remove(mesh)
        raise MifOpError("build_tree made %d face corners and computed UVs for %d - a defect in this op, not in the "
                         "request. NOTHING was created." % (len(mesh.loops), len(loop_uv)))
    uv_layer = mesh.uv_layers.new(name="UVMap")
    uv_layer.data.foreach_set("uv", loop_uv.astype(np.float32).ravel())
    col = mesh.color_attributes.new(name="Col", type="BYTE_COLOR", domain="CORNER")
    # WRITTEN THROUGH color_srgb, NOT color: the numbers are DATA for an engine, and this is the property the FBX
    # exporter writes with colors_type SRGB, which export_mesh pins (export_fbx_bin.py 3.6:1251 ... 5.2:1307). So each
    # byte is value x 255, the FBX carries the value, and Unreal's VertexColor node reads it unchanged - the importer
    # quantizes the FBX number with no color-space change and the build re-encodes what it decoded
    # (FbxStaticMeshImport.cpp 5.3:944-952 and 5.8:943-952, StaticMeshBuilder.cpp 5.3:998 and 5.8:1706-1708, the same
    # on 5.4-5.7; LocalVertexFactory.ush fetches the bytes with no decode). Written through `color` (linear),
    # 0.5 was stored as byte 188 and reached Unreal as 0.737, measured on a bare tree 2026-09-27.
    col.data.foreach_set("color_srgb", loop_col.astype(np.float32).ravel())
    n_bark_faces = len(quads) + len(caps)
    n_faces = n_bark_faces + len(leaf_t)
    mesh.polygons.foreach_set("material_index", np.concatenate([np.zeros(n_bark_faces, np.int32),
                                                                 np.ones(len(leaf_t), np.int32)]))
    mesh.polygons.foreach_set("use_smooth", np.ones(n_faces, dtype=bool))
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    _link_new(obj, coll)
    _place(obj, placed)
    bark_m, leaf_m = _materials(obj.name, bark_color, leaf_color, leaf_color2)
    mesh.materials.append(bark_m)
    if baked is not None:
        # THE CARDS' SLOT HOLDS THE BAKED MATERIAL: images a game export carries, where the dense leaves' material is
        # a node chain that reaches an engine as nothing
        bpy.data.materials.remove(leaf_m)
        leaf_m = baked[0]
    mesh.materials.append(leaf_m)

    hull, hull_note = None, None
    if want_hull:
        cut = min(max(1.0, cfg["baseSize"] * height), 0.35 * height)
        hull, hull_note = _trunk_hull(obj, coll, trunk_parts, cut, radius)
    lod_reply = None
    if game and lods:
        from .ops_kit import op_build_lod_chain                             # noqa: PLC0415
        try:
            lod_reply = op_build_lod_chain({"object": obj.name, "ratios": lods})
        except MifOpError as exc:
            hull_obj = bpy.data.objects.get(hull["object"]) if hull else None
            for o in (obj, hull_obj):
                if o is not None:
                    me = o.data
                    remove_objects([o])
                    bpy.data.meshes.remove(me)
            for m in (bark_m, leaf_m):
                bpy.data.materials.remove(m)
            raise MifOpError("build_lod_chain refused the tree's LODs (%s). The tree and its hull were removed; the card "
                             "textures stay in %s. NOTHING was created." % (str(exc).rstrip("."), tex_folder)) from None

    out = _created(obj, coll, "MESH", "build_tree", name)
    radial = np.hypot(verts[:, 0], verts[:, 1])
    counts = {"level%d" % lv: sum(1 for s in stems if s.level == lv) for lv in range(twig_level + 1)}
    tris_per_leaf = len(_leaf_template(leaf_shape)[1])
    bark_tris = 2 * len(quads) + len(caps)
    out.update({
        "preset": preset_name, "seed": seed,
        "heightM": round(float(verts[:, 2].max()), 3),
        "rootDepthM": round(float(-verts[:, 2].min()), 3),
        "trunkRadiusM": round(float(radius), 4),
        "crownWidthM": round(float(2.0 * np.percentile(radial, 99)), 3),
        "levels": levels, "crownShape": crown, "stems": dict(counts, total=len(stems)),
        "leafShape": leaf_shape, "leafSizeCm": round(float(leaf_size), 2), "leaves": int(n_leaves),
        "trianglesPerLeaf": tris_per_leaf,
        "clusters": ({"count": int(n_units), "leavesPerCluster": per_cluster,
                      "trianglesPerCluster": len(template[1]), "twigTriangles": _CLUSTER_TWIG_TRIS,
                      "spanM": round(unit_m, 3)} if clusters and not game else None),
        "vertices": len(mesh.vertices), "faces": len(mesh.polygons),
        "triangles": {"bark": bark_tris, "leaves": len(leaf_t), "total": bark_tris + len(leaf_t)},
        "materialSlots": [{"index": 0, "part": "bark", "material": bark_m.name},
                          {"index": 1, "part": "leaves", "material": leaf_m.name}],
        "uvMap": {"name": "UVMap",
                  "bark": "tiles along each stem at %.2f m per unit, round the girth and along it alike" % _BARK_TILE_M,
                  "leaves": "each leaf spans the whole 0-1 square, stalk at v 0, tip at v 1"
                            + ("; each cluster's twig maps a sliver at u 0.49-0.51, v 0-0.016" if clusters else "")},
        "colorAttribute": {"name": "Col", "domain": "CORNER", "type": "BYTE_COLOR",
                           "stored": "each byte is value x 255 (sRGB space), so export_mesh's FBX carries the values "
                                     "and Unreal's VertexColor node reads them as written; Blender's shading and "
                                     "mesh_stats read them linearized",
                           "R": "random per leaf, for color variation; 0.5 on bark",
                           "G": "wind weight: 0 at the trunk's base, rising along every branch, 1 at each leaf's tip",
                           "B": "phase, shared by a main branch and everything on it",
                           "A": "1"},
        "collisionHull": hull, "collisionNote": hull_note,
        "pivot": "the trunk's base, at the object's origin, Z up; the root flare reaches rootDepthM below it",
        "elapsedSeconds": round(time.perf_counter() - started, 2),
    })
    out["output"] = output
    if game:
        total = bark_tris + len(leaf_t)
        out["uvMap"]["leaves"] = "each card spans the whole 0-1 square, u along the cluster from its base"
        out["gameReady"] = {
            "triangleBudget": budget, "budget": budget_from, "triangles": total, "withinBudget": total <= budget,
            "bark": {"triangles": bark_tris, "stemsBuilt": len(built), "stemsLeftOut": dropped,
                     "rule": "no twigs (the cards carry them), %d%% of the dense sides and half the rings on each "
                             "branch, then the shortest of the deepest level left out until the bark fits %d%% of the "
                             "budget" % (50, round(100 * _GAME_BARK_SHARE))},
            "cards": game_cards,
            "material": leaf_m.name if baked is not None else None,
            "textures": ({"files": baked[1], "size": list(baked[2]), "folder": tex_folder} if baked is not None
                         else None),
            "lods": (lod_reply or {}).get("lods"), "lodGroup": (lod_reply or {}).get("group"),
            "check": {"call": "mesh_quality", "args": {"object": obj.name, "gameReady": (
                {"triangleBudget": budget} if budget_given is not None else True)}},
        }
        out["elapsedSeconds"] = round(time.perf_counter() - started, 2)
    return out


OPS = {
    "build_tree": op_build_tree,
}
