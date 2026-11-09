"""Rocks and ground cover - build_rock and build_plant.

WHAT WAS MISSING. MifBlender could make a tree (build_tree) and not the rock a natural scene is dressed with. A
displaced sphere is the usual shortcut and reads as a blob; this module builds the forms from what makes real ones
read as real.

ROCKS ARE A SIGNED-DISTANCE FIELD, MESHED. A rock is the set of points where a field is negative, and the field is
composed from what a photographed rock shows: FRACTURE PLANES of varied size, a few of them large, meeting at edges
that are sharp in one place and worn in the next - each plane BENT by a smooth domain warp, so faces curve and edges
wander (flat planes and straight edges read as a cut gem); CHIPS, many small planes each shaving a facet off whatever
corner sticks out in its direction; NOTCHES where a block broke out of an edge; on slabs and cliffs STRATA - layers of
their own thickness and set-back with bedding planes between them, fine LAMINAE on a slab's rim, and vertical JOINTS
that do not line up from one layer to the next, so a cliff is ledges and blocks. The grain finer than the mesh is left
to the material. Tried and removed, each for reading false at some seed: round scoops and flat-bottomed round scars
(thumbprints, stamps), a V crack across a boulder (a mouth), an equal grid of joints (a brick wall), large facet planes
across a cliff face (they cut every block back to one plane).

The field is sampled on a grid in a narrow band round the surface and meshed by surface nets, the vertices are
stepped back onto the field, and the mesh is reduced to the triangle budget by Blender's own collapse decimation, which
keeps triangles on the edges and ledges and spends few on flat faces.

GROUND COVER IS REAL GEOMETRY (build_plant): a grass tuft is tapered, creased blades leaning out more the further
from the center they rise; a fern is arching fronds with paired pinnae, longest a third of the way up; a broadleaf plant
is a rosette of stalked leaves, inner ones standing and outer ones lying out; a bush is stems from the ground forking
twice, leaved on their outer two thirds. No alpha cards - Nanite carries the triangles - and no collision. The materials
and the vertex color are build_tree's (R per blade or leaf, G wind weight, B phase), so a material written for a
tree's leaves - its wind on G - reads a plant the same way.

PURE PYTHON AND NUMPY. Geometry nodes move between Blender 3.6 and 5.2; mesh.from_pydata, foreach_set, a color
attribute on the CORNER domain, the Decimate modifier and the shader nodes used here do not.
"""
import math
import random
import time

import bmesh
import bpy

try:
    import numpy as np
except ImportError:                     # pragma: no cover - every Blender 3.6-5.2 bundles numpy
    # The offline suites import every ops module with bpy stubbed, in a Python that may have no numpy - so a missing
    # numpy costs these ops, not the import (ops_tree and ops_audio do the same).
    np = None

from .ops_common import remove_objects, MifOpError, finite_float, reject_unknown, take, take_bool, take_float, take_int
from .ops_create import _created, _link_new, _place, _place_values, _resolve_collection
from .ops_mesh import op_create_collision_hull, op_uv_unwrap, uv_degeneracy
from .ops_tree import _materials as _tree_materials, _outline_leaf, _tube

# MODULE-LEVEL AND PLAIN LITERALS: parity_check.py resolves an accepted-key expression statically.
BUILD_ROCK_PARAMS = (
    "name", "kind", "seed", "size", "triangles", "sink", "weathering", "strata", "stoneColor", "stoneColor2",
    "moss", "collision", "location", "rotation", "collection",
)

_ROCK_NOTES = {
    "scale": "sizes are in centimeters: pass size [length, width, height] - the mesh's bounds, sunk part included.",
    "dimensions": "pass size [length, width, height] in centimeters.",
    "detail": "detail is set by triangles, the count the rock is reduced to (Nanite takes the dense end).",
    "subdivisions": "detail is set by triangles, the count the rock is reduced to.",
    "noise": "the surface detail comes from the kind and weathering; there is no free noise amplitude to set.",
    "material": "the rock gets its own procedural stone material; stoneColor, stoneColor2 and moss tune it.",
    "pivot": "the pivot is always the bottom center, with the rock sunk `sink` centimeters below it.",
    "lod": "no LODs are made: in Unreal a rock is meant for Nanite, and generate_lods covers an engine without it.",
    "uv": "two UV layers are always made: UVMap tiles at one meter per unit, UVLightmap is 0-1 without overlaps.",
    "color": "pass stoneColor (and stoneColor2) as [r, g, b], linear 0-1.",
}

_ROCK_KINDS = ("boulder", "slab", "cliff")
# Per kind: the default size in cm [length, width, height], the triangle budget, the sink as a share of the height,
# and whether strata show. The budgets sit in the large-mesh range a Nanite rock is built at; pass triangles for less.
_ROCK_DEFAULTS = {
    "boulder": {"size": (140.0, 110.0, 85.0), "triangles": 30000, "sink": 0.06, "strata": False,
                "weathering": 0.2},
    "slab": {"size": (240.0, 170.0, 55.0), "triangles": 30000, "sink": 0.08, "strata": True, "weathering": 0.2},
    "cliff": {"size": (800.0, 300.0, 600.0), "triangles": 120000, "sink": 0.03, "strata": True, "weathering": 0.25},
}
_MIN_TRIANGLES, _MAX_TRIANGLES = 500, 1000000
_UV_TILE_M = 1.0
_CHUNK = 400000

# ---------------------------------------------------------------------------------------------------------------------
# noise
# ---------------------------------------------------------------------------------------------------------------------

_GRADS = (np.array([(1, 1, 0), (-1, 1, 0), (1, -1, 0), (-1, -1, 0), (1, 0, 1), (-1, 0, 1), (1, 0, -1), (-1, 0, -1),
                    (0, 1, 1), (0, -1, 1), (0, 1, -1), (0, -1, -1)], dtype=float) if np is not None else None)
_CORNERS = [(dx, dy, dz) for dz in (0, 1) for dy in (0, 1) for dx in (0, 1)]


def _hash(ix, iy, iz, seed):
    """A 32-bit integer hash of lattice points, in int64 so nothing overflows on the way."""
    h = (ix * 0x8DA6B343) ^ (iy * 0xD8163841) ^ (iz * 0xCB1AB31F) ^ (seed * 0x165667B1)
    h &= 0xFFFFFFFF
    h ^= h >> 15
    h = (h * 0x2C1B3C6D) & 0xFFFFFFFF
    h ^= h >> 12
    h = (h * 0x297A2D39) & 0xFFFFFFFF
    h ^= h >> 15
    return h


def _perlin(p, seed):
    """Gradient noise at points p (N, 3), about -1 to 1: twelve edge gradients on the integer lattice, quintic fade."""
    pf = np.floor(p)
    i = pf.astype(np.int64)
    f = p - pf
    u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0)
    vals = []
    for dx, dy, dz in _CORNERS:
        g = _GRADS[_hash(i[:, 0] + dx, i[:, 1] + dy, i[:, 2] + dz, seed) % 12]
        vals.append(g[:, 0] * (f[:, 0] - dx) + g[:, 1] * (f[:, 1] - dy) + g[:, 2] * (f[:, 2] - dz))
    x00 = vals[0] + u[:, 0] * (vals[1] - vals[0])
    x10 = vals[2] + u[:, 0] * (vals[3] - vals[2])
    x01 = vals[4] + u[:, 0] * (vals[5] - vals[4])
    x11 = vals[6] + u[:, 0] * (vals[7] - vals[6])
    y0 = x00 + u[:, 1] * (x10 - x00)
    y1 = x01 + u[:, 1] * (x11 - x01)
    return y0 + u[:, 2] * (y1 - y0)


def _fbm(p, seed, octaves, gain=0.5):
    """Fractal sum of `octaves` noises, each twice the frequency of the last, normalized to about -1 to 1. Each octave
    is shifted by an irrational offset so the lattices do not line up."""
    total, amp, norm = 0.0, 1.0, 0.0
    q = p
    for o in range(octaves):
        total = total + amp * _perlin(q + (o * 17.31, o * 5.87, o * 11.13), seed + 101 * o)
        norm += amp
        amp *= gain
        q = q * 2.03
    return total / max(norm, 1e-9) * 1.6


def _facets(p, seed):
    """Angular fracture cells at points p (N, 3), in cell units: the nearest of the jittered seeds in the eight lattice
    cells around each point. Returns (offset (N, 3) from that seed, hash of its cell). Each cell's own plane, read off
    those two, breaks a surface into flat faces that step at the borders, the way broken rock does - no noise sum does
    that, since it has no edges."""
    b = np.floor(p - 0.5).astype(np.int64)
    best = np.full(len(p), 1e18)
    off = np.empty_like(p)
    cell = np.empty(len(p), dtype=np.int64)
    for dx, dy, dz in _CORNERS:
        ix, iy, iz = b[:, 0] + dx, b[:, 1] + dy, b[:, 2] + dz
        h = _hash(ix, iy, iz, seed)
        jit = np.stack((h & 1023, (h >> 10) & 1023, (h >> 20) & 1023), axis=1) / 1023.0
        d = p - (np.stack((ix, iy, iz), axis=1) + 0.15 + 0.7 * jit)
        dd = np.einsum("ij,ij->i", d, d)
        m = dd < best
        best[m] = dd[m]
        off[m] = d[m]
        cell[m] = h[m]
    return off, cell


def _smax(a, b, k):
    """Polynomial smooth maximum: max(a, b) with the crease rounded over a width k (k 0 is the plain max)."""
    if np.isscalar(k) and k <= 0.0:
        return np.maximum(a, b)
    k = np.maximum(k, 1e-9)
    h = np.maximum(k - np.abs(a - b), 0.0) / k
    return np.maximum(a, b) + h * h * k * 0.25


def _smin(a, b, k):
    return -_smax(-a, -b, k)


def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _unit_rows(v):
    return v / np.maximum(1e-12, np.linalg.norm(v, axis=-1, keepdims=True))


def _random_unit(rng):
    z = rng.uniform(-1.0, 1.0)
    a = rng.uniform(0.0, 2.0 * math.pi)
    r = math.sqrt(max(0.0, 1.0 - z * z))
    return np.array((r * math.cos(a), r * math.sin(a), z))


# ---------------------------------------------------------------------------------------------------------------------
# the rock field
# ---------------------------------------------------------------------------------------------------------------------

class _RockField:
    """A rock's signed distance field (negative inside), in meters, centered near the origin. Built once from a seed;
    eval(P) takes points (N, 3). Every feature is drawn up front from the seed, so evaluation is pure numpy.

    THE FIELD IS THE MACRO FORM, THEN CHIPS, THEN CARVING. The macro form is the kind's own (fracture planes; layers;
    blocks) plus a low undulation. CHIPS are many small planes, each set just inside the form's extreme point in its
    own direction, so each shaves a small flat facet off whatever corner or edge sticks out there - the way a broken
    rock's edges are faceted. An isotropic noise on top was tried first and read as dough, not stone. Carving is the
    concave work: notches and cracks."""

    def __init__(self, kind, size_m, weathering, strata, seed, cell):
        self.kind = kind
        self.L, self.W, self.H = size_m
        self.s = (self.L * self.W * self.H) ** (1.0 / 3.0)
        self.weathering = weathering
        self.strata = strata
        self.seed = seed
        self.cell = cell
        rng = random.Random(seed)
        self.rng = rng
        self.planes = []          # (normal, offset, rounding) - intersected
        self.chips = []           # (normal, offset, rounding)
        self.notches = []         # (n1, d1, n2, d2, rounding)
        self.cracks = []          # (normal, offset, width, depth, seed)
        self.layers = None
        self.lamina = None
        self.warp = (0.0, 1.0)
        self.fractures = []       # (cell size per axis, amplitude, slope, seed)
        getattr(self, "_setup_" + kind)()
        self._setup_chips()

    # ---- shared pieces ----------------------------------------------------------------------------------------------
    def _round_k(self, base):
        """The edge rounding for a plane: weathering scales it; a random spread per plane makes one edge crisp and
        the next worn, as broken and weathered faces meet on one rock."""
        return base * (0.2 + 1.6 * self.weathering) * self.rng.uniform(0.3, 1.5)

    def _add_cracks(self, count, normal_fn, width, depth):
        for _ in range(count):
            n = normal_fn()
            n = n / np.linalg.norm(n)
            off = float(np.dot(n, (self.rng.uniform(-0.25, 0.25) * self.L, self.rng.uniform(-0.25, 0.25) * self.W,
                                   self.rng.uniform(-0.2, 0.2) * self.H)))
            self.cracks.append((n, off, max(width * self.rng.uniform(0.7, 1.3), 3.5 * self.cell),
                                depth * self.rng.uniform(0.6, 1.3), self.rng.randrange(1 << 20)))

    def _setup_laminae(self, thickness):
        """LAMINAE: the fine layering inside a stratum, drawn as shallow grooves at irregular spacing - some deep,
        some missing - so a slab's rim and a cliff's ledges show thin ribs, and a top surface steps where it crosses
        one. Only when the grid can draw it: a groove narrower than about two cells would alias into a sawtooth."""
        self.lamina = None
        if not self.strata:
            return
        width = max(0.22 * thickness, 1.8 * self.cell)
        if width > 0.4 * thickness:
            return
        self.lamina = (thickness, width, 0.55 * width, self.rng.randrange(1 << 20))

    @staticmethod
    def _ellipsoid_support(n, a, b, c):
        return math.sqrt((a * n[0]) ** 2 + (b * n[1]) ** 2 + (c * n[2]) ** 2)

    def _chip_directions(self):
        """Where chips come from, per kind: all round a boulder; round the rim and over the top of a slab; over the
        face and the top of a cliff."""
        rng = self.rng
        out = []
        if self.kind == "boulder":
            for _ in range(rng.randint(60, 100)):
                out.append(_random_unit(rng))
        elif self.kind == "slab":
            for _ in range(rng.randint(30, 50)):
                ang = rng.uniform(0.0, 2.0 * math.pi)
                out.append(np.array((math.cos(ang), math.sin(ang), rng.uniform(-0.35, 1.2))))
        else:
            for _ in range(rng.randint(50, 90)):
                out.append(np.array((rng.uniform(-0.9, 0.9), -1.0, rng.uniform(-0.6, 1.4))))
        return [v / np.linalg.norm(v) for v in out]

    def _warped(self, P):
        """The point moved by a smooth vector noise. The macro form is read at the moved point, so every fracture
        plane bends by its OWN amount (its normal's share of the move): faces curve the way a conchoidal break does,
        and the edges between them wander. Flat planes and straight edges read as a cut gem."""
        amp, freq = self.warp
        if amp <= 0.0:
            return P
        q = P * freq
        return P + amp * np.stack([_fbm(q, self.seed + 61 + i, 2) for i in range(3)], axis=1)

    def _add_fractures(self, rows):
        """rows: (cell size per axis in meters, step amplitude in meters, the face's tilt). Weathering wears them down. A
        scale the grid cannot draw - a cell under about three grid cells - is left out rather than aliased."""
        for size, amp, slope in rows:
            if min(size) >= 3.0 * self.cell:
                self.fractures.append((np.array(size), amp * (1.25 - self.weathering), slope, self.rng.randrange(1 << 20)))

    def _fracture(self, P):
        """How far each fracture cell's own plane moves the surface at P."""
        out = np.zeros(len(P))
        for size, amp, slope, fseed in self.fractures:
            off, h = _facets(P / size, fseed)
            lift = ((h >> 3) & 255) / 255.0 - 0.5
            nx = ((h >> 11) & 255) / 255.0 - 0.5
            ny = ((h >> 19) & 255) / 255.0 - 0.5
            nz = ((h >> 5) & 255) / 255.0 - 0.5
            out += amp * 2.0 * lift + slope * amp * 2.0 * (nx * off[:, 0] + ny * off[:, 1] + nz * off[:, 2])
        return out

    def _surface_cloud(self):
        """Points on the macro form, one per ray from the center, found by bisection."""
        rng = np.random.default_rng(self.seed)
        d = rng.normal(size=(3000, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        d *= np.array((self.L, self.W, self.H)) / max(self.L, self.W, self.H)
        lo, hi = np.zeros(len(d)), np.full(len(d), 1.3 * max(self.L, self.W, self.H) / np.linalg.norm(d, axis=1))
        for _ in range(28):
            m = 0.5 * (lo + hi)
            P = d * m[:, None]
            inside = self._macro(self._warped(P)) < 0.0
            lo = np.where(inside, m, lo)
            hi = np.where(inside, hi, m)
        keep = lo > 1e-6
        return d[keep] * lo[keep][:, None]

    def _setup_chips(self):
        """Each chip plane sits `depth` inside the macro form's extreme point in its direction - read off points on
        the form's surface - so it cuts a small facet there and nowhere else. (Round scars - a sphere, or a plane
        inside a sphere - were tried for spalled flakes and read as thumbprints and stamps at every size; the chips
        and the warp carry that job.)"""
        cloud = self._surface_cloud()
        if len(cloud) < 20:
            return
        rng = self.rng
        for u in self._chip_directions():
            top = float(np.max(cloud @ u))
            # many small chips and a few large ones
            depth = self.s * min(0.04, 0.004 + rng.expovariate(1.0 / 0.01))
            if self.kind == "slab":
                depth = min(depth, 0.35 * self.H)
            self.chips.append((u, top - depth, self.s * 0.006 * (0.2 + self.weathering) * rng.uniform(0.3, 1.5)))

    # ---- boulder ----------------------------------------------------------------------------------------------------
    def _setup_boulder(self):
        rng = self.rng
        a, b, c = self.L * 0.5, self.W * 0.5, self.H * 0.5
        count = rng.randint(11, 16)
        golden = math.pi * (3.0 - math.sqrt(5.0))
        spin = rng.uniform(0.0, 2.0 * math.pi)
        for i in range(count):
            z = 1.0 - 2.0 * (i + 0.5) / count
            r = math.sqrt(max(0.0, 1.0 - z * z))
            ang = spin + golden * i
            n = np.array((r * math.cos(ang), r * math.sin(ang), z))
            n = n + np.array([rng.gauss(0.0, 0.22) for _ in range(3)])
            n /= np.linalg.norm(n)
            d = self._ellipsoid_support(n, a, b, c) * rng.uniform(0.80, 0.97)
            self.planes.append((n, d, self._round_k(self.s * 0.022)))
        # TWO OR THREE DOMINANT FRACTURE FACES cut deeper: a broken boulder shows a few large flat faces, not a ball of
        # equal facets
        for _ in range(rng.randint(2, 3)):
            n = _random_unit(rng)
            n[2] *= 0.6
            n /= np.linalg.norm(n)
            d = self._ellipsoid_support(n, a, b, c) * rng.uniform(0.66, 0.78)
            self.planes.append((n, d, self._round_k(self.s * 0.022)))
        # the base: flatter, so the rock sits rather than balances
        self.planes.append((np.array((0.0, 0.0, -1.0)), c * 0.86, self._round_k(self.s * 0.03)))
        self._stack_planes()
        for _ in range(rng.randint(0, 2)):
            # A NOTCH: a block broken out of an upper edge, a concave step two planes meet in
            u = _random_unit(rng)
            u[2] = abs(u[2]) * 0.4
            u /= np.linalg.norm(u)
            n1 = u
            n2 = np.array((0.0, 0.0, 1.0)) + u * 0.25
            n2 /= np.linalg.norm(n2)
            d1 = self._ellipsoid_support(n1, a, b, c) * rng.uniform(0.5, 0.68)
            d2 = self._ellipsoid_support(n2, a, b, c) * rng.uniform(0.4, 0.62)
            self.notches.append((n1, d1, n2, d2, self.s * 0.008))
        self.undulation = (self.s * 0.008, 2.5 / self.s, 2)
        self.warp = (self.s * 0.05, 1.5 / self.s)
        self._add_fractures((((self.s * 0.3,) * 3, self.s * 0.008, 3.0), ((self.s * 0.11,) * 3, self.s * 0.003, 3.0)))

    def _stack_planes(self):
        self._pn = np.array([p[0] for p in self.planes])
        self._pd = np.array([p[1] for p in self.planes])
        self._pk = [p[2] for p in self.planes]

    def _macro_boulder(self, P):
        V = P @ self._pn.T - self._pd
        f = V[:, 0]
        for i in range(1, V.shape[1]):
            f = _smax(f, V[:, i], self._pk[i])
        return f

    # ---- slab -------------------------------------------------------------------------------------------------------
    def _setup_slab(self):
        rng = self.rng
        a, b = self.L * 0.5, self.W * 0.5
        count = rng.randint(7, 10)
        spin = rng.uniform(0.0, 2.0 * math.pi)
        for i in range(count):
            ang = spin + 2.0 * math.pi * (i + rng.uniform(-0.3, 0.3)) / count
            tilt = math.radians(rng.uniform(-16.0, 16.0))
            n = np.array((math.cos(ang) * math.cos(tilt), math.sin(ang) * math.cos(tilt), math.sin(tilt)))
            # a superellipse outline: squarer than an ellipse, as a slab broken from a bed is
            ca, sa = abs(math.cos(ang)), abs(math.sin(ang))
            d = (a * b) / max(1e-9, ((b * ca) ** 4 + (a * sa) ** 4) ** 0.25) * rng.uniform(0.84, 0.98)
            self.planes.append((n, d, self._round_k(self.H * 0.06)))
        self._stack_planes()
        # STRATA: 2-4 layers of their own thickness, each inset from the one below by its own amount, side by side
        n_layers = rng.randint(2, 4) if self.strata else 1
        while True:
            cuts = sorted(rng.uniform(0.18, 0.85) for _ in range(n_layers - 1))
            if all(y - x > 0.14 for x, y in zip([0.0] + cuts, cuts + [1.0])):
                break
        self.layers = [-0.5 * self.H] + [(-0.5 + t) * self.H for t in cuts] + [0.5 * self.H]
        # per layer, per side: how far that side of that layer is set back
        self.layer_inset = np.array([[0.0] * count] + [[self.W * rng.uniform(0.0, 0.06) + self.W * 0.02 * j
                                                        * rng.random() for _ in range(count)]
                                                       for j in range(1, n_layers)])
        # the top layer is broken off along an irregular line across part of the slab
        cut_n = _random_unit(rng)
        cut_n[2] = 0.0
        cut_n /= np.linalg.norm(cut_n)
        self.top_cut = (cut_n, rng.uniform(0.0, 0.45) * min(a, b)) if n_layers > 1 and rng.random() < 0.7 else None
        self.tilt = (rng.uniform(-0.04, 0.04), rng.uniform(-0.04, 0.04))

        def vertical():
            ang = rng.uniform(0.0, math.pi)
            return np.array((math.cos(ang), math.sin(ang), rng.uniform(-0.1, 0.1)))
        self._add_cracks(rng.randint(1, 2), vertical, self.s * 0.006, self.H * 0.45)
        self.undulation = (self.H * 0.05, 1.2 / self.s, 2)
        self.warp = (self.s * 0.02, 1.4 / self.s)
        self._setup_laminae(self.H * rng.uniform(0.09, 0.16))
        self._add_fractures((((self.s * 0.28, self.s * 0.28, self.H * 0.5), self.H * 0.03, 2.0),
                             ((self.s * 0.09, self.s * 0.09, self.H * 0.2), self.H * 0.012, 2.0)))

    def _slab_warp(self, P):
        return self.H * 0.07 * _fbm(P * (1.3 / self.s), self.seed + 3, 2)

    def _macro_slab(self, P):
        z = P[:, 2]
        warp = self._slab_warp(P)
        zt = z - self.tilt[0] * P[:, 0] - self.tilt[1] * P[:, 1]
        V = P @ self._pn.T - self._pd
        n_layers = len(self.layers) - 1
        f = None
        for j in range(n_layers):
            lo, hi = self.layers[j], self.layers[j + 1]
            lo_w = -1e9 if j == 0 else lo + warp
            hi_w = hi + (warp if j < n_layers - 1 else 0.0)
            side = V[:, 0] + self.layer_inset[j, 0]
            for i in range(1, V.shape[1]):
                side = _smax(side, V[:, i] + self.layer_inset[j, i], self._pk[i])
            layer = _smax(side, np.maximum(zt - hi_w, lo_w - zt), self.H * 0.05 * (0.3 + self.weathering))
            if j == n_layers - 1 and self.top_cut is not None and n_layers > 1:
                cn, cd = self.top_cut
                wig = self.s * 0.05 * _fbm(P * (4.0 / self.s), self.seed + 29, 2)
                layer = _smax(layer, P @ cn - cd + wig, self.H * 0.04)
            f = layer if f is None else np.minimum(f, layer)
        return f

    # ---- cliff ------------------------------------------------------------------------------------------------------
    def _setup_cliff(self):
        """A cliff face is LAYERS of very different thickness - thin ones soft and eroded back, thick ones standing
        out as ledges - cut by vertical JOINTS that are sparse, lean, and do not line up from one layer to the next
        (a new set starts in about half the layers; the rest carry the set below on). Each block between them is set
        back and tilted by its own amount. A grid of equal joints read as a brick wall."""
        rng = self.rng
        L, D, H = self.L, self.W, self.H
        edges = [-0.5 * H]
        while self.strata:
            t = H * math.exp(rng.uniform(math.log(0.03), math.log(0.2)))
            if edges[-1] + t >= 0.47 * H:
                break
            edges.append(edges[-1] + t)
        edges.append(0.5 * H)
        self.layers = edges[1:-1]
        self.cliff_layers = []
        prev = None
        for j in range(len(edges) - 1):
            thick = edges[j + 1] - edges[j]
            hard = min(1.0, thick / (0.12 * H)) * rng.uniform(0.45, 1.0)
            if prev is None or rng.random() < 0.3:
                joints, x = [], -0.5 * L
                while True:
                    x += L * rng.uniform(0.08, 0.42)
                    if x >= 0.46 * L:
                        break
                    joints.append(x)
                lean = rng.uniform(-0.55, 0.55)
            else:
                joints, lean = prev["joints"], prev["lean"]
            n = len(joints) + 1
            base = D * (0.03 + 0.18 * (1.0 - hard) * rng.uniform(0.5, 1.0))
            layer = {
                "joints": np.array(joints), "lean": lean, "zc": 0.5 * (edges[j] + edges[j + 1]),
                "recess": np.array([base + D * 0.1 * rng.random() for _ in range(n)]),
                "tiltX": np.array([rng.uniform(-0.1, 0.1) for _ in range(n)]),
                "tiltZ": np.array([rng.uniform(-0.25, 0.08) for _ in range(n)]),
                "var": np.array([rng.random() for _ in range(n)]),
                "open": np.array([rng.random() < 0.14 for _ in joints], dtype=bool),
                "mids": self._centres(joints, L),
            }
            self.cliff_layers.append(layer)
            prev = layer
        top = self.cliff_layers[-1]
        top["drop"] = np.array([H * rng.uniform(0.0, 0.08) for _ in range(len(top["joints"]) + 1)])
        self.dip = rng.uniform(0.05, 0.16) * rng.choice((-1.0, 1.0))
        # NO FRONT FACET PLANES: large planes across the face cut every block back to one flat plane; the face's
        # large form comes from the warp and the undulation instead. Only the two ends are planes.
        for sgn in (-1.0, 1.0):
            yaw = math.radians(rng.uniform(-25.0, 20.0))
            n = np.array((sgn * math.cos(yaw), math.sin(yaw), rng.uniform(-0.15, 0.15)))
            n /= np.linalg.norm(n)
            self.planes.append((n, 0.5 * L * rng.uniform(0.9, 1.0), 0.04 * D))
        self._stack_planes()
        self.undulation = (D * 0.08, 0.9 / min(L, H), 3)
        self.warp = (min(L, H) * 0.05, 1.3 / min(L, H))
        self.groove = (max(0.018 * min(L, H), 2.5 * self.cell), 0.05 * D + 2.0 * self.cell)
        # BROKEN FACES: tall cells a meter or so across and smaller ones in them, so a block's face is not one sheet with
        # a straight edge. Without them the joints and layers were all the eye had, and they read as coursed masonry
        s = min(L, H)
        self._add_fractures((((s * 0.22, s * 0.22, s * 0.36), D * 0.04, 3.0), ((s * 0.08, s * 0.08, s * 0.12), D * 0.012, 3.0),
                             ((s * 0.03, s * 0.03, s * 0.04), D * 0.003, 3.0)))

    @staticmethod
    def _centres(cuts, span):
        edges = [-0.5 * span] + list(cuts) + [0.5 * span]
        return np.array([(edges[i] + edges[i + 1]) * 0.5 for i in range(len(edges) - 1)])

    @staticmethod
    def _soft_cells(v, cuts, width):
        """Soft membership of v in each interval between `cuts`: rows sum to 1, each boundary eased over `width`."""
        cols, prev = [], np.ones_like(v)
        for c in cuts:
            s = _smoothstep(-width, width, v - c)
            cols.append(prev - s)
            prev = s
        cols.append(prev)
        return np.stack(cols, axis=1)

    def _cliff_blocks(self, P):
        """Per point: the face's set-back (recess and block tilt, eased across block edges), the skyline's drop,
        the distance to the nearest OPEN joint in the point's layer, and the block's variation value."""
        x, z = P[:, 0], P[:, 2]
        wx = x + 0.05 * self.L * _fbm(P * (3.0 / self.H), self.seed + 41, 3)
        wz = z + 0.035 * self.H * _fbm(P * (2.5 / self.L), self.seed + 43, 3) + self.dip * x
        ease = max(0.006 * min(self.L, self.H), 1.5 * self.cell)
        cz = self._soft_cells(wz, self.layers, ease) if self.layers else np.ones((len(P), 1))
        back = np.zeros(len(P))
        joint = np.full(len(P), 1e9)
        var = np.full(len(P), 0.5)
        own = np.argmax(cz, axis=1)
        for j, lay in enumerate(self.cliff_layers):
            m = np.nonzero(cz[:, j] > 1e-6)[0]
            if not len(m):
                continue
            xj = wx[m] - lay["lean"] * (z[m] - lay["zc"])
            cx = self._soft_cells(xj, lay["joints"], ease)
            face = cx @ lay["recess"] + (cx @ lay["tiltX"]) * (x[m] - cx @ lay["mids"]) \
                + (cx @ lay["tiltZ"]) * (z[m] - lay["zc"])
            back[m] += cz[m, j] * face
            mine = own[m] == j
            if lay["open"].any():
                d = np.min(np.abs(xj[:, None] - lay["joints"][lay["open"]][None, :]), axis=1)
                # the groove narrows to nothing toward the layer's top and bottom, not a slot cut to its edges
                joint[m[mine]] = d[mine] + self.groove[0] * (1.0 - _smoothstep(0.5, 0.95, cz[m[mine], j]))
            var[m[mine]] = lay["var"][np.searchsorted(lay["joints"], xj[mine])]
        top = self.cliff_layers[-1]
        drop = self._soft_cells(wx, top["joints"], ease) @ top["drop"]
        return back, drop, joint, var

    def _macro_cliff(self, P, blocks=None):
        y, z = P[:, 1], P[:, 2]
        D, H = self.W, self.H
        back, drop, _joint, _var = blocks if blocks is not None else self._cliff_blocks(P)
        # THE ENDS TURN BACK AND DOWN into the slope, so a piece reads as part of a hillside rather than a cut block
        end = _smoothstep(0.25 * self.L, 0.5 * self.L, np.abs(P[:, 0]))
        back = back + 0.6 * D * end * end
        drop = drop + 0.45 * H * end ** 1.5
        f = (-0.5 * D + back) - y
        V = P @ self._pn.T - self._pd
        for i in range(V.shape[1]):
            f = _smax(f, V[:, i], self._pk[i])
        f = _smax(f, z - (0.5 * H - drop), 0.02 * H * (0.3 + self.weathering))
        f = np.maximum(f, y - 0.5 * D)
        return np.maximum(f, -0.5 * H - z)

    # ---- evaluation -------------------------------------------------------------------------------------------------
    def _macro(self, P):
        return getattr(self, "_macro_" + self.kind)(P)

    def eval(self, P):
        out = np.empty(len(P))
        for a in range(0, len(P), _CHUNK):
            out[a:a + _CHUNK] = self._eval(P[a:a + _CHUNK])
        return out

    def _eval(self, P):
        amp, freq, octaves = self.undulation
        Pw = self._warped(P)
        blocks = self._cliff_blocks(Pw) if self.kind == "cliff" else None
        macro = self._macro_cliff(Pw, blocks) if blocks is not None else self._macro(Pw)
        f = macro + amp * _fbm(P * freq, self.seed + 7, octaves)
        if self.fractures:
            f = f + self._fracture(Pw)
        for n, d, k in self.chips:
            f = _smax(f, Pw @ n - d, k)
        if self.lamina is not None:
            t, w, amp, lseed = self.lamina
            u = (P[:, 2] + 0.8 * t * _fbm(P * (0.7 / self.s), lseed, 2)) / t
            k = np.floor(u)
            dist = np.minimum(u - k, k + 1.0 - u) * t
            depth = (_hash(k.astype(np.int64), 0, 0, lseed) % 1000) / 999.0
            f = f + amp * np.where(depth > 0.55, depth, 0.0) * np.exp(-(dist / w) ** 2)
        if self.kind == "slab" and len(self.layers) > 2:
            # BEDDING PLANES: a groove where two layers meet, coming and going along the rim
            warp = self._slab_warp(P)
            gw = max(self.H * 0.012, 1.6 * self.cell)
            fade = np.clip(0.4 + 1.2 * _fbm(P * (3.0 / self.s), self.seed + 17, 2), 0.0, 1.0)
            for j in range(1, len(self.layers) - 1):
                dist = P[:, 2] - self.tilt[0] * P[:, 0] - self.tilt[1] * P[:, 1] - (self.layers[j] + warp)
                t = np.clip(-f / (self.W * 0.03), 0.0, 1.0)
                f = np.maximum(f, -(np.abs(dist) - gw * fade * (1.0 - t)))
        if blocks is not None:
            # OPEN JOINTS: a V groove down the joints that are open, coming and going along them
            gw, gd = self.groove
            fade = np.clip(0.5 + 1.2 * _fbm(P * (2.5 / min(self.L, self.H)), self.seed + 19, 2), 0.0, 1.0)
            t = np.clip(-f / gd, 0.0, 1.0)
            f = np.maximum(f, -(blocks[2] - gw * fade * (1.0 - t)))
        for n1, d1, n2, d2, k in self.notches:
            f = _smax(f, -_smax(d1 - P @ n1, d2 - P @ n2, k), k)
        for n, off, width, depth, cseed in self.cracks:
            dist = P @ n - off
            # the crack pinches out along its length and narrows to nothing at its depth: a V, not a saw cut
            w = width * np.clip(0.35 + 1.1 * _fbm(P * (2.5 / self.s), cseed, 2), 0.0, 1.0)
            t = np.clip(-f / depth, 0.0, 1.0)
            f = np.maximum(f, -(np.abs(dist) - w * (1.0 - t)))
        return f

    def variation(self, P):
        """A value per layer or block, 0-1, for a material to vary the stone by (vertex color A)."""
        if self.kind == "cliff":
            out = np.empty(len(P))
            for a in range(0, len(P), _CHUNK):
                out[a:a + _CHUNK] = self._cliff_blocks(P[a:a + _CHUNK])[3]
            return out
        if self.kind == "slab" and len(self.layers) > 2:
            j = np.searchsorted(np.array(self.layers[1:-1]), P[:, 2] - self._slab_warp(P))
            return (np.arange(len(self.layers) - 1)[j] * 0.618034) % 1.0
        return np.full(len(P), 0.5)

    def gradient(self, P, eps):
        g = np.empty_like(P)
        for axis in range(3):
            d = np.zeros(3)
            d[axis] = eps
            g[:, axis] = (self.eval(P + d) - self.eval(P - d)) / (2.0 * eps)
        return g


# ---------------------------------------------------------------------------------------------------------------------
# meshing
# ---------------------------------------------------------------------------------------------------------------------

def _surface_nets(F, origin, h):
    """Quads on the zero set of a sampled field F (nx, ny, nz), negative inside. One vertex per grid cell the surface
    crosses, at the mean of its edge crossings; one quad per crossed grid edge, wound so its normal points out.
    Returns (verts (V, 3), quads (Q, 4))."""
    inside = F < 0.0
    nx, ny, nz = F.shape
    cs = (nx - 1, ny - 1, nz - 1)
    acc = np.zeros(cs + (3,))
    cnt = np.zeros(cs)
    crossings = []
    for axis in range(3):
        sl_a = [slice(None)] * 3
        sl_b = [slice(None)] * 3
        sl_a[axis] = slice(0, -1)
        sl_b[axis] = slice(1, None)
        a, b = F[tuple(sl_a)], F[tuple(sl_b)]
        m = inside[tuple(sl_a)] != inside[tuple(sl_b)]
        t = np.clip(np.where(m, a / np.where(m, a - b, 1.0), 0.0), 0.0, 1.0)
        crossings.append((m, inside[tuple(sl_a)]))
        others = [ax for ax in range(3) if ax != axis]
        for d0 in (0, 1):
            for d1 in (0, 1):
                sl = [slice(None)] * 3
                sl[others[0]] = slice(d0, d0 + cs[others[0]])
                sl[others[1]] = slice(d1, d1 + cs[others[1]])
                mm = m[tuple(sl)]
                acc[..., axis] += np.where(mm, t[tuple(sl)], 0.0)
                acc[..., others[0]] += mm * d0
                acc[..., others[1]] += mm * d1
                cnt += mm
    vid = np.full(cs, -1, dtype=np.int64)
    hit = cnt > 0
    vid[hit] = np.arange(int(hit.sum()))
    idx = np.argwhere(hit).astype(float)
    verts = origin + (idx + acc[hit] / cnt[hit][:, None]) * h
    quads = []
    for axis, (m, lower_inside) in enumerate(crossings):
        others = [ax for ax in range(3) if ax != axis]
        sl = [slice(None)] * 3
        sl[others[0]] = slice(1, m.shape[others[0]] - 1)
        sl[others[1]] = slice(1, m.shape[others[1]] - 1)
        e = np.argwhere(m[tuple(sl)])
        e[:, others[0]] += 1
        e[:, others[1]] += 1
        # the ring below runs counter-clockwise about +axis for x and z; for y its two axes come in (x, z) order,
        # whose cross product is -y, so the sense is reversed there
        flip = ~lower_inside[e[:, 0], e[:, 1], e[:, 2]]
        if axis == 1:
            flip = ~flip
        ring = []
        for d0, d1 in ((1, 1), (0, 1), (0, 0), (1, 0)):
            c = e.copy()
            c[:, others[0]] -= d0
            c[:, others[1]] -= d1
            ring.append(vid[c[:, 0], c[:, 1], c[:, 2]])
        q = np.stack(ring, axis=1)
        q[flip] = q[flip][:, ::-1]
        quads.append(q)
    quads = np.concatenate(quads) if quads else np.zeros((0, 4), np.int64)
    return verts, quads


def _unambiguous(F, cell, passes=64):
    """F with no AMBIGUOUS GRID FACE - a face whose corners alternate in sign round it, inside on one diagonal and
    outside on the other - so surface nets meshes it as a closed manifold.

    WHY. Surface nets puts one vertex in each crossed cell and one quad on each crossed grid edge. Across an ambiguous
    face all four of its edges are crossed, and each quad round them joins the same two cells, so the edge between
    those two vertices is shared by FOUR faces. Measured on 5.2.1 (2026-10-01, benchmark round ten's first two): seed 7 at 5,000 triangles and
    180x120x90 cm, 2 such edges; seed 23 at 2,500 and 90x80x60, 1; seed 1 at 8,000 as a 200x150x40 slab, 52 - each one
    already in surface nets' own output, before the decimation. One outside corner of each ambiguous face moves just
    inside, a millionth of a cell past zero, which moves the surface by less than that corner's distance to it; the
    vertices are stepped back onto the field afterward anyway.

    ALWAYS OUTSIDE TO INSIDE, the outside corner nearest the surface (smallest F). Flipping whichever corner was
    nearest, either way, oscillated - a cell's fix undid a face's - and did not settle within 8 passes in a seed sweep.
    One way only, the inside only grows, so it ends; a flip can make a neighbor ambiguous, so it repeats, and a field
    that still has one after `passes` is answered as this op's defect. The outer shell is never flipped: its nodes are
    all outside, and an ambiguous face or cell needs inside corners on both of its diagonals."""
    tiny = cell * 1e-6
    for _ in range(passes):
        found = 0
        for axis in range(3):
            a, b = [ax for ax in range(3) if ax != axis]
            n = F.shape

            def corner(da, db):
                sl = [slice(None)] * 3
                sl[a] = slice(da, n[a] - 1 + da)
                sl[b] = slice(db, n[b] - 1 + db)
                return tuple(sl)

            sl00, sl10, sl11, sl01 = corner(0, 0), corner(1, 0), corner(1, 1), corner(0, 1)
            s00, s10, s11, s01 = F[sl00] < 0.0, F[sl10] < 0.0, F[sl11] < 0.0, F[sl01] < 0.0
            amb = (s00 == s11) & (s10 == s01) & (s00 != s10)
            if not amb.any():
                continue
            found += int(amb.sum())
            idx = np.argwhere(amb)
            vals = np.stack([F[sl][amb] for sl in (sl00, sl10, sl11, sl01)], axis=1)
            pick = np.argmin(np.where(vals < 0.0, np.inf, vals), axis=1)
            offsets = np.array([(0, 0), (1, 0), (1, 1), (0, 1)])[pick]
            node = idx.copy()
            node[:, a] += offsets[:, 0]
            node[:, b] += offsets[:, 1]
            F[node[:, 0], node[:, 1], node[:, 2]] = -tiny
        # THE CELL'S OWN AMBIGUITY, once no face has one: two opposite corners of a cell on one side and the other six
        # on the other. Its one vertex then joins two sheets at a point - a vertex that is not manifold though every
        # edge has two faces (8 on the 200x150x40 slab once the faces were fixed). Same remedy, on the cell's 8 corners.
        n = F.shape
        cells = [F[dx:n[0] - 1 + dx, dy:n[1] - 1 + dy, dz:n[2] - 1 + dz] for dx, dy, dz in _CORNERS]
        signs = np.stack([c < 0.0 for c in cells], axis=-1)
        count = signs.sum(axis=-1)
        amb = np.zeros(count.shape, dtype=bool)
        for k, (dx, dy, dz) in enumerate(_CORNERS):
            opp = _CORNERS.index((1 - dx, 1 - dy, 1 - dz))
            if opp < k:
                continue
            pair = signs[..., k] == signs[..., opp]
            amb |= pair & (((count == 2) & signs[..., k]) | ((count == 6) & ~signs[..., k]))
        if amb.any():
            found += int(amb.sum())
            idx = np.argwhere(amb)
            vals = np.stack([c[amb] for c in cells], axis=1)
            offsets = np.array(_CORNERS)[np.argmin(np.where(vals < 0.0, np.inf, vals), axis=1)]
            node = idx + offsets
            F[node[:, 0], node[:, 1], node[:, 2]] = -tiny
        if not found:
            return F
    raise MifOpError("build_rock's sampled field still had an ambiguous grid face after %d passes, so its mesh would "
                     "not be closed - a defect in this op, not in the request. NOTHING was created." % passes)


def _project(field, verts, cell, steps=1):
    """Newton steps onto the field's zero set along its gradient (forward differences), each clamped to a cell."""
    eps = cell * 0.3
    for _ in range(steps):
        f = field.eval(verts)
        g = np.stack([(field.eval(verts + d) - f) / eps for d in np.eye(3) * eps], axis=1)
        gg = np.maximum(np.sum(g * g, axis=1), 1e-9)
        step = (f / gg)[:, None] * g
        norm = np.linalg.norm(step, axis=1, keepdims=True)
        step *= np.minimum(1.0, (cell * 0.75) / np.maximum(norm, 1e-12))
        verts = verts - step
    return verts


def _decimate(mesh, target):
    """The mesh reduced to about `target` triangles by Blender's collapse decimation, evaluated on a temporary object
    in the scene's own collection (always in the view layer) and returned as a new mesh; the input is removed."""
    tris = sum(len(p.vertices) - 2 for p in mesh.polygons)
    if tris <= target:
        return mesh
    tmp = bpy.data.objects.new("_mif_rock_decimate", mesh)
    bpy.context.scene.collection.objects.link(tmp)
    try:
        mod = tmp.modifiers.new("MifDecimate", "DECIMATE")
        mod.decimate_type = "COLLAPSE"
        mod.ratio = max(1e-4, min(1.0, target / float(tris)))
        mod.use_collapse_triangulate = True
        depsgraph = bpy.context.evaluated_depsgraph_get()
        reduced = bpy.data.meshes.new_from_object(tmp.evaluated_get(depsgraph))
    finally:
        remove_objects([tmp])
    bpy.data.meshes.remove(mesh)
    return reduced


def _drop_crumbs(verts, quads):
    """The surface nets' output without what is not the rock: a closed part that faces inward (a pocket sealed inside
    the stone) and a crumb free of the rest under a tenth of the largest part's surface. Layered kinds made both - a slab
    measured 5 inside-out parts and 8 floating, a cliff 9 and 9, and each failed mesh_quality gameReady for it. A separate
    part of real size stays: a slab's layer can stand free.
    Returns (verts, quads, {"sealedPockets", "crumbs"})."""
    labels = np.arange(len(verts))
    a = quads.ravel()
    b = quads[:, [1, 2, 3, 0]].ravel()
    while True:
        before = labels.copy()
        np.minimum.at(labels, a, labels[b])
        np.minimum.at(labels, b, labels[a])
        labels = labels[labels]
        if np.array_equal(before, labels):
            break
    owner = labels[quads[:, 0]]
    ids, part = np.unique(owner, return_inverse=True)
    if len(ids) < 2:
        return verts, quads, {"sealedPockets": 0, "crumbs": 0}
    p0, p1, p2, p3 = (verts[quads[:, i]] for i in range(4))
    cross = np.cross(p2 - p0, p3 - p1)
    area = np.bincount(part, 0.5 * np.linalg.norm(cross, axis=1), len(ids))
    # the divergence theorem over each quad as two triangles: six times the part's signed volume
    vol = np.bincount(part, np.einsum("ij,ij->i", p0, np.cross(p1, p2)) + np.einsum("ij,ij->i", p0, np.cross(p2, p3)),
                      len(ids))
    inward = vol < 0.0
    crumb = ~inward & (area < 0.1 * area[~inward].max())
    keep = ~(inward | crumb)
    kept = quads[keep[part]]
    used, inv = np.unique(kept, return_inverse=True)
    return verts[used], inv.reshape(kept.shape), {"sealedPockets": int(inward.sum()), "crumbs": int(crumb.sum())}


def _clear_slivers(mesh, longest):
    """Open up the needles the collapse leaves: triangles under 1/10000 of the longest side wide, the width
    mesh_quality's sliverFaces names. The vertex across the long edge moves away from that edge until the triangle is
    1.5 times that wide - a fraction of a millimeter on a boulder, and no vertex or edge comes or goes, so the surface
    stays closed. Taking them out instead was tried first and both ways broke it: bmesh's dissolve_degenerate left
    non-manifold edges on a layered rock, and collapsing the short edge left those and loose vertices too.
    Returns how many triangles it opened."""
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.normal_update()
    limit = longest * 1e-4
    opened = set()
    for _ in range(12):
        moved = False
        for f in bm.faces:
            if len(f.verts) != 3:
                continue
            edges = sorted(f.edges, key=lambda e: e.calc_length())
            long_edge = edges[-1].calc_length()
            if long_edge <= 0.0:
                continue
            width = 2.0 * f.calc_area() / long_edge
            if width >= limit or long_edge < 100.0 * width:
                continue
            a, b = edges[-1].verts
            c = next(v for v in f.verts if v is not a and v is not b)
            along = (b.co - a.co).normalized()
            foot = a.co + along * (c.co - a.co).dot(along)
            away = c.co - foot
            if away.length < 1e-12:
                away = c.normal.cross(along)
            if away.length < 1e-12:
                away = along.orthogonal()
            c.co = foot + away.normalized() * (1.5 * limit)
            opened.add(f.index)
            moved = True
        if not moved:
            break
    if opened:
        bm.to_mesh(mesh)
        mesh.update()
    bm.free()
    return len(opened)


def _ease_folds(mesh, uv_layer):
    """The handful of vertices a sharp fold in the surface leaves with no usable tangent: their smoothed normal stands
    nearly in the plane the tangent has to lie in, so MikkTSpace answers about zero there and Unreal warns on import (1 to 3
    on most layered rocks, on triangles of healthy size and healthy UVs - a fresh unwrap does not change them). Each moves to
    the middle of its neighbors, which eases that fold and no other. Returns how many vertices moved."""
    moved = set()
    for _ in range(4):
        row = (uv_degeneracy(mesh, uv_layer).get("tangents") or {})
        verts = row.get("vertices") or []
        if not verts:
            break
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bm.verts.ensure_lookup_table()
        for i in verts:
            v = bm.verts[i]
            ring = [e.other_vert(v).co.copy() for e in v.link_edges]
            if ring:
                v.co = sum(ring, v.co * 0.0) / len(ring)
            moved.add(i)
        bm.to_mesh(mesh)
        bm.free()
        mesh.update()
    return len(moved)


def _find_type(nodes, type_name):
    return next((n for n in nodes if n.type == type_name), None)


def _sock(sockets, *names, kind=None):
    """A socket by the first of `names` this Blender has (and of `kind`, where one node has several sockets of one
    name - the Mix node's A and B come as float, vector and color)."""
    for name in names:
        for s in sockets:
            if s.name == name and (kind is None or s.type == kind) and getattr(s, "enabled", True):
                return s
        for s in sockets:
            if s.name == name and (kind is None or s.type == kind):
                return s
    raise MifOpError("a shader socket this op needs is missing on this Blender: wanted %s, the node has %s."
                     % (" or ".join(names), ", ".join(s.name for s in sockets)))


def _mix_color(nt, fac, a, b, blend="MIX"):
    """A color mix node (ShaderNodeMix, RGBA) linking fac, a and b; each may be a socket or a constant."""
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.blend_type = blend
    for value, sock in ((fac, _sock(mix.inputs, "Factor", kind="VALUE")), (a, _sock(mix.inputs, "A", kind="RGBA")),
                        (b, _sock(mix.inputs, "B", kind="RGBA"))):
        if isinstance(value, bpy.types.NodeSocket):
            nt.links.new(value, sock)
        elif isinstance(value, (int, float)):
            sock.default_value = value
        else:
            sock.default_value = tuple(value) + (1.0,) if len(value) == 3 else tuple(value)
    return _sock(mix.outputs, "Result", kind="RGBA")


def _math(nt, op, a, b=None, clamp=False):
    node = nt.nodes.new("ShaderNodeMath")
    node.operation = op
    node.use_clamp = clamp
    for i, value in enumerate((a, b)):
        if value is None:
            continue
        if isinstance(value, bpy.types.NodeSocket):
            nt.links.new(value, node.inputs[i])
        else:
            node.inputs[i].default_value = value
    return node.outputs[0]


def _noise(nt, vector, scale, detail, roughness=None):
    node = nt.nodes.new("ShaderNodeTexNoise")
    _sock(node.inputs, "Scale").default_value = scale
    _sock(node.inputs, "Detail").default_value = detail
    if roughness is not None:
        _sock(node.inputs, "Roughness").default_value = roughness
    nt.links.new(vector, _sock(node.inputs, "Vector"))
    return _sock(node.outputs, "Fac", "Factor")


def _ramp(nt, fac, stops):
    node = nt.nodes.new("ShaderNodeValToRGB")
    els = node.color_ramp.elements
    while len(els) < len(stops):
        els.new(0.5)
    for el, (pos, rgb) in zip(els, stops):
        el.position = pos
        el.color = tuple(rgb) + (1.0,)
    nt.links.new(fac, node.inputs[0])
    return node.outputs[0]


def _rock_material(name, color, color2, moss, strata):
    """The stone, all procedural. Color: a large mottle of three tones (a dark, color, color2) with a warm-cool drift;
    MINERAL GRAINS as crisp Voronoi cells a few millimeters across, a share of them dark and a share light, as a
    granular stone shows at arm's length; small weathering pits; streaks that run DOWN the faces (a noise stretched
    in Z); lichen crusts on the open stone; darker in the cavities (vertex color R); moss on up-facing, exposed faces
    (G, gated by `moss`); soil at the base (B); a per-layer tint (A); strata bands on slabs and cliffs. Bump: the grains
    and a medium noise. A smooth noise alone read as felt. Nodes by type, sockets by name, so a localized or renamed
    UI does not break it."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = _find_type(nt.nodes, "BSDF_PRINCIPLED")
    coord = nt.nodes.new("ShaderNodeTexCoord")
    obj_co = _sock(coord.outputs, "Object")
    mid = tuple(0.5 * (a + b) for a, b in zip(color, color2))
    dark = tuple(0.55 * c for c in color)
    mottle = _noise(nt, obj_co, 0.9, 5.0, 0.62)
    base = _ramp(nt, mottle, ((0.3, dark), (0.47, color), (0.6, mid), (0.76, color2)))
    drift = _noise(nt, obj_co, 0.45, 2.0)
    base = _mix_color(nt, 1.0, base, _ramp(nt, drift, ((0.35, (1.0, 0.94, 0.86)), (0.65, (0.9, 0.95, 1.0)))),
                      "MULTIPLY")
    # MINERAL GRAINS
    vor = nt.nodes.new("ShaderNodeTexVoronoi")
    _sock(vor.inputs, "Scale").default_value = 450.0
    nt.links.new(obj_co, _sock(vor.inputs, "Vector"))
    cell = nt.nodes.new("ShaderNodeRGBToBW")
    nt.links.new(_sock(vor.outputs, "Color"), cell.inputs[0])
    grains = _ramp(nt, cell.outputs[0], ((0.05, (0.3, 0.3, 0.3)), (0.07, (0.62, 0.62, 0.62)),
                                         (0.95, (1.0, 1.0, 1.0))))
    base = _mix_color(nt, 1.0, base, grains, "MULTIPLY")
    light = _ramp(nt, cell.outputs[0], ((0.955, (0.0, 0.0, 0.0)), (0.965, (1.0, 1.0, 1.0))))
    light_f = nt.nodes.new("ShaderNodeRGBToBW")
    nt.links.new(light, light_f.inputs[0])
    base = _mix_color(nt, _math(nt, "MULTIPLY", light_f.outputs[0], 0.35), base,
                      tuple(min(1.0, 1.6 * c) for c in color2))
    # small weathering pits
    pits = _ramp(nt, _noise(nt, obj_co, 35.0, 2.0), ((0.25, (0.5, 0.5, 0.5)), (0.36, (1.0, 1.0, 1.0))))
    base = _mix_color(nt, 0.5, base, pits, "MULTIPLY")
    # STREAKS down the faces
    mapping = nt.nodes.new("ShaderNodeMapping")
    _sock(mapping.inputs, "Scale").default_value = (3.0, 3.0, 0.6)
    nt.links.new(obj_co, _sock(mapping.inputs, "Vector"))
    streak = _noise(nt, _sock(mapping.outputs, "Vector"), 3.0, 4.0, 0.5)
    base = _mix_color(nt, 0.35, base, _ramp(nt, streak, ((0.35, (0.6, 0.6, 0.62)), (0.6, (1.0, 1.0, 1.0)))),
                      "MULTIPLY")
    if strata:
        wave = nt.nodes.new("ShaderNodeTexWave")
        wave.wave_type = "BANDS"
        wave.bands_direction = "Z"
        _sock(wave.inputs, "Scale").default_value = 1.6
        _sock(wave.inputs, "Distortion").default_value = 6.0
        _sock(wave.inputs, "Detail").default_value = 4.0
        nt.links.new(obj_co, _sock(wave.inputs, "Vector"))
        band = _ramp(nt, _sock(wave.outputs, "Fac", "Factor"), ((0.0, (0.78, 0.76, 0.74)), (1.0, (1.0, 1.0, 1.0))))
        base = _mix_color(nt, 0.6, base, band, "MULTIPLY")
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "Col"
    # the bytes are stored in sRGB space (see build_rock); a gamma of 1/2.2 brings the masks back to what was written
    unlin = nt.nodes.new("ShaderNodeGamma")
    _sock(unlin.inputs, "Gamma").default_value = 1.0 / 2.2
    nt.links.new(_sock(attr.outputs, "Color"), _sock(unlin.inputs, "Color"))
    sep = nt.nodes.new("ShaderNodeSeparateColor")
    nt.links.new(_sock(unlin.outputs, "Color"), sep.inputs[0])
    cav, up, low = sep.outputs[0], sep.outputs[1], sep.outputs[2]
    tint = _ramp(nt, _sock(attr.outputs, "Alpha"), ((0.0, (0.8, 0.8, 0.82)), (1.0, (1.0, 0.96, 0.9))))
    base = _mix_color(nt, 1.0, base, tint, "MULTIPLY")
    # LICHEN: soft pale crusts on the open stone
    crust = _ramp(nt, _noise(nt, obj_co, 5.0, 4.0, 0.65), ((0.63, (0.0, 0.0, 0.0)), (0.78, (1.0, 1.0, 1.0))))
    crust_f = nt.nodes.new("ShaderNodeRGBToBW")
    nt.links.new(crust, crust_f.inputs[0])
    open_stone = _math(nt, "SUBTRACT", 0.5, cav, clamp=True)
    base = _mix_color(nt, _math(nt, "MULTIPLY", crust_f.outputs[0], _math(nt, "MULTIPLY", open_stone, 0.8),
                                clamp=True), base,
                      tuple(min(1.0, 0.75 * a + 0.3 * b) for a, b in zip(color2, (0.2, 0.2, 0.16))))
    base = _mix_color(nt, _math(nt, "MULTIPLY", cav, 0.65, clamp=True), base, (0.02, 0.018, 0.016))
    base = _mix_color(nt, _math(nt, "MULTIPLY", low, 0.7, clamp=True), base, (0.07, 0.055, 0.04))
    if moss > 0.0:
        mn = _noise(nt, obj_co, 4.0, 12.0, 0.7)
        patch = _math(nt, "MULTIPLY", _math(nt, "ADD", up, _math(nt, "MULTIPLY", cav, 0.4)), mn)
        # the threshold falls as moss rises: at 1 most of an up face is covered
        patch = _math(nt, "SUBTRACT", patch, 0.5 - 0.4 * moss)
        patch = _math(nt, "MULTIPLY", patch, 5.0, clamp=True)
        clump = _noise(nt, obj_co, 60.0, 3.0, 0.6)
        patch = _math(nt, "MULTIPLY", patch, _math(nt, "ADD", clump, 0.35), clamp=True)
        mcol = _mix_color(nt, _noise(nt, obj_co, 18.0, 6.0), (0.02, 0.035, 0.008), (0.075, 0.095, 0.025))
        base = _mix_color(nt, patch, base, mcol)
    nt.links.new(base, _sock(bsdf.inputs, "Base Color"))
    rough = nt.nodes.new("ShaderNodeMapRange")
    nt.links.new(mottle, _sock(rough.inputs, "Value", kind="VALUE"))
    _sock(rough.inputs, "To Min", kind="VALUE").default_value = 0.62
    _sock(rough.inputs, "To Max", kind="VALUE").default_value = 0.9
    nt.links.new(_sock(rough.outputs, "Result", kind="VALUE"), _sock(bsdf.inputs, "Roughness"))
    spec = bsdf.inputs.get("Specular IOR Level") or bsdf.inputs.get("Specular")
    if spec is not None:
        spec.default_value = 0.45
    medium = _noise(nt, obj_co, 14.0, 8.0, 0.6)
    height = _math(nt, "ADD", _math(nt, "MULTIPLY", _sock(vor.outputs, "Distance"), 0.3), medium)
    bump = nt.nodes.new("ShaderNodeBump")
    _sock(bump.inputs, "Strength").default_value = 0.35
    _sock(bump.inputs, "Distance").default_value = 0.006
    nt.links.new(height, _sock(bump.inputs, "Height"))
    nt.links.new(_sock(bump.outputs, "Normal"), _sock(bsdf.inputs, "Normal"))
    return mat


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


def _check_name(params, default, want_hull):
    name = take(params, "name", default=default, kind=str)
    if not name.strip():
        raise MifOpError("name cannot be empty. NOTHING was created.")
    if any(ord(ch) < 32 for ch in name):
        raise MifOpError("'name' contains a control character, which Blender's names cannot hold - got %r. NOTHING "
                         "was created." % name)
    if want_hull and len(name) > 56:
        raise MifOpError("name is %d characters; with collision on it can be at most 56, so that UCX_<name>_00 fits the "
                         "63 characters Blender keeps in a name - a truncated hull name attaches to nothing. Pass a "
                         "shorter name, or collision false. NOTHING was created." % len(name))
    return name


def _check_seed(params):
    seed = take_int(params, "seed", default=1)
    if not 0 <= seed <= 2147483647:
        raise MifOpError("seed must be 0 to 2147483647, got %d. NOTHING was created." % seed)
    return seed


def _loop_values(mesh, per_vertex):
    """Per-vertex rows expanded to the face corners, in loop order."""
    vi = np.empty(len(mesh.loops), dtype=np.int64)
    mesh.loops.foreach_get("vertex_index", vi)
    return per_vertex[vi]


def _make_hull(obj, max_vertices):
    """A UCX_<obj>_00 hull through create_collision_hull (its audit included), drawn as a wire and left out of renders.
    Returns (hull dict or None, note or None) - a refused hull is reported, the object itself stays."""
    name = "UCX_%s_00" % obj.name
    if name in bpy.data.objects:
        return None, ("an object named '%s' already exists, so no hull was made: Blender would name this one '%s.001', "
                      "and that suffix survives FBX export, attaching the hull to nothing." % (name, name))
    # A REFUSED HULL IS TRIED AGAIN AT SMALLER BUDGETS before the rock goes out without one. Seed 11 at 120x90x70 cm
    # went out with only a collisionNote when its one 48-vertex hull failed the audit (benchmark round ten, 5.2.1); the
    # enclosing hull create_collision_hull now builds passes there, and a fewer-plane hull is the fallback if one ever
    # does not. Every budget refused is named in the note.
    refusals = []
    got = None
    for budget in (max_vertices, max(16, max_vertices // 2), 12):
        try:
            got = op_create_collision_hull({"object": obj.name, "maxVertices": budget})
            break
        except MifOpError as exc:
            refusals.append("at %d vertices: %s" % (budget, exc))
    if got is None:
        return None, "no hull was made, after %d tries - %s" % (len(refusals), " | ".join(refusals))
    hull = bpy.data.objects.get(got["object"])
    if hull is not None:
        hull.display_type = "WIRE"
        hull.hide_render = True
    audit = got.get("audit") or {}
    row = (got.get("hulls") or [{}])[0]
    return {"object": got["object"], "vertices": audit.get("vertices"), "volume": audit.get("volume"),
            "audit": "closed, manifold, convex", "maxVertices": budget,
            "sourceOutsideM": row.get("sourceOutsideM"),
            "encloses": "every vertex of the rock, on or inside it (sourceOutsideM is the farthest any lies outside)"}, (
        None if not refusals else "the hull was made at %d vertices after %s" % (budget, " | ".join(refusals)))


def _size_cm(params, default):
    v = params.get("size")
    if v is None:
        return tuple(default)
    if not isinstance(v, (list, tuple)) or len(v) != 3:
        raise MifOpError("'size' takes [length, width, height] in centimeters, got %r. NOTHING was created." % (v,))
    if any(isinstance(c, bool) for c in v):
        raise MifOpError("'size' components must be numbers (centimeters), got %r. NOTHING was created." % (v,))
    try:
        return tuple(finite_float(c, "size") for c in v)
    except MifOpError as exc:
        raise MifOpError("%s NOTHING was created." % exc)


def _world_uvs(obj):
    """One smart unwrap: its 0-1 layout is copied to UVLightmap as it is, and UVMap is scaled so one unit is
    _UV_TILE_M meters of surface, for a tiling texture. Returns a note, or None when both layers were made."""
    mesh = obj.data
    try:
        op_uv_unwrap({"object": obj.name, "method": "SMART", "uvLayer": "UVMap", "islandMargin": 0.01})
    except MifOpError as exc:
        return "the UV unwrap failed and the rock has none: %s" % exc
    layout = np.empty(len(mesh.loops) * 2, dtype=np.float32)
    mesh.uv_layers["UVMap"].data.foreach_get("uv", layout)
    light = mesh.uv_layers.new(name="UVLightmap")
    light.data.foreach_set("uv", layout)
    uv = layout.reshape(-1, 2).astype(float)
    starts = np.empty(len(mesh.polygons), dtype=np.int64)
    mesh.polygons.foreach_get("loop_start", starts)
    areas = np.empty(len(mesh.polygons))
    mesh.polygons.foreach_get("area", areas)
    a, b, c = uv[starts], uv[starts + 1], uv[starts + 2]
    uv_area = float(np.sum(0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                                        - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))))
    k = math.sqrt(float(areas.sum()) / max(uv_area, 1e-12)) / _UV_TILE_M
    mesh.uv_layers["UVMap"].data.foreach_set("uv", (uv * k).astype(np.float32).ravel())
    mesh.uv_layers.active = mesh.uv_layers["UVMap"]
    return None


def _rock_masks(field, fv, final, scale, cell):
    """Vertex color rows from the field at the vertices: R cavity (the field's own occlusion: how far the surface
    closes in along the normal), G up-facing and exposed, B ground contact, A the stratum or block value."""
    g = field.gradient(fv, cell * 0.5)
    normals = _unit_rows(g / scale)
    fn = _unit_rows(g)
    occl = np.zeros(len(fv))
    for k, dist in enumerate((0.5, 1.0, 2.0, 4.0)):
        h = dist * max(cell, field.s * 0.01)
        occl += (h - np.minimum(h, field.eval(fv + fn * h))) / h * (0.5 ** k)
    cavity = np.clip(occl / 1.2, 0.0, 1.0)
    up = _smoothstep(0.35, 0.85, normals[:, 2]) * (1.0 - cavity)
    height_m = float(final[:, 2].max() - final[:, 2].min())
    ground = 1.0 - _smoothstep(0.0, max(0.15, 0.12 * height_m), final[:, 2])
    return np.stack([cavity, up, ground, field.variation(fv)], axis=1)


def _rock_grid(field, size_m, cell):
    """The field sampled on a grid a quarter larger than the rock, in a NARROW BAND: a coarse grid of every fourth
    node first, then the fine nodes only in the coarse blocks the surface can reach (a corner within two block
    diagonals of it, or corners of both signs). Elsewhere a node takes its block's sign - all surface nets reads there.
    The outer shell is set outside, so the surface always closes. Returns (F, origin)."""
    half = np.array(size_m) * 0.5
    blocks = np.ceil(half * 2.5 / (4.0 * cell)).astype(int) + 2
    lo = -blocks * 4.0 * cell * 0.5
    coarse_axes = [lo[i] + np.arange(blocks[i] + 1) * 4.0 * cell for i in range(3)]
    cx, cy, cz = np.meshgrid(*coarse_axes, indexing="ij")
    C = field.eval(np.stack([cx.ravel(), cy.ravel(), cz.ravel()], axis=1)).reshape(cx.shape)
    del cx, cy, cz
    corners = [C[i:i + blocks[0], j:j + blocks[1], k:k + blocks[2]] for i in (0, 1) for j in (0, 1) for k in (0, 1)]
    cmin = np.minimum.reduce(corners)
    cmax = np.maximum.reduce(corners)
    margin = 2.0 * 4.0 * cell * math.sqrt(3.0)
    near = (np.minimum(np.abs(cmin), np.abs(cmax)) < margin) | ((cmin < 0.0) & (cmax > 0.0))
    fill = np.where(corners[0] < 0.0, -margin, margin)
    counts = blocks * 4 + 1
    bi = [np.minimum(np.arange(counts[a]) // 4, blocks[a] - 1) for a in range(3)]
    idx = np.ix_(bi[0], bi[1], bi[2])
    F = fill[idx]
    todo = np.nonzero(near[idx])
    del idx
    pts = np.stack([lo[a] + todo[a] * cell for a in range(3)], axis=1)
    F[todo] = field.eval(pts)
    F[0, :, :] = F[-1, :, :] = F[:, 0, :] = F[:, -1, :] = F[:, :, 0] = F[:, :, -1] = cell
    return F, lo


def op_build_rock(params):
    """Build one rock - a boulder, a slab or a cliff face - as ONE mesh object with one stone material, its pivot at
    the bottom center with the rock sunk slightly below it, in the size asked for. The same seed and parameters give
    the same rock on a given Blender.

    params:
      name (str)            default "Rock"
      kind (str)            boulder (default), slab, cliff
      seed (int)            0 or more, default 1
      size ([cm, cm, cm])   [length, width, height], the mesh's bounds, sunk part included. Defaults: boulder
                            140x110x85, slab 240x170x55, cliff 800x300x600. Each side 5 to 5000 cm; a cliff's width
                            is its depth, with the face on -Y
      triangles (int)       the count the rock is reduced to, 500 to 1000000; default 30000 (cliff 120000)
      sink (cm)             how far below the pivot the rock reaches, so it sits into the ground; default 6% of the
                            height for a boulder, 8% for a slab, 3% for a cliff; 0 to half the height
      weathering (0-1)      0 fresh-broken, sharp edges; 1 worn and rounded. Default boulder 0.2, slab 0.2, cliff 0.25
      strata (bool)         layers and bedding planes; default on for slab and cliff, off for boulder
      stoneColor, stoneColor2 ([r,g,b] linear 0-1)   the stone's two tones
      moss (0-1)            moss on up-facing exposed faces in the preview material; default 0
      collision (bool)      a UCX_<name>_00 convex hull; default on
      location ([x,y,z] or {x,y,z})   in METERS (Blender units), not centimeters like size and sink
      rotation ([x,y,z] or {x,y,z})   in RADIANS, XYZ Euler
      collection
    UNITS, read off the code (2026-10-03, a short film): size and sink are centimeters (divided by 100 before
    anything is built); location and rotation go to the object as given, through ops_create._place_values - meters
    and radians.
    """
    reject_unknown(params, BUILD_ROCK_PARAMS, "build_rock", _ROCK_NOTES)
    started = time.perf_counter()
    # EVERY VALUE IS READ AND CHECKED BEFORE ANYTHING IS CREATED
    kind = take(params, "kind", default="boulder", kind=str)
    if kind not in _ROCK_KINDS:
        raise MifOpError("kind must be one of %s, got '%s'. NOTHING was created." % (", ".join(_ROCK_KINDS), kind))
    cfg = _ROCK_DEFAULTS[kind]
    want_hull = take_bool(params, "collision", default=True)
    name = _check_name(params, "Rock", want_hull)
    seed = _check_seed(params)
    size = _size_cm(params, cfg["size"])
    for axis, v in zip(("length", "width", "height"), size):
        if not 5.0 <= v <= 5000.0:
            raise MifOpError("size %s is %g cm; each side is 5 to 5000 cm. NOTHING was created." % (axis, v))
    if max(size) / min(size) > 40.0:
        raise MifOpError("size %s is more than 40 to 1 between its longest and shortest sides; a rock that thin is a "
                         "sheet. NOTHING was created." % (list(size),))
    triangles = take_int(params, "triangles", default=cfg["triangles"])
    if not _MIN_TRIANGLES <= triangles <= _MAX_TRIANGLES:
        raise MifOpError("triangles is %d; it must be %d to %d. NOTHING was created."
                         % (triangles, _MIN_TRIANGLES, _MAX_TRIANGLES))
    sink = take_float(params, "sink", default=cfg["sink"] * size[2])
    if not 0.0 <= sink <= 0.5 * size[2]:
        raise MifOpError("sink is %g cm; it must be 0 to half the height (%g cm). NOTHING was created."
                         % (sink, 0.5 * size[2]))
    weathering = take_float(params, "weathering", default=cfg["weathering"])
    if not 0.0 <= weathering <= 1.0:
        raise MifOpError("weathering is %g; it must be 0 (fresh-broken) to 1 (worn round). NOTHING was created."
                         % weathering)
    strata = take_bool(params, "strata", default=cfg["strata"])
    color = _color3(params, "stoneColor", (0.085, 0.08, 0.072))
    color2 = _color3(params, "stoneColor2", (0.23, 0.215, 0.19))
    moss = take_float(params, "moss", default=0.0)
    if not 0.0 <= moss <= 1.0:
        raise MifOpError("moss is %g; it must be 0 to 1. NOTHING was created." % moss)
    try:
        placed = _place_values(params)
    except MifOpError as exc:
        raise MifOpError("%s. NOTHING was created." % str(exc).rstrip("."))
    coll = _resolve_collection(params)
    if np is None:
        raise MifOpError("this Blender's Python has no numpy, which build_rock builds with. NOTHING was created.")

    size_m = tuple(v / 100.0 for v in size)
    # ---- the grid: fine enough that the mesh is about three times the budget before it is reduced ----------------
    L, W, H = size_m
    area = 2.0 * (L * W + L * H + W * H) * (0.8 if kind == "boulder" else 1.1)
    cell = math.sqrt(2.0 * area / ((2.2 if kind == "cliff" else 3.0) * triangles))
    # memory: no more than about 8 million grid nodes (only the band near the surface is evaluated)
    while (1.25 * L / cell + 9) * (1.25 * W / cell + 9) * (1.25 * H / cell + 9) > 6.0e6:
        cell *= 1.1
    field = _RockField(kind, size_m, weathering, strata, seed, cell)
    F, lo = _rock_grid(field, size_m, cell)
    verts, quads = _surface_nets(_unambiguous(F, cell), lo, cell)
    del F
    if len(quads) < 8:
        raise MifOpError("build_rock sampled no surface for this seed and size - a defect in this op, not in the "
                         "request. NOTHING was created.")
    verts = _project(field, verts, cell)
    verts, quads, dropped = _drop_crumbs(verts, quads)

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts.tolist(), [], quads.tolist())
    mesh.update()
    mesh = _decimate(mesh, triangles)
    n = len(mesh.vertices)
    fv = np.empty(n * 3)
    mesh.vertices.foreach_get("co", fv)
    fv = fv.reshape(-1, 3)
    # ---- the size asked for, exactly, with the pivot at the bottom center and the rock sunk `sink` below it -------
    bmin, bmax = fv.min(axis=0), fv.max(axis=0)
    scale = np.array(size_m) / np.maximum(bmax - bmin, 1e-9)
    final = (fv - bmin) * scale
    final[:, 0] -= 0.5 * size_m[0]
    final[:, 1] -= 0.5 * size_m[1]
    final[:, 2] -= sink / 100.0
    mesh.vertices.foreach_set("co", final.ravel())
    mesh.polygons.foreach_set("use_smooth", np.ones(len(mesh.polygons), dtype=bool))
    mesh.update()
    slivers_cleared = _clear_slivers(mesh, max(size_m))
    col = _rock_masks(field, fv, final, scale, cell)
    attr = mesh.color_attributes.new(name="Col", type="BYTE_COLOR", domain="CORNER")
    # color_srgb, as build_tree writes: each byte is value x 255, which the FBX carries and Unreal reads as written
    attr.data.foreach_set("color_srgb", _loop_values(mesh, col).astype(np.float32).ravel())

    obj = bpy.data.objects.new(name, mesh)
    _link_new(obj, coll)
    _place(obj, placed)
    mat = _rock_material(obj.name + "_Stone", color, color2, moss, strata and kind != "boulder")
    mesh.materials.append(mat)
    uv_note = _world_uvs(obj)
    folds_eased = _ease_folds(mesh, "UVMap") if uv_note is None else 0

    hull, hull_note = None, None
    if want_hull:
        hull, hull_note = _make_hull(obj, 48 if kind != "cliff" else 96)

    out = _created(obj, coll, "MESH", "build_rock", name)
    tris = sum(len(p.vertices) - 2 for p in mesh.polygons)
    out.update({
        "kind": kind, "seed": seed,
        "sizeCm": [round(float(v) * 100.0, 2) for v in (final.max(axis=0) - final.min(axis=0))],
        "sinkCm": round(float(-final[:, 2].min()) * 100.0, 2),
        "weathering": weathering, "strata": strata,
        "vertices": len(mesh.vertices), "faces": len(mesh.polygons), "triangles": tris,
        "material": mat.name,
        "uvLayers": [u.name for u in mesh.uv_layers], "uvNote": uv_note,
        "uvMap": {"UVMap": "tiles at %.1f m of surface per unit, for a tiling rock texture" % _UV_TILE_M,
                  "UVLightmap": "the same islands packed 0-1 without overlaps, for a lightmap"},
        "colorAttribute": {"name": "Col", "domain": "CORNER", "type": "BYTE_COLOR",
                           "stored": "each byte is value x 255 (sRGB space), so export_mesh's FBX carries the values "
                                     "and Unreal's VertexColor node reads them as written",
                           "R": "cavity: 0 on open faces, toward 1 in cracks and hollows",
                           "G": "up-facing and exposed: where moss or dust settles",
                           "B": "ground contact: 1 at the pivot's height, 0 from about 12% of the height up",
                           "A": "a value per stratum or block, for tint variation (0.5 on a boulder)"},
        "collisionHull": hull, "collisionNote": hull_note,
        "pivot": "the bottom center, at the object's origin, Z up; the rock reaches sinkCm below it",
        "cellCm": round(cell * 100.0, 3),
        "cleanup": dict(dropped, sliversOpened=slivers_cleared, foldsEased=folds_eased),
        "elapsedSeconds": round(time.perf_counter() - started, 2),
    })
    return out


# =====================================================================================================================
# ground cover - build_plant
# =====================================================================================================================

BUILD_PLANT_PARAMS = (
    "name", "kind", "seed", "height", "width", "count", "bare", "weeds", "weedMix", "flowers", "length", "ends", "shag",
    "leafLength", "leafColor", "leafColor2", "location", "rotation", "collection",
)

_PLANT_NOTES = {
    "size": "pass height and width in centimeters.",
    "scale": "pass height and width in centimeters.",
    "collision": "plants get no collision hull: ground cover is walked through.",
    "preset": "the form is chosen by kind: grass, fern, plant, bush, turf or hedge.",
    "lawn": "kind turf is a square tile of lawn, width on a side, that tiles: lay copies edge to edge, each turned by "
            "any quarter turn (a tile with bare patches: all at the same turn).",
    "density": "for turf, count is the blades on the whole tile (bare patches take their share of it).",
    "tileSize": "for turf, width is the tile's side in centimeters.",
    "species": "kinds are forms (grass, fern, plant, bush, turf, hedge), not species; height, width, count and the colors make "
               "the species.",
    "texture": "every blade and leaf is geometry, so there is no texture and no opacity mask; color comes from "
               "leafColor, leafColor2 and the per-blade value in vertex color R.",
    "cards": "alpha cards are not made: every blade and leaf is geometry, which Nanite carries.",
    "tree": "trees are build_tree's.",
    "barkColor": "only a bush has stems; they take a bark tone derived from the leaf colors.",
    "clover": "lawn weeds are turf's: weeds is the share of the tile they cover and weedMix [clover, plantain, "
              "dandelion] splits it.",
    "dandelion": "lawn weeds are turf's: weeds is the share of the tile they cover and weedMix [clover, plantain, "
                 "dandelion] splits it; flowers puts heads on the dandelions and the clover.",
    "plantain": "lawn weeds are turf's: weeds is the share of the tile they cover and weedMix [clover, plantain, "
                "dandelion] splits it.",
    "weedColor": "each weed gets its own material slot (<name>_Clover, _Plantain, _Dandelion, _Flowers) with preview "
                 "colors; set the look on those slots' materials.",
    "flowerColor": "flowers share one slot, <name>_Flowers: vertex color R near 0 is a clover head, near 1 a "
                   "dandelion's.",
    "thickness": "for a hedge, width is its thickness at the foot in centimeters (the top is cut a tenth narrower).",
    "depth": "for a hedge, width is its thickness at the foot in centimeters.",
    "hedgeLength": "a hedge's run along X is length, in centimeters: the step to lay pieces at.",
    "tileLength": "a hedge's run along X is length, in centimeters: the step to lay pieces at.",
    "caps": "a hedge's clipped ends are ends: none (a middle piece that tiles), start (-X), end (+X) or both.",
    "clipped": "a hedge is always clipped; shag is the growth since the last cut, 0 freshly sheared to 1 a summer's "
               "shoots.",
    "overgrown": "shag is a hedge's growth since the last cut: 0 freshly sheared to 1 shoots a third of a meter out.",
    "leafSize": "a hedge's leaves are leafLength centimeters long: about 4 for privet, 2 for box.",
    "litter": "a hedge's fallen leaves lie along its foot in their own slot, <name>_Litter; there is no switch.",
}

_PLANT_KINDS = ("grass", "fern", "plant", "bush", "turf", "hedge")
# Per kind: height and width in cm, the count (blades, fronds, leaves, leaves), and the two greens (linear). The
# defaults land in the triangle ranges game pieces of their size are built at: a grass tuft about 1,100 and a
# broadleaf plant about 500 (a small piece's middle), a fern about 3,500 and a 1.2 m bush about 22,000 (the upper end
# of small and medium pieces, which Nanite carries); count scales each. The default height and width are each kind's
# own grown proportions, so the fit to them barely moves it; a height and width far from those stretch the form.
_PLANT_DEFAULTS = {
    "grass": {"height": 35.0, "width": 60.0, "count": 60, "colors": ((0.09, 0.19, 0.035), (0.24, 0.30, 0.07)),
              "range": (1, 400)},
    "fern": {"height": 55.0, "width": 130.0, "count": 10, "colors": ((0.04, 0.13, 0.025), (0.10, 0.22, 0.045)),
             "range": (1, 40)},
    "plant": {"height": 30.0, "width": 60.0, "count": 14, "colors": ((0.05, 0.14, 0.03), (0.12, 0.24, 0.05)),
              "range": (1, 60)},
    "bush": {"height": 110.0, "width": 130.0, "count": 2000, "colors": ((0.06, 0.15, 0.03), (0.13, 0.23, 0.045)),
             "range": (50, 20000)},
    # a metre of city lawn at mowing height: 3,000 blades (15,000 triangles) cover it seen from standing height over a
    # ground material of the same green; lusher lawns take more, a 2 m tile four times as many for the same density
    "turf": {"height": 9.0, "width": 100.0, "count": 3000, "colors": ((0.07, 0.15, 0.03), (0.17, 0.25, 0.06)),
             "range": (100, 100000)},
    # a metre of privet hedge 1.2 m tall and 60 cm thick: about 14,000 leaves of 10 triangles. count, unless given, is
    # the clipped face's area in leaves twice over (_HEDGE_COVER), so it follows the size and the leaf asked for
    "hedge": {"height": 120.0, "width": 60.0, "count": None, "colors": ((0.03, 0.085, 0.022), (0.075, 0.14, 0.035)),
              "range": (200, 400000)},
}


class _Geo:
    """Triangles accumulated with their per-corner UV and color and a material index each."""

    def __init__(self):
        self.verts, self.tris, self.uv, self.col, self.mat = [], [], [], [], []

    def add(self, verts, tris, uv, col, mat):
        """verts (n, 3); tris (m, 3) local indices; uv (n, 2) and col (n, 4) per vertex; one material index."""
        base = len(self.verts)
        self.verts.extend(map(tuple, verts))
        for t in tris:
            self.tris.append((t[0] + base, t[1] + base, t[2] + base))
            self.uv.extend(uv[i] for i in t)
            self.col.extend(col[i] for i in t)
            self.mat.append(mat)

    def add_bulk(self, verts, tris, uv_corner, col_corner, mat):
        """Many triangles at once: verts (n, 3), tris (m, 3) local indices, uv (3m, 2) and color (3m, 4) PER CORNER."""
        base = len(self.verts)
        self.verts.extend(map(tuple, np.asarray(verts).tolist()))
        self.tris.extend(map(tuple, (np.asarray(tris) + base).tolist()))
        self.uv.extend(map(tuple, np.asarray(uv_corner).tolist()))
        self.col.extend(map(tuple, np.asarray(col_corner).tolist()))
        self.mat.extend([mat] * len(tris))


def _strip(spine, widths, sides, normals, fold):
    """A leaf or blade along a spine: three vertices per row (left edge, midrib, right edge), the midrib LOWERED along
    the (upward) face normal by `fold` times the half-width, so the leaf is a shallow V as real leaves and blades are -
    a midrib raised into a tent lit one half and shaded the other, and read as half-moons. The last spine point is the
    tip, one vertex. Returns (verts, tris, uv) - u across (0-1), v along (0 at the base, 1 at the tip)."""
    n = len(spine)
    verts, uv = [], []
    for i in range(n - 1):
        h = 0.5 * widths[i]
        verts += [spine[i] - sides[i] * h, spine[i] - normals[i] * (fold * h), spine[i] + sides[i] * h]
        v = i / (n - 1)
        uv += [(0.0, v), (0.5, v), (1.0, v)]
    verts.append(spine[-1])
    uv.append((0.5, 1.0))
    tris = []
    for i in range(n - 2):
        a = 3 * i
        b = a + 3
        tris += [(a, a + 1, b + 1), (a, b + 1, b), (a + 1, a + 2, b + 2), (a + 1, b + 2, b + 1)]
    a = 3 * (n - 2)
    tip = len(verts) - 1
    tris += [(a, a + 1, tip), (a + 1, a + 2, tip)]
    return np.array(verts), tris, uv


def _frame(spine, lean_side):
    """Tangents, a side vector per point (perpendicular to the tangent, kept close to `lean_side`) and the face normal."""
    t = np.gradient(spine, axis=0)
    t = _unit_rows(t)
    side = lean_side[None, :] - t * (t @ lean_side)[:, None]
    side = _unit_rows(side)
    normal = _unit_rows(np.cross(side, t))
    if float(normal[:, 2].sum()) < 0.0:
        normal = -normal
    return t, side, normal


def _arc(base, direction, length, droop, segs):
    """A spine from base along `direction`, bending down under its own weight by `droop` (0 straight, 1 hanging)."""
    d = _unit_rows(np.asarray(direction, dtype=float))
    pts = [np.asarray(base, dtype=float)]
    step = length / segs
    for i in range(segs):
        f = (i + 0.5) / segs
        dd = d + np.array((0.0, 0.0, -1.0)) * droop * f * f * 2.2
        pts.append(pts[-1] + _unit_rows(dd) * step)
    return np.array(pts)


def _leaf_profile(kind, n):
    """Relative width along a leaf, base to tip."""
    t = np.linspace(0.0, 1.0, n)
    if kind == "blade":
        return 1.0 - t ** 1.6
    if kind == "pinna":
        return np.sin(np.pi * np.clip(t * 0.9 + 0.1, 0.0, 1.0)) ** 0.8
    # ovate, from a petiole's width at its base: from zero the first row collapsed to a point and each leaf carried two
    # zero-area triangles (a default bush measured 4000 degenerate faces)
    return np.maximum(np.sin(np.pi * t) ** 0.7, 0.07) * (1.0 - 0.3 * t)


def _col_rows(n, r, g0, g1, b):
    g = np.linspace(g0, g1, n)
    return [(r, float(x), b, 1.0) for x in g]


def _col_strip(n_rows, r, g0, g1, b):
    """Per-vertex colors for a _strip of n_rows spine points: three per row, then the tip."""
    rows = _col_rows(n_rows, r, g0, g1, b)
    return [c for c in rows[:-1] for _ in range(3)] + [rows[-1]]


def _plant_grass(geo, rng, H, Wd, count):
    for _ in range(count):
        ang = rng.uniform(0.0, 2.0 * math.pi)
        rad = 0.16 * Wd * math.sqrt(rng.random())
        base = np.array((rad * math.cos(ang), rad * math.sin(ang), -0.03 * H))
        out = np.array((math.cos(ang), math.sin(ang), 0.0))
        # OUTER BLADES LEAN FURTHER OUT: a tuft is a fountain, not a brush
        lean = math.radians(rng.uniform(4.0, 16.0) + 38.0 * rad / max(1e-6, 0.16 * Wd))
        spin = out * math.sin(lean) + np.array((0.0, 0.0, math.cos(lean)))
        length = H * rng.uniform(0.55, 1.05)
        spine = _arc(base, spin, length, rng.uniform(0.05, 0.3) + (0.5 if rng.random() < 0.15 else 0.0), 5)
        side0 = np.cross((0.0, 0.0, 1.0), out)
        side0 = _rotate_vec(side0, spin, rng.uniform(-0.6, 0.6))
        _t, side, normal = _frame(spine, side0)
        widths = _leaf_profile("blade", len(spine)) * H * rng.uniform(0.018, 0.03)
        v, tris, uv = _strip(spine, widths, side, normal, 0.35)
        geo.add(v, tris, uv, _col_strip(len(spine), rng.random(), 0.0, 1.0, rng.random()), 0)


def _rotate_vec(v, axis, angle):
    axis = _unit_rows(np.asarray(axis, dtype=float))
    v = np.asarray(v, dtype=float)
    c, s = math.cos(angle), math.sin(angle)
    return v * c + np.cross(axis, v) * s + axis * float(np.dot(axis, v)) * (1.0 - c)


def _plant_fern(geo, rng, H, Wd, count):
    for k in range(count):
        ang = 2.0 * math.pi * (k + rng.uniform(-0.3, 0.3)) / count
        out = np.array((math.cos(ang), math.sin(ang), 0.0))
        rise = math.radians(rng.uniform(35.0, 60.0))
        d = out * math.cos(rise) + np.array((0.0, 0.0, math.sin(rise)))
        length = math.hypot(0.5 * Wd, H) * rng.uniform(0.8, 1.05)
        spine = _arc((0.0, 0.0, -0.02 * H), d, length, rng.uniform(0.45, 0.7), 14)
        side0 = np.cross((0.0, 0.0, 1.0), out)
        t, side, normal = _frame(spine, side0)
        r, b = rng.random(), rng.random()
        # the rachis: a thin strip
        rw = np.full(len(spine), 0.004 * H / 0.7)
        rw[-1] = 0.0
        v, tris, uv = _strip(spine, rw, side, normal, 0.0)
        geo.add(v, tris, uv, _col_strip(len(spine), r, 0.0, 1.0, b), 0)
        # PINNAE in pairs, longest a third of the way up, set at 55-75 degrees and drooping at their tips
        pairs = rng.randint(24, 32)
        for j in range(pairs):
            f = 0.12 + 0.86 * (j + 0.5) / pairs
            i = min(len(spine) - 2, int(f * (len(spine) - 1)))
            u = f * (len(spine) - 1) - i
            p = spine[i] * (1 - u) + spine[i + 1] * u
            env = math.sin(math.pi * min(1.0, f * 1.15)) ** 0.6
            plen = length * 0.16 * env * rng.uniform(0.85, 1.1)
            if plen < 0.01 * length:
                continue
            for sgn in (-1.0, 1.0):
                a = math.radians(rng.uniform(55.0, 75.0))
                pd = t[i] * math.cos(a) + side[i] * sgn * math.sin(a)
                ps = _arc(p, pd, plen, 0.25, 2)
                _tt, pside, pnorm = _frame(ps, np.cross(normal[i], _unit_rows(pd)))
                pnorm = np.where((pnorm @ normal[i])[:, None] < 0, -pnorm, pnorm)
                pw = _leaf_profile("pinna", len(ps)) * plen * 0.24
                v, tris, uv = _strip(ps, pw, pside, pnorm, 0.2)
                g0 = f
                geo.add(v, tris, uv, _col_strip(len(ps), r, g0, min(1.0, g0 + 0.15), b), 0)


def _plant_broadleaf(geo, rng, H, Wd, count):
    golden = math.radians(137.5)
    for k in range(count):
        ang = k * golden + rng.uniform(-0.2, 0.2)
        out = np.array((math.cos(ang), math.sin(ang), 0.0))
        inner = k / max(1, count - 1)
        # INNER LEAVES STAND UP, OUTER ONES LIE OUT: a rosette
        rise = math.radians(75.0 - 45.0 * (1.0 - inner) + rng.uniform(-8.0, 8.0))
        stalk_d = out * math.cos(rise) + np.array((0.0, 0.0, math.sin(rise)))
        stalk_len = H * rng.uniform(0.2, 0.35)
        stalk = _arc((0.0, 0.0, -0.02 * H), stalk_d, stalk_len, 0.1, 3)
        side0 = np.cross((0.0, 0.0, 1.0), out)
        _t, side, normal = _frame(stalk, side0)
        r, b = rng.random(), rng.random()
        sw = np.full(len(stalk), 0.012 * H)
        sw[-1] = 0.0
        v, tris, uv = _strip(stalk, sw, side, normal, 0.6)
        geo.add(v, tris, uv, _col_strip(len(stalk), r, 0.0, 0.3, b), 0)
        blade_len = 0.5 * Wd * rng.uniform(0.6, 0.85) - stalk_len * math.cos(rise) * 0.6
        blade_len = max(blade_len, 0.3 * H)
        bd = out * math.cos(rise * 0.5) + np.array((0.0, 0.0, math.sin(rise * 0.5)))
        blade = _arc(stalk[-1], bd, blade_len, rng.uniform(0.25, 0.5), 7)
        _t, bside, bnorm = _frame(blade, side0)
        widths = _leaf_profile("ovate", len(blade)) * blade_len * rng.uniform(0.42, 0.55)
        v, tris, uv = _strip(blade, widths, bside, bnorm, 0.3)
        geo.add(v, tris, uv, _col_strip(len(blade), r, 0.3, 1.0, b), 0)


def _bush_skeleton(rng, H, Wd):
    """A shrub's branches grown toward its foliage: points scattered through a lumpy dome that comes down near the ground
    draw the stems out from the base (space colonization), so twigs fill the crown from the skirt to the top. The first
    form, stems that forked twice at fixed lengths with tufts on the tips, read as bare sticks with leaves on the ends,
    however many leaves it was given. Returns (positions (n, 3), parent index per node, -1 at a root)."""
    nrng = np.random.default_rng(rng.randrange(1 << 30))
    # THE ENVELOPE: a unit dome with a few broad lobes, so the outline is not a ball
    lobes = _unit_rows(nrng.normal(size=(7, 3)) * (1.0, 1.0, 0.6) + (0.0, 0.0, 0.25))
    heights = nrng.uniform(0.12, 0.26, 7) * nrng.choice((-0.6, 1.0), 7)
    dirs = _unit_rows(nrng.normal(size=(2600, 3)))
    dirs[:, 2] = np.abs(dirs[:, 2]) * 0.98 + 0.02
    dirs = _unit_rows(dirs)
    reach = 1.0 + (np.maximum(dirs @ lobes.T, 0.0) ** 4 * heights).sum(axis=1)
    # mostly in the outer shell, where a shrub's leaves live, some inside
    depth = nrng.uniform(0.0, 1.0, 2600) ** 0.5 * 0.6 + 0.4
    pts = dirs * (reach * depth)[:, None] * (0.5 * Wd, 0.5 * Wd, H)
    pts = pts[pts[:, 2] > 0.08 * H][:1100]
    # THE ROOTS: a tight clump of stems at the base, each started up and out
    n_main = rng.randint(9, 14)
    step = 0.05 * max(H, 0.6 * Wd)
    pos, parent = [], []
    for k in range(n_main):
        ang = 2.0 * math.pi * (k + rng.uniform(-0.3, 0.3)) / n_main
        r0 = 0.05 * Wd * rng.uniform(0.3, 1.0)
        base = np.array((r0 * math.cos(ang), r0 * math.sin(ang), -0.02 * H))
        tilt = math.radians(rng.uniform(12.0, 42.0))
        d = np.array((math.cos(ang) * math.sin(tilt), math.sin(ang) * math.sin(tilt), math.cos(tilt)))
        pos.append(base)
        parent.append(-1)
        pos.append(base + d * step * 1.2)
        parent.append(len(pos) - 2)
    alive = np.ones(len(pts), dtype=bool)
    for _ in range(90):
        if not alive.any():
            break
        nodes = np.array(pos)
        live = pts[alive]
        dist = np.linalg.norm(live[:, None, :] - nodes[None, :, :], axis=2)
        near = np.argmin(dist, axis=1)
        pull = {}
        for i, n in enumerate(near.tolist()):
            if dist[i, n] < 0.8 * H:
                pull.setdefault(n, []).append(i)
        if not pull:
            break
        for n, who in sorted(pull.items()):
            d = _unit_rows(live[who] - nodes[n]).sum(axis=0)
            if parent[n] >= 0:
                d = _unit_rows(d) + 0.7 * _unit_rows(nodes[n] - nodes[parent[n]])
            d = _unit_rows(d + nrng.normal(size=3) * 0.18 + np.array((0.0, 0.0, 0.1)))
            pos.append(nodes[n] + d * step)
            parent.append(n)
        nodes = np.array(pos)
        reached = np.linalg.norm(pts[:, None, :] - nodes[None, -len(pull):, :], axis=2).min(axis=1) < 1.3 * step
        alive &= ~reached
    return np.array(pos), parent


def _plant_bush(geo, rng, H, Wd, count):
    """Stems from the ground grown out through a lumpy dome, thick at the base and thin at the twig tips; leaves on the
    twigs from the skirt to the top, facing up and out, sized to the bush."""
    pos, parent = _bush_skeleton(rng, H, Wd)
    n = len(pos)
    kids = [[] for _ in range(n)]
    for i, par in enumerate(parent):
        if par >= 0:
            kids[par].append(i)
    # THE PIPE MODEL: a stem's section carries what grows from it, so a stem tapers at each fork and not along its run
    tip_r = 0.0016 * H
    radius = np.full(n, tip_r)
    order = np.zeros(n, dtype=int)
    for i in range(n - 1, -1, -1):
        if kids[i]:
            radius[i] = math.sqrt(sum(radius[k] ** 2 for k in kids[i]))
            order[i] = 1 + min(order[k] for k in kids[i])
    radius = np.minimum(radius, 0.014 * H)
    # stems: runs from a root or a fork to the next fork or tip, as 5-sided tubes (4 on the twigs)
    starts = [i for i in range(n) if parent[i] < 0 or len(kids[parent[i]]) > 1]
    for s in starts:
        run = [parent[s], s] if parent[s] >= 0 else [s]
        while len(kids[run[-1]]) == 1:
            run.append(kids[run[-1]][0])
        spine, rad = pos[run], radius[run]
        if len(spine) < 2:
            continue
        sides = 5 if rad[0] > 3.0 * tip_r else 4
        _t, side, normal = _frame(spine, np.array((1.0, 0.3, 0.2)))
        rings = []
        for i, c in enumerate(spine):
            rings += [c + (side[i] * math.cos(a) + normal[i] * math.sin(a)) * rad[i]
                      for a in np.arange(sides) * 2.0 * math.pi / sides]
        m = len(spine)
        tris = []
        for i in range(m - 1):
            for j in range(sides):
                a0, b2 = i * sides + j, i * sides + (j + 1) % sides
                tris += [(a0, b2, b2 + sides), (a0, b2 + sides, a0 + sides)]
        uv = [((j % sides) / sides, i / (m - 1)) for i in range(m) for j in range(sides)]
        phase = rng.random()
        col = [(0.5, 0.15 + 0.35 * i / (m - 1), phase, 1.0) for i in range(m) for _ in range(sides)]
        geo.add(np.array(rings), tris, uv, col, 0)
    # leaves: on the outer twigs, more of them the nearer a tip, on every node down to the skirt
    segs = [i for i in range(n) if parent[i] >= 0 and order[i] <= 4]
    weights = np.array([1.0 / (1.0 + 0.5 * order[i]) for i in segs])
    owners = np.random.default_rng(rng.randrange(1 << 30)).choice(segs, size=count, p=weights / weights.sum())
    leaf_len = 0.08 * H / 1.2 * (Wd / 1.5) ** 0.3
    centre = np.array((0.0, 0.0, 0.25 * H))
    for o in owners.tolist():
        a0, b0 = pos[parent[o]], pos[o]
        p = a0 + (b0 - a0) * rng.random()
        stem_d = _unit_rows(b0 - a0)
        radial = _unit_rows(p - centre)
        d = _unit_rows(stem_d * 0.45 + radial * 0.75 + np.array((rng.gauss(0, 0.55), rng.gauss(0, 0.55),
                                                                 rng.uniform(-0.15, 0.55))))
        ln = leaf_len * rng.uniform(0.7, 1.25)
        spine = _arc(p, d, ln, 0.2, 3)
        side0 = np.cross((0.0, 0.0, 1.0), d)
        if np.linalg.norm(side0) < 1e-6:
            side0 = np.array((1.0, 0.0, 0.0))
        side0 = _rotate_vec(side0, d, rng.uniform(-0.5, 0.5))
        _t, side, normal = _frame(spine, side0)
        widths = _leaf_profile("ovate", len(spine)) * ln * 0.62
        v, tris, uv = _strip(spine, widths, side, normal, 0.3)
        g0 = 0.45 + 0.15 * min(2, max(0, 2 - order[o]))
        geo.add(v, tris, uv, _col_strip(len(spine), rng.random(), g0, 1.0, rng.random()), 1)


def _periodic_field(rng, side, x, y, waves, kmax):
    """A smooth random field over a square tile `side` on a side that repeats with it: a sum of plane waves whose
    frequencies are whole numbers of cycles across the tile, so the field's left edge meets its right and its bottom its
    top. Scaled to about -1..1."""
    f = np.zeros_like(x)
    total = 0.0
    for _ in range(waves):
        i, j = 0, 0
        while i == 0 and j == 0:
            i, j = rng.randint(-kmax, kmax), rng.randint(-kmax, kmax)
        a = 1.0 / math.hypot(i, j)            # the coarse waves carry the most, as in a real lawn's thin and lush patches
        f += a * np.sin(2.0 * math.pi * (i * x + j * y) / side + rng.uniform(0.0, 2.0 * math.pi))
        total += a
    return f / max(total, 1e-9) * 2.2


def _turnable_field(rng, side, x, y, waves, shortest, longest):
    """A periodic field like _periodic_field that is also the SAME under a quarter turn about the tile's center, so a
    tile laid turned by any quarter turn meets its neighbors with no seam: each wave is laid at its four turns, which a
    quarter turn of the field only swaps, so the sum stays. A wave's half turn is its opposite, so the four come down to
    two cosines about the center (a phase would only scale them). Turned integer frequencies are still integer ones,
    so it still repeats with the tile. Only waves between `shortest` and `longest` (the units of `side`) are used: a
    wave as long as the tile shows the tile's repeat across a lawn, and the lawn's larger thin and lush places belong
    to the ground it is laid on, not to the tile. When no whole-cycle frequency falls between them (a tile shorter
    than `shortest`), the tile's longest wave is taken. Scaled to a standard deviation of about 0.4, as
    _periodic_field's is."""
    # one frequency of each quarter-turn set: i > 0, j >= 0 holds exactly one of (i, j), (-j, i), (-i, -j), (j, -i)
    reach = math.ceil(side / shortest)
    picks = [(i, j) for i in range(1, reach + 1) for j in range(reach + 1)
             if shortest <= side / math.hypot(i, j) <= longest]
    if not picks:
        picks = [(1, 0)]
    picks = rng.sample(picks, min(waves, len(picks)))
    f = np.zeros_like(x)
    power = 0.0
    for i, j in picks:
        # the longer waves of the band carry the most; a random sign stands in for the phase
        a = rng.choice((-1.0, 1.0)) / math.hypot(i, j)
        for fi, fj in ((i, j), (-j, i)):
            f += a * np.cos(2.0 * math.pi * (fi * x + fj * y) / side)
        power += a * a
    return f / math.sqrt(max(power, 1e-12)) * 0.4


def _plant_turf(geo, rng, H, Wd, count, bare, weeds=0.0, mix=None, flowers=0.0, slots=None, report=None):
    """A square of lawn, Wd on a side and centered on the pivot, that TILES: blade roots are spread over the square by a
    density that repeats with it and is the same under a quarter turn, so tiles laid edge to edge, each turned by any
    quarter turn, show no seam, and the blades leaning over an edge stand over the next tile as they would in one lawn.
    A lawn is not tufts: blades rise all over, thicker in some places, thinner in others (20 to 70 cm across, nothing
    as wide as the tile, so its repeat does not show), a share leaning far or lying where they were trodden, longer
    ones bending over. `bare` is the share of the tile left as bare patches (worn ground, dog runs), soft-edged, from a
    second repeating field that is NOT the same under a turn (four-fold patches look made): a tile with bare patches
    meets its neighbors with no seam only laid at the same turn. The blades grow in SHOOTS, as lawn grass does: 1 to 4
    leaves (about 2.5) from one crown, the first upright and the rest fanning out round it, each curving outward along
    its length; blades scattered one by one stand like seedlings in a tray. A trodden shoot lies over as one. Each blade
    is a flat tapered strip of three segments (5 triangles): at lawn density the fold a tuft's blade gets is below a
    pixel, and the triangles go to more blades instead. Built in one pass of numpy arrays.

    `weeds` (with `mix`, `flowers`, `slots`) grows the broadleaf weeds of a lawn among the blades: _plan_weeds places
    them, the blades thin out under each (a rosette smothers the grass under it, clover patches let some through)
    and the weeds themselves are added after the blades (_turf_weeds), so the blades keep the vertex order a tile
    without weeds has. Each weed lies whole where it grows, its leaves reaching over the tile's edge as blades do,
    and none is cut at the edge, so a tile with weeds still meets turned neighbors with no seam."""
    nrng = np.random.default_rng(rng.randint(0, 2 ** 31 - 1))
    half = 0.5 * Wd
    # THE DENSITY: a coarse field for the bare patches, a finer one for the lawn's own thin and thick places. Each is
    # rebuilt from its own seed wherever it is read, so the same waves are read every time.
    s_bare, s_lush = rng.randint(0, 2 ** 30), rng.randint(0, 2 ** 30)

    def bare_field(px, py):
        return _periodic_field(random.Random(s_bare), Wd, px, py, 9, 3)

    def lush_field(px, py):
        return _turnable_field(random.Random(s_lush), Wd, px, py, 14, 0.2, 0.7)

    cut = -9.0
    if bare > 0:
        grid = np.linspace(-half, half, 96, endpoint=False)
        gx, gy = np.meshgrid(grid, grid)
        cut = float(np.quantile(bare_field(gx, gy), bare))
    # count is the blades of a full tile; bare patches take their share of it
    goal = int(round(count * (1.0 - bare)))
    plan = None
    if weeds > 0:
        # drawn after the fields' seeds, so a tile without weeds draws exactly what it always did
        wrng = random.Random(rng.randint(0, 2 ** 31 - 1))
        plan = _plan_weeds(wrng, H, Wd, weeds, mix)
        grid = np.linspace(-half, half, 96, endpoint=False) + Wd / 192.0
        gx, gy = np.meshgrid(grid, grid)
        # and so do the weeds: the share of the blades they smother comes off the count, as bare patches' does
        goal = int(round(goal * float(plan["cover"](gx, gy).mean())))
    # THE SHOOTS: how many leaves each crown carries, drawn until they add up to the blades asked for (the last one cut)
    leaves = nrng.choice(np.array((1, 2, 3, 4)), size=goal + 1, p=(0.15, 0.35, 0.3, 0.2))
    leaves = leaves[:int(np.searchsorted(np.cumsum(leaves), goal)) + 1]
    leaves[-1] -= int(leaves.sum()) - goal
    want, xs, ys = len(leaves), [], []
    for _ in range(16):
        if want <= 0:
            break
        m = int(want * 2.5) + 64
        px = nrng.uniform(-half, half, m)
        py = nrng.uniform(-half, half, m)
        keep = 0.72 + 0.28 * lush_field(px, py)
        if plan is not None:
            keep = keep * plan["cover"](px, py)
        if bare > 0:
            # soft-edged patches: thinning over a band of the field above the cut, none below it
            keep = keep * np.clip((bare_field(px, py) - cut) / 0.25 + 0.15, 0.0, 1.0) ** 1.5
        ok = nrng.random(m) < keep
        xs.append(px[ok][:want])
        ys.append(py[ok][:want])
        want -= len(xs[-1])
    sx = np.concatenate(xs)
    sy = np.concatenate(ys)
    shoots = len(sx)
    if shoots == 0:
        return 0
    leaves = leaves[:shoots]
    n = int(leaves.sum())
    first = np.repeat(np.cumsum(leaves) - leaves, leaves)
    nth = np.arange(n) - first                      # which leaf of its shoot a blade is: 0 the upright one
    shoot = np.repeat(np.arange(shoots), leaves)
    # a crown's leaves rise within a few millimetres of each other; wrapped into the square, so the tile still repeats
    x = (sx[shoot] + nrng.normal(0.0, 0.0015, n) + half) % Wd - half
    y = (sy[shoot] + nrng.normal(0.0, 0.0015, n) + half) % Wd - half
    # THE BLADES. Lean grows with length: a mown lawn stands, uncut grass bends and lies over.
    tall = min(1.0, max(0.0, (H - 0.1) / 0.4))
    # a shoot's leaves fan round it, evenly spaced with a wobble, the first upright and the others leaning out
    a0 = nrng.uniform(0.0, 2.0 * math.pi, shoots)[shoot]
    az = a0 + nth * (2.0 * math.pi / leaves[shoot]) + nrng.uniform(-0.6, 0.6, n)
    lean = np.abs(nrng.normal(0.0, math.radians(13.0 + 26.0 * tall), n))
    lean = np.where(nth == 0, 0.5 * lean, lean + nrng.uniform(math.radians(8.0), math.radians(22.0), n))
    # a trodden shoot lies over as one, its leaves the same way
    lying = (nrng.random(shoots) < (0.06 + 0.1 * tall))[shoot]
    lean[lying] = nrng.uniform(math.radians(55.0), math.radians(78.0), int(lying.sum()))
    az[lying] = a0[lying] + nrng.uniform(-0.35, 0.35, int(lying.sum()))
    length = H * nrng.uniform(0.5, 1.0, n) ** 0.7
    droop = nrng.uniform(0.0, 0.18 + 0.4 * tall, n)
    # every leaf curves outward along its length, the way it leans: the droop alone (straight down) leaves an upright
    # blade a straight stick
    curl = nrng.uniform(0.05, 0.32, n)
    width = (0.003 + 0.011 * H) * nrng.uniform(0.6, 1.0, n)
    twist = nrng.uniform(-0.7, 0.7, n)
    out = np.stack((np.cos(az), np.sin(az), np.zeros(n)), axis=1)
    d0 = out * np.sin(lean)[:, None] + np.array((0.0, 0.0, 1.0)) * np.cos(lean)[:, None]
    segs = 3
    step = (length / segs)[:, None]
    pts = [np.stack((x, y, np.full(n, -0.03 * H)), axis=1)]
    for i in range(segs):
        f = (i + 0.5) / segs
        dd = d0 + np.array((0.0, 0.0, -1.0)) * (droop * f * f * 2.2)[:, None] + out * (curl * f * 1.6)[:, None]
        dd /= np.linalg.norm(dd, axis=1)[:, None]
        nxt = pts[-1] + dd * step
        # a blade bent down lies ON the ground, it does not go into it
        nxt[:, 2] = np.maximum(nxt[:, 2], 0.002 + 0.001 * i)
        pts.append(nxt)
    # the blade's flat face: across its lean, turned a little about its own length
    across = np.stack((-np.sin(az + twist), np.cos(az + twist), np.zeros(n)), axis=1)
    rows = np.array([1.0 - (i / segs) ** 1.6 for i in range(segs)])      # the blade profile: wide at the root, a point
    verts = np.empty((n, 2 * segs + 1, 3))
    for i in range(segs):
        hw = (0.5 * width * rows[i])[:, None]
        verts[:, 2 * i] = pts[i] - across * hw
        verts[:, 2 * i + 1] = pts[i] + across * hw
    verts[:, 2 * segs] = pts[segs]
    local = []
    for i in range(segs - 1):
        a = 2 * i
        local += [(a, a + 1, a + 3), (a, a + 3, a + 2)]
    a = 2 * (segs - 1)
    local.append((a, a + 1, 2 * segs))
    local = np.array(local, dtype=np.int64)
    per = 2 * segs + 1
    tris = (local[None, :, :] + (np.arange(n) * per)[:, None, None]).reshape(-1, 3)
    uv1 = [(0.0 if k % 2 == 0 else 1.0, (k // 2) / segs) for k in range(2 * segs)] + [(0.5, 1.0)]
    uv1 = np.array(uv1)
    g1 = np.array([(k // 2) / segs for k in range(2 * segs)] + [1.0])
    r = np.clip(nrng.random(shoots)[shoot] + nrng.normal(0.0, 0.08, n), 0.0, 1.0)
    b = nrng.random(n)
    col = np.empty((n, per, 4))
    col[:, :, 0] = r[:, None]
    col[:, :, 1] = g1[None, :]
    col[:, :, 2] = b[:, None]
    col[:, :, 3] = 1.0
    corner = local.reshape(-1)
    geo.add_bulk(verts.reshape(-1, 3), tris, np.tile(uv1[corner], (n, 1)), col[:, corner].reshape(-1, 4), 0)
    if plan is not None:
        got = _turf_weeds(geo, wrng, H, plan, flowers, slots or {})
        if report is not None:
            report.update(got)
    return n


# =====================================================================================================================
# lawn weeds - build_plant turf's `weeds`
# =====================================================================================================================

_WEED_KINDS = ("clover", "plantain", "dandelion")
_WEED_MIX = (0.5, 0.3, 0.2)
# The weeds' preview greens (linear) for the slots they get, and the flowers': R picks between the two, near 0 a clover
# head's white, near 1 a dandelion's yellow. A game sets its own materials on these slots; these show the mix in a render.
_WEED_COLORS = {
    "clover": ((0.03, 0.075, 0.03), (0.065, 0.12, 0.05)),         # a darker, bluer green than the grass, matte
    "plantain": ((0.05, 0.1, 0.03), (0.1, 0.16, 0.05)),
    "dandelion": ((0.04, 0.095, 0.028), (0.09, 0.15, 0.045)),
    "flowers": ((0.78, 0.78, 0.72), (0.85, 0.55, 0.015)),
}
# A white clover leaflet: obovate, widest two thirds of the way out, its tip round (rows of along, margin along,
# width). Five rows, 18 triangles: a leaflet is a centimeter or so, and a patch has a thousand leaves a square meter.
_CLOVER_ROWS = ((0.0, 0.0, 0.12), (0.25, 0.25, 0.6), (0.55, 0.55, 0.9), (0.8, 0.8, 0.93), (0.94, 0.94, 0.62))


def _spine_at(spine, s):
    """Points at fractions `s` (0 the base, 1 the tip) of a polyline's length."""
    seg = np.linalg.norm(np.diff(spine, axis=0), axis=1)
    acc = np.concatenate(([0.0], np.cumsum(seg)))
    acc = acc / max(float(acc[-1]), 1e-12)
    s = np.clip(np.asarray(s, dtype=float), 0.0, 1.0)
    return np.stack([np.interp(s, acc, spine[:, k]) for k in range(3)], axis=-1)


def _leaf(spine, side0, rows, wmax, fold=0.3, cup=0.0, wave=0.0, waves=2.0, phase=0.0):
    """A broad leaf along a spine, from rows of (along, margin, width): each row's midrib vertex lies at `along`, a
    fraction of the spine's length, and its two margin vertices at `margin`, width x wmax / 2 out to either side - a
    margin drawn back toward the base (margin under along) is a lobe pointing back, as a dandelion's teeth do. The tip
    is the spine's end, one vertex. fold lowers the midrib (a shallow V, as _strip's), cup raises the margins and wave
    ripples them, out of step on the two sides. Rows must rise in `along`; UVs as _strip's (u 0, 0.5, 1 across, v the
    row's along). Returns (verts, tris, uv, alongs)."""
    fine_s = np.linspace(0.0, 1.0, 33)
    fine = _spine_at(spine, fine_s)
    _t, side, normal = _frame(fine, side0)

    def at(arr, s):
        return np.array([np.interp(s, fine_s, arr[:, k]) for k in range(3)])
    verts, uv, alongs = [], [], []
    for along, margin, w in rows:
        h = 0.5 * w * wmax
        pe, se, ne = at(fine, margin), at(side, margin), at(normal, margin)
        lift_l = ne * h * (cup + wave * math.sin(2.0 * math.pi * (waves * margin + phase)))
        lift_r = ne * h * (cup + wave * math.sin(2.0 * math.pi * (waves * margin + phase + 0.37)))
        verts += [pe - se * h + lift_l, at(fine, along) - at(normal, along) * (fold * h), pe + se * h + lift_r]
        uv += [(0.0, along), (0.5, along), (1.0, along)]
        alongs.append(along)
    verts.append(np.asarray(spine[-1], dtype=float))
    uv.append((0.5, 1.0))
    n = len(rows) + 1
    tris = []
    for i in range(n - 2):
        a = 3 * i
        b = a + 3
        tris += [(a, a + 1, b + 1), (a, b + 1, b), (a + 1, a + 2, b + 2), (a + 1, b + 2, b + 1)]
    a = 3 * (n - 2)
    tip = len(verts) - 1
    tris += [(a, a + 1, tip), (a + 1, a + 2, tip)]
    return np.array(verts), tris, uv, alongs


def _col_leaf(alongs, r, g0, g1, b):
    """Per-vertex colors for a _leaf: three a row, then the tip; G (the wind weight) from g0 at the base to g1."""
    out = []
    for s in alongs:
        out += [(r, g0 + (g1 - g0) * s, b, 1.0)] * 3
    return out + [(r, g1, b, 1.0)]


def _plan_weeds(wr, H, Wd, weeds, mix):
    """Where a tile's weeds grow: `weeds` of its ground split by `mix` among white clover, which spreads in patches 30
    cm to a metre across, and broadleaf plantain and dandelion, which grow as single flat rosettes, half of them near
    another of their kind (their seed falls close). Rosettes do not crowd onto each other; clover runs under anything.
    Rosettes' centers lie anywhere in the tile, a clover patch's half its size inside the edge, and nothing is
    wrapped: a weed near an edge reaches over it whole. Returns the patches,
    the rosettes and `cover`, the share of blades left growing at any point (1 clear of weeds, about 0.1 in a
    rosette's middle, 0.65 in a clover patch's)."""
    half = 0.5 * Wd
    area = Wd * Wd
    tall = min(1.0, max(0.0, (H - 0.08) / 0.3))
    mix = tuple(mix or _WEED_MIX)
    total = float(sum(mix))
    share = [weeds * m / total for m in mix]
    clover, rosettes = [], []
    target, got = share[0] * area, 0.0
    while target > 0 and got < target and len(clover) < 64:
        aspect = wr.uniform(0.5, 0.9)
        rx = min(wr.uniform(0.15, 0.5), 0.4 * Wd, max(0.06, math.sqrt((target - got) / (0.75 * math.pi * aspect))))
        ry = rx * aspect
        # a patch's middle half its size inside the edge: it still runs to the edge and on over it, by half its size
        keep_in = half - 0.5 * rx
        clover.append((wr.uniform(-keep_in, keep_in), wr.uniform(-keep_in, keep_in), rx, ry, wr.uniform(0.0, math.pi),
                       wr.random()))
        got += 0.75 * math.pi * rx * ry
    for kind, k, (r0, r1), grow in (("plantain", 1, (0.06, 0.12), 0.6), ("dandelion", 2, (0.07, 0.15), 0.5)):
        target, got, mine = share[k] * area, 0.0, []
        while target > 0 and got < target and len(mine) < 400:
            r = min(wr.uniform(r0, r1) * (1.0 + grow * tall), 0.3 * Wd)
            spot = None
            for _ in range(24):
                if mine and wr.random() < 0.5:
                    px, py, pr = mine[wr.randrange(len(mine))][1:4]
                    a, d = wr.uniform(0.0, 2.0 * math.pi), wr.uniform(1.0, 2.6) * (pr + r) * 0.6
                    x, y = px + d * math.cos(a), py + d * math.sin(a)
                    if abs(x) > half or abs(y) > half:
                        continue
                else:
                    x, y = wr.uniform(-half, half), wr.uniform(-half, half)
                if all((x - o[1]) ** 2 + (y - o[2]) ** 2 >= (0.75 * (r + o[3])) ** 2 for o in rosettes):
                    spot = (x, y)
                    break
            if spot is None:
                break
            ros = (kind, spot[0], spot[1], r, wr.random())
            mine.append(ros)
            rosettes.append(ros)
            got += 0.7 * math.pi * r * r

    def cover(px, py):
        f = np.ones_like(px, dtype=float)
        for cx, cy, rx, ry, ang, _tone in clover:
            dx, dy = px - cx, py - cy
            u = (dx * math.cos(ang) + dy * math.sin(ang)) / rx
            v = (-dx * math.sin(ang) + dy * math.cos(ang)) / ry
            f *= 1.0 - 0.35 * np.clip(1.0 - (u * u + v * v), 0.0, 1.0)
        for _kind, cx, cy, r, _tone in rosettes:
            d = np.hypot(px - cx, py - cy) / r
            f *= 1.0 - 0.9 * _smoothstep(0.95, 0.35, d)
        return f
    return {"clover": clover, "rosettes": rosettes, "cover": cover, "tall": tall}


def _clover_leaf(geo, nr, H, x, y, tone, slot):
    """A white clover leaf: a stalk from the creeping stem up into the grass, three leaflets held out flat from its top."""
    hp = H * nr.uniform(0.35, 0.9)
    lean, az = nr.uniform(0.0, 0.4), nr.uniform(0.0, 2.0 * math.pi)
    root = np.array((x, y, -0.004))
    top = root + np.array((math.sin(lean) * math.cos(az), math.sin(lean) * math.sin(az), math.cos(lean))) * hp
    mid = 0.5 * (root + top) + np.array((-math.sin(az), math.cos(az), 0.0)) * hp * nr.uniform(-0.08, 0.08)
    stalk = np.array((root, mid, top))
    _t, side, normal = _frame(stalk, np.array((-math.sin(az), math.cos(az), 0.0)))
    r, b = float(np.clip(tone + nr.normal(0.0, 0.07), 0.0, 1.0)), float(nr.random())
    v, tris, uv = _strip(stalk, np.array((0.0014, 0.0012, 0.0)), side, normal, 0.0)
    geo.add(v, tris, uv, _col_strip(3, r, 0.0, 0.35, b), slot)
    size = 0.009 + 0.03 * H
    a0, tilt0 = nr.uniform(0.0, 2.0 * math.pi), nr.uniform(-0.1, 0.35)
    for k in range(3):
        a = a0 + k * 2.0 * math.pi / 3.0 + nr.uniform(-0.15, 0.15)
        length = size * nr.uniform(0.85, 1.15)
        tilt = tilt0 + nr.uniform(-0.15, 0.15)
        d = np.array((math.cos(a) * math.cos(tilt), math.sin(a) * math.cos(tilt), math.sin(tilt)))
        spine = _arc(top, d, length, nr.uniform(0.0, 0.1), 5)
        v, tris, uv, al = _leaf(spine, np.array((-math.sin(a), math.cos(a), 0.0)), _CLOVER_ROWS, length * 0.85,
                                fold=0.3, cup=0.08)
        geo.add(v, tris, uv, _col_leaf(al, r, 0.35, 0.6, b), slot)


def _flower_head(geo, nr, centre, up, radius, rings, r, b, slot):
    """Florets standing out from a flower's center: `rings` of (count, angle from `up` in degrees, length as a share of
    the radius, width in meters), each floret a strap that narrows to its tip."""
    up = _unit_rows(np.asarray(up, dtype=float))
    ref = np.array((1.0, 0.0, 0.0)) if abs(up[0]) < 0.9 else np.array((0.0, 1.0, 0.0))
    e1 = _unit_rows(np.cross(up, ref))
    e2 = np.cross(up, e1)
    made = 0
    for count, deg, share, width in rings:
        a0 = nr.uniform(0.0, 2.0 * math.pi)
        for k in range(count):
            a = a0 + 2.0 * math.pi * k / count + nr.uniform(-0.12, 0.12)
            tilt = math.radians(deg + nr.uniform(-6.0, 6.0))
            d = up * math.cos(tilt) + (e1 * math.cos(a) + e2 * math.sin(a)) * math.sin(tilt)
            length = radius * share * nr.uniform(0.85, 1.1)
            base = centre + d * 0.15 * radius
            spine = np.array((base, base + d * 0.5 * length, base + d * length))
            _t, side, normal = _frame(spine, np.cross(up, d) if np.linalg.norm(np.cross(up, d)) > 1e-6 else e1)
            v, tris, uv = _strip(spine, np.array((width, 0.9 * width, 0.0)), side, normal, 0.2)
            geo.add(v, tris, uv, _col_strip(3, float(np.clip(r + nr.normal(0.0, 0.04), 0.0, 1.0)), 1.0, 1.0, b), slot)
            made += 1
    return made


def _flower_stalk(geo, nr, base, height, width, r, b, slot):
    """A flower's stalk from the ground up, bowed a little; returns its top and the way it faces there."""
    az, lean = nr.uniform(0.0, 2.0 * math.pi), nr.uniform(0.0, 0.3)
    d = np.array((math.sin(lean) * math.cos(az), math.sin(lean) * math.sin(az), math.cos(lean)))
    spine = _arc(np.asarray(base, dtype=float), d, height, nr.uniform(-0.05, 0.1), 4)
    _t, side, normal = _frame(spine, np.array((-math.sin(az), math.cos(az), 0.0)))
    w = np.full(len(spine), width)
    w[-1] = 0.6 * width
    v, tris, uv = _strip(spine, w, side, normal, 0.0)
    geo.add(v, tris, uv, _col_strip(len(spine), r, 0.0, 1.0, b), slot)
    return spine[-1], _unit_rows(spine[-1] - spine[-2])


# A broadleaf plantain's blade past its stalk: (along the blade, width), broadest a little past halfway, the tip blunt
_PLANTAIN_BLADE = ((0.0, 0.3), (0.08, 0.5), (0.18, 0.72), (0.3, 0.88), (0.42, 0.97), (0.54, 1.0), (0.66, 0.96),
                   (0.77, 0.86), (0.86, 0.7), (0.93, 0.5), (0.98, 0.25))


def _dandelion_rows(nr):
    """A dandelion leaf's rows for _leaf: narrow at the root and broadest in its outer third (oblanceolate), 3-6 teeth
    of uneven spacing and size pointing back toward the root, cut deep on one leaf and shallow on the next (lawn
    dandelions often barely toothed), and a large triangular end lobe. A tooth's point is drawn back by a share of
    its lobe that shrinks with how shallow the cut is, and its midrib vertex sits 0.3 of the way along the lobe, so
    the quad from the sinus to the point never folds over: 0.3 (wp - ws) > ws x back holds for every depth here."""
    def envelope(at):
        return 0.22 + 0.78 * float(_smoothstep(0.05, 0.85, at))
    depth = nr.uniform(0.25, 0.7)
    lobes = int(nr.integers(3, 7))
    end = nr.uniform(0.72, 0.82)
    edges = np.linspace(0.14, end, lobes + 1)
    edges[1:-1] += nr.uniform(-0.3, 0.3, lobes - 1) * (edges[1] - edges[0])
    rows = [(0.0, 0.0, 0.06), (0.07, 0.07, 0.1)]

    def sinus(at):
        return max(0.1, envelope(at) * (1.0 - depth))
    for j in range(lobes):
        s0, s1 = float(edges[j]), float(edges[j + 1])
        span = s1 - s0
        wp = envelope(s0 + 0.3 * span) * nr.uniform(0.85, 1.1)
        ws0, ws1 = min(sinus(s0), 0.8 * wp), min(sinus(s1), 0.8 * wp)
        back = depth * nr.uniform(0.05, 0.14) * span
        rows.append((s0, s0, ws0))
        rows.append((s0 + 0.3 * span, s0 - back, wp))
        e2 = s0 + 0.62 * span
        # the long edge from the point to the next sinus, a little hollow
        rows.append((e2, e2, 0.93 * (wp + (ws1 - wp) * (e2 - (s0 - back)) / (s1 - (s0 - back)))))
    tip = 1.0 - end
    rows += [(end, end, min(sinus(end), 0.7)), (end + 0.08 * tip, end + 0.02 * tip, nr.uniform(0.8, 0.95)),
             (end + 0.45 * tip, end + 0.45 * tip, 0.6), (end + 0.8 * tip, end + 0.8 * tip, 0.3)]
    return rows


def _rosette(geo, nr, H, tall, kind, cx, cy, radius, tone, slot):
    """A plantain's or a dandelion's rosette pressed flat in a lawn: leaves on a spiral (137.5 degrees), the outer and
    older ones longest and lying on the ground, the inner ones shorter and rising (more as the grass round them is
    taller, reaching for light). Plantain: broad oval blades on channelled stalks, ribbed and a little wavy;
    dandelion: long leaves with teeth pointing back toward the root, a deep midrib."""
    n = int(nr.integers(5, 10)) if kind == "plantain" else int(nr.integers(7, 15))
    golden = math.radians(137.5)
    a0 = nr.uniform(0.0, 2.0 * math.pi)
    for k in range(n):
        inner = k / max(1, n - 1)
        a = a0 + k * golden + nr.uniform(-0.15, 0.15)
        out = np.array((math.cos(a), math.sin(a), 0.0))
        rise = math.radians(6.0 + 24.0 * tall + 26.0 * inner + nr.uniform(-5.0, 5.0))
        length = radius * (1.0 - 0.45 * inner) * nr.uniform(0.85, 1.05)
        d = out * math.cos(rise) + np.array((0.0, 0.0, math.sin(rise)))
        spine = _arc((cx, cy, 0.002), d, length, nr.uniform(0.2, 0.45), 10)
        # lying ON the ground, each leaf a little over the older one under it
        spine[:, 2] = np.maximum(spine[:, 2], 0.002 + 0.0008 * k + 0.002 * np.linspace(0.0, 1.0, len(spine)))
        side0 = np.array((-math.sin(a), math.cos(a), 0.0))
        r, b = float(np.clip(tone + nr.normal(0.0, 0.06), 0.0, 1.0)), float(nr.random())
        if kind == "plantain":
            # a broad, channelled stalk widening into the blade (a thin stalk under a round blade reads as a spoon),
            # the blade broadly oval with a blunt tip
            pf = nr.uniform(0.2, 0.35)
            rows = [(0.0, 0.0, 0.15), (0.5 * pf, 0.5 * pf, 0.17), (pf, pf, 0.24)]
            for q, w in _PLANTAIN_BLADE:
                at = pf + q * (1.0 - pf)
                rows.append((at, at, w))
            v, tris, uv, al = _leaf(spine, side0, rows, length * (1.0 - pf) * nr.uniform(0.55, 0.68), fold=0.18,
                                    cup=0.16, wave=0.1, waves=2.5, phase=nr.random())
        else:
            v, tris, uv, al = _leaf(spine, side0, _dandelion_rows(nr), length * nr.uniform(0.17, 0.25), fold=0.3,
                                    cup=0.05, wave=0.06, waves=3.0, phase=nr.random())
        geo.add(v, tris, uv, _col_leaf(al, r, 0.0, 0.2, b), slot)
    return n


def _turf_weeds(geo, wr, H, plan, flowers, slots):
    """The weeds _plan_weeds placed, each into its slot (`slots` maps clover, plantain, dandelion and flowers to a
    material index). White clover: about 1,200 leaves a square meter in a patch's middle, thinning to its edge; in
    flower (`flowers`) a white round head a meter of patch has up to 70 of, on a stalk above the leaves. A dandelion
    rosette in flower carries 1-3 yellow heads, each a dome of florets over green bracts turned down."""
    nr = np.random.default_rng(wr.randint(0, 2 ** 31 - 1))
    tall = plan["tall"]
    out = {"clover": {"patches": len(plan["clover"]), "leaves": 0, "heads": 0},
           "plantain": {"rosettes": 0, "leaves": 0},
           "dandelion": {"rosettes": 0, "leaves": 0, "heads": 0}}
    for cx, cy, rx, ry, ang, tone in plan["clover"]:
        ca, sa = math.cos(ang), math.sin(ang)
        for _ in range(int(nr.poisson(math.pi * rx * ry * 1200.0))):
            for _t in range(8):
                u, v = nr.uniform(-1.0, 1.0, 2)
                d2 = u * u + v * v
                if d2 <= 1.0 and nr.random() < 1.0 - d2 ** 1.5:
                    break
            x, y = cx + u * rx * ca - v * ry * sa, cy + u * rx * sa + v * ry * ca
            _clover_leaf(geo, nr, H, x, y, tone, slots["clover"])
            out["clover"]["leaves"] += 1
        if flowers > 0 and "flowers" in slots:
            for _ in range(int(nr.poisson(math.pi * rx * ry * 70.0 * flowers))):
                u, v = nr.uniform(-0.8, 0.8, 2)
                x, y = cx + u * rx * ca - v * ry * sa, cy + u * rx * sa + v * ry * ca
                b = float(nr.random())
                top, up = _flower_stalk(geo, nr, (x, y, -0.004), H * nr.uniform(0.75, 1.1) + 0.015, 0.0016,
                                        float(np.clip(tone, 0.0, 1.0)), b, slots["clover"])
                rad = nr.uniform(0.007, 0.011)
                # a ball of small florets standing out all round, the lowest turned down
                _flower_head(geo, nr, top, up, rad, ((5, 10, 0.8, 0.0018), (9, 40, 1.0, 0.0018), (11, 70, 1.0, 0.0018),
                                                     (11, 100, 1.0, 0.0018), (8, 130, 0.9, 0.0017)),
                             0.03, b, slots["flowers"])
                out["clover"]["heads"] += 1
    for kind, cx, cy, radius, tone in plan["rosettes"]:
        made = _rosette(geo, nr, H, tall, kind, cx, cy, radius, tone, slots[kind])
        out[kind]["rosettes"] += 1
        out[kind]["leaves"] += made
        if kind == "dandelion" and flowers > 0 and "flowers" in slots and nr.random() < flowers:
            for _ in range(int(nr.integers(1, 4))):
                b = float(nr.random())
                base = (cx + nr.normal(0.0, 0.01), cy + nr.normal(0.0, 0.01), 0.0)
                top, up = _flower_stalk(geo, nr, base, H * nr.uniform(0.4, 1.0) + 0.03, 0.003, tone, b, slots[kind])
                up = _unit_rows(up + np.array((0.0, 0.0, 1.5)))
                rad = nr.uniform(0.014, 0.02)
                # the bracts under the head, turned down, green with the leaves
                _flower_head(geo, nr, top, up, rad * 0.7, ((10, 120, 1.0, 0.0028),), tone, b, slots[kind])
                # a dense dome of fine florets, the outer ones lying out flat and the middle ones short and upright
                _flower_head(geo, nr, top + up * 0.002, up, rad, ((26, 82, 1.0, 0.0024), (22, 62, 0.85, 0.0024),
                                                                  (16, 40, 0.65, 0.0022), (8, 15, 0.4, 0.002)),
                             0.95, b, slots["flowers"])
                out["dandelion"]["heads"] += 1
    return out


# =====================================================================================================================
# clipped hedges - build_plant hedge
# =====================================================================================================================

_HEDGE_ENDS = ("none", "start", "end", "both")
# A hedge leaf's outline for ops_tree's _outline_leaf: the upper half from beside the tip back toward the stalk. Privet's
# (4-6 cm) is an ellipse with a pointed tip and a wedge base, about two and a half times as long as wide; box's (1.5-2.5
# cm) a blunt oval about half as wide as long. A leafLength between the two blends them, fold and droop too.
_HEDGE_LEAF_LONG = ((0.86, 0.09), (0.64, 0.19), (0.38, 0.205), (0.14, 0.12))
_HEDGE_LEAF_SHORT = ((0.93, 0.14), (0.72, 0.27), (0.42, 0.30), (0.15, 0.18))
_HEDGE_FOLD = (0.16, 0.26)
_HEDGE_DROOP = (0.03, 0.07)
# the fallen leaves' preview colors (linear): brown to straw. A game sets its own material on <name>_Litter.
_HEDGE_LITTER_COLORS = ((0.16, 0.10, 0.03), (0.36, 0.24, 0.07))
# leaves per square meter of clipped face come from the leaf's own area: a sheared face is about two leaves deep
# wherever you look at it, so the count is the face's area twice over in leaves
_HEDGE_COVER = 2.0


def _hedge_leaf(leaf_len):
    """The leaf for a hedge whose leaves are `leaf_len` meters long, from box's oval to privet's ellipse. Returns the
    template (verts, tris, uvs) of a leaf one unit long and its area in that unit."""
    t = min(1.0, max(0.0, (leaf_len - 0.02) / 0.025))
    upper = [tuple(a + (b - a) * t for a, b in zip(s, l)) for s, l in zip(_HEDGE_LEAF_SHORT, _HEDGE_LEAF_LONG)]
    fold = _HEDGE_FOLD[0] + (_HEDGE_FOLD[1] - _HEDGE_FOLD[0]) * t
    droop = _HEDGE_DROOP[0] + (_HEDGE_DROOP[1] - _HEDGE_DROOP[0]) * t
    tv, tt, tu = _outline_leaf(upper, (0.5,), fold, droop)
    a, b, c = tv[tt[:, 0], :2], tv[tt[:, 1], :2], tv[tt[:, 2], :2]
    area = float(0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])).sum())
    return tv, tt, tu, area


def _convex_sdf(py, pz, poly):
    """Signed distance from points (py, pz) to a convex polygon given counter-clockwise: exact inside and out."""
    inside = np.ones(py.shape, dtype=bool)
    best = np.full(py.shape, np.inf)
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        ey, ez = b[0] - a[0], b[1] - a[1]
        wy, wz = py - a[0], pz - a[1]
        t = np.clip((wy * ey + wz * ez) / (ey * ey + ez * ez), 0.0, 1.0)
        best = np.minimum(best, (wy - ey * t) ** 2 + (wz - ez * t) ** 2)
        inside &= ey * wz - ez * wy >= 0.0
    d = np.sqrt(best)
    return np.where(inside, -d, d)


class _HedgeShape:
    """The clipped outline of a hedge as a signed distance in meters (negative inside). Across it a trapezoid: the foot
    as thick as asked and the top a tenth narrower, as a hedge is cut so light reaches its lower leaves, the top's two
    edges rounded as shears leave them. Along X it runs on for ever, or stops at a clipped end cut square with rounded
    corners. The foot draws in a few centimeters toward the ground, where a hedge goes thin and leggy in its own shade.
    The shears' unevenness is a centimeter or two of bumps whose waves are whole cycles along the piece and fade to
    nothing at its two ends, so a piece meets itself and any other piece with no step."""

    def __init__(self, rng, H, Wd, Ln, ends):
        self.H, self.Ln = H, Ln
        hw = 0.5 * Wd
        thw = 0.9 * hw
        self.hw, self.thw = hw, thw
        self.rc = rc = min(0.12, 0.35 * thw, 0.25 * H)
        lean = hw - thw
        s = math.hypot(H, lean)
        # the outline's sides as outward normals (y, z) and offsets - bottom (far under the ground), right, top, left -
        # moved in by the rounding radius; their corners, counter-clockwise, are the outline the radius is grown from
        sides = [((0.0, -1.0), 1.0), ((H / s, lean / s), H * hw / s), ((0.0, 1.0), H), ((-H / s, lean / s), H * hw / s)]
        sides = [(n, c - rc) for n, c in sides]
        corners = []
        for i in range(4):
            (n1, c1), (n2, c2) = sides[i - 1], sides[i]
            det = n1[0] * n2[1] - n1[1] * n2[0]
            corners.append(((c1 * n2[1] - c2 * n1[1]) / det, (n1[0] * c2 - n2[0] * c1) / det))
        self.poly = np.array(corners)
        self.cap_start = ends in ("start", "both")
        self.cap_end = ends in ("end", "both")
        self.foot_h = min(0.3, max(0.08, 0.2 * H))
        self.foot = min(0.05, 0.15 * hw)
        reach = max(1, int(Ln / 0.15))
        picks = [i for i in range(1, reach + 1) if 0.15 <= Ln / i <= 0.7] or [1]
        k = 2.0 * math.pi / 0.4
        self.waves = np.array([(2.0 * math.pi * rng.choice(picks) / Ln, rng.gauss(0.0, k), rng.gauss(0.0, k),
                                rng.uniform(0.0, 2.0 * math.pi)) for _ in range(10)])
        self.amp = 0.012 / math.sqrt(0.5 * len(self.waves))

    def half_width(self, z):
        return self.hw - (self.hw - self.thw) * min(1.0, max(0.0, z / self.H))

    def bumps(self, x, y, z):
        u = x + 0.5 * self.Ln
        f = np.zeros_like(x)
        for kx, ky, kz, ph in self.waves:
            f += np.sin(kx * u + ky * y + kz * z + ph)
        return self.amp * f * np.sin(math.pi * u / self.Ln) ** 2

    def __call__(self, x, y, z):
        q = _convex_sdf(y, z, self.poly)
        dx = None
        if self.cap_start:
            dx = (-0.5 * self.Ln + self.rc) - x
        if self.cap_end:
            e = x - (0.5 * self.Ln - self.rc)
            dx = e if dx is None else np.maximum(dx, e)
        if dx is None:
            d = q - self.rc
        else:
            d = np.minimum(np.maximum(q, dx), 0.0) + np.hypot(np.maximum(q, 0.0), np.maximum(dx, 0.0)) - self.rc
        d = d + self.foot * (1.0 - _smoothstep(0.0, self.foot_h, z))
        return d - self.bumps(x, y, z)

    def normal(self, x, y, z, e=0.002):
        g = np.stack((self(x + e, y, z) - self(x - e, y, z), self(x, y + e, z) - self(x, y - e, z),
                      self(x, y, z + e) - self(x, y, z - e)), axis=1)
        return _unit_rows(g)

    def surface(self, nr, want, xlo, xhi, keep=None):
        """`want` points spread evenly over the clipped face between xlo and xhi, above the first 2.5 cm of the
        ground, with their outward normals: drawn in a thin band round the face and pulled onto it. `keep(x, y, z)`
        (0-1) thins them where it is low."""
        out_p, out_n, got = [], [], 0
        for _ in range(60):
            if got >= want:
                break
            m = int(min(400000, max(20000, (want - got) * 30)))
            x = nr.uniform(xlo, xhi, m)
            y = nr.uniform(-self.hw - 0.06, self.hw + 0.06, m)
            z = nr.uniform(0.0, self.H + 0.06, m)
            d = self(x, y, z)
            ok = np.abs(d) < 0.02
            x, y, z, d = x[ok], y[ok], z[ok], d[ok]
            for _step in range(2):
                n = self.normal(x, y, z)
                x, y, z = x - n[:, 0] * d, y - n[:, 1] * d, z - n[:, 2] * d
                d = self(x, y, z)
            n = self.normal(x, y, z)
            ok = (z > 0.025) & (x >= xlo) & (x < xhi)
            if keep is not None:
                ok &= nr.random(len(x)) < keep(x, y, z)
            out_p.append(np.stack((x, y, z), axis=1)[ok])
            out_n.append(n[ok])
            got += int(ok.sum())
        if not out_p:
            return np.zeros((0, 3)), np.zeros((0, 3))
        return np.concatenate(out_p)[:want], np.concatenate(out_n)[:want]

    def area(self):
        return _hedge_area(self.H, 2.0 * self.hw, self.Ln, int(self.cap_start) + int(self.cap_end))


def _hedge_area(H, Wd, Ln, caps):
    """A hedge's clipped face in square meters: both sides, the top and `caps` clipped ends (_HedgeShape's outline,
    its rounding left out)."""
    hw, thw = 0.5 * Wd, 0.45 * Wd
    return (2.0 * math.hypot(H, hw - thw) + 2.0 * thw) * Ln + (hw + thw) * H * caps


def _hedge_count(H, Wd, Ln, caps, leaf_len):
    """The leaves a hedge carries when count is not given: its face's area in leaves, _HEDGE_COVER times over."""
    return int(round(_hedge_area(H, Wd, Ln, caps) * _HEDGE_COVER / (_hedge_leaf(leaf_len)[3] * leaf_len * leaf_len)))


def _hedge_tubes(geo, tubes, shape):
    """The stems as 5- and 3-sided tubes (ops_tree's, so bark tiles at the trees' scale), slot 0. Vertex alpha is how
    open each corner lies, from its depth under the clipped face."""
    for pts, radii, sides, w0, w1, phase in tubes:
        verts, quads, quad_uv, quad_col, tris, tri_uv, tri_col = _tube(pts, radii, sides, None, 0.0, w0, w1, phase)
        split = np.array((0, 1, 2, 0, 2, 3))
        # each quad as two triangles, in the corner order of `split`, so the per-corner UVs and colors follow
        all_tris = np.concatenate((quads[:, split].reshape(-1, 3), tris))
        uv = np.concatenate((quad_uv[:, split].reshape(-1, 2), tri_uv.reshape(-1, 2)))
        col = np.concatenate((quad_col[:, split].reshape(-1, 4), tri_col.reshape(-1, 4)))
        corners = verts[all_tris.reshape(-1)]
        col[:, 3] = _hedge_open(shape, corners)
        geo.add_bulk(verts, all_tris, uv, col, 0)


def _hedge_open(shape, p, depth=None):
    """How open a point of the hedge lies, 0-1: 1 on the clipped face and outside it, falling toward 0.1 twenty
    centimeters in, and less again at the foot, which the hedge's own bulk hides from half the sky."""
    if depth is None:
        depth = -shape(p[:, 0], p[:, 1], p[:, 2])
    a = np.clip(1.0 - np.maximum(depth, 0.0) / 0.17, 0.1, 1.0) ** 1.3
    return a * (0.65 + 0.35 * _smoothstep(0.0, 1.5 * shape.foot_h, p[:, 2]))


def _hedge_leaves(geo, tmpl, P, T, N, length, widen, col, slot):
    """Many leaves at once from one template: leaf k's stalk at P[k], its tip along T[k], its face toward N[k] (T and N
    orthonormal), `length` long and `widen` times as wide as the template. col (n, 4) per leaf; G rises a little toward
    each tip."""
    tv, tt, tu = tmpl
    S = np.cross(N, T)
    n, k = len(P), len(tv)
    if n == 0:
        return
    V = (P[:, None, :] + length[:, None, None] * (tv[None, :, 0, None] * T[:, None, :]
                                                + (tv[None, :, 1, None] * widen[:, None, None]) * S[:, None, :]
                                                + tv[None, :, 2, None] * N[:, None, :]))
    tris = (tt[None, :, :] + (np.arange(n) * k)[:, None, None]).reshape(-1, 3)
    corner = tt.reshape(-1)
    c = np.repeat(col, len(corner), axis=0)
    c[:, 1] = np.clip(c[:, 1] + np.tile(0.1 * tv[corner, 0], n), 0.0, 1.0)
    geo.add_bulk(V.reshape(-1, 3), tris, np.tile(tu[corner], (n, 1)), c, slot)


def _plant_hedge(geo, rng, H, Ln, Wd, count, ends, shag, leaf_len, report):
    """A clipped hedge, Ln long along X and centered on the pivot, Wd thick at the foot and H tall, as a privet or box
    hedge is: a row (two staggered rows past 45 cm thick) of plants a step of about 30 cm apart, each 3-5 stems from
    the ground forking into branches and twigs that reach the leaves, and the leaves in a shell a few centimeters deep
    under the sheared face, turned out and up to the light, with the inside dark and open (vertex alpha) and the foot
    thinning out. `shag` is the growth since the last cut: new shoots standing out of the face with their leaves in
    opposite pairs, each pair a quarter turn on from the last, upright on the top and arching out of the sides. Fallen
    leaves lie on the ground along the foot (their own slot). Unless `ends` clips an end, the piece TILES along X: its
    plants stand a whole number of steps apart along it, its twigs reach past its ends as the next piece's reach into
    it, and the shears' bumps fade out at its ends, so copies laid end to end (and any piece of another seed) meet with
    no seam."""
    nr = np.random.default_rng(rng.randint(0, 2 ** 31 - 1))
    shape = _HedgeShape(rng, H, Wd, Ln, ends)
    tv, tt, tu, _area = _hedge_leaf(leaf_len)
    tmpl = (tv, tt, tu)
    half = 0.5 * Ln
    area = shape.area()
    if count is None:
        count = _hedge_count(H, Wd, Ln, int(shape.cap_start) + int(shape.cap_end), leaf_len)
    # the shell's points the twigs reach for, past the piece's ends too (a clipped end has no face past it)
    pool, pool_n = shape.surface(nr, int(3000 * (Ln + 0.8)), -half - 0.4, half + 0.4)

    # THE PLANTS AND THEIR STEMS
    per_row = max(1, int(round(Ln / 0.3)))
    step = Ln / per_row
    rows = (-0.17 * Wd, 0.17 * Wd) if Wd >= 0.45 else (0.0,)
    lo = -half + 0.12 if shape.cap_start else -math.inf
    hi = half - 0.12 if shape.cap_end else math.inf
    tubes, n_stems, n_branches, n_twigs = [], 0, 0, 0

    def bend(a, b, lift, wobble, n):
        a, b = np.asarray(a, float), np.asarray(b, float)
        c = 0.5 * (a + b) + np.array((rng.gauss(0.0, wobble), rng.gauss(0.0, wobble), lift))
        t = np.linspace(0.0, 1.0, n)[:, None]
        return (1.0 - t) ** 2 * a + 2.0 * t * (1.0 - t) * c + t * t * b

    def at(pts, f):
        s = f * (len(pts) - 1)
        i = min(len(pts) - 2, int(s))
        return pts[i] + (pts[i + 1] - pts[i]) * (s - i)

    for r, y0 in enumerate(rows):
        off = 0.5 if len(rows) == 1 else 0.25 + 0.5 * r
        for k in range(per_row):
            x0 = min(max(-half + (k + off + rng.uniform(-0.15, 0.15)) * step, lo), hi)
            phase = rng.random()
            for _s in range(rng.randint(3, 5)):
                root = (x0 + rng.gauss(0.0, 0.015), y0 + rng.gauss(0.0, 0.015), -0.04)
                tgt = (x0, 0.0, 0.5 * H)
                for _try in range(8):
                    zt = H * rng.uniform(0.4, 0.68)
                    inner = max(0.02, shape.half_width(zt) - 0.1)
                    cand = (min(max(x0 + rng.gauss(0.0, 0.1), lo), hi), rng.uniform(-0.75, 0.75) * inner, zt)
                    if float(shape(*(np.array([v]) for v in cand))[0]) < -0.05:
                        tgt = cand
                        break
                pts = bend(root, tgt, 0.06 * H, 0.03, 6)
                r0 = rng.uniform(0.009, 0.015)
                tubes.append((pts, np.linspace(r0, 0.6 * r0, 6), 6, 0.0, 0.03, phase))
                n_stems += 1
                # branches from the stem's upper half to the shell, just inside the leaves
                for _b in range(rng.randint(2, 4)):
                    f = rng.uniform(0.45, 1.0)
                    s = at(pts, f)
                    idx = np.flatnonzero((np.abs(pool[:, 0] - s[0]) < 0.35) & (pool[:, 2] > s[2] - 0.05))
                    if not len(idx):
                        continue
                    j = idx[int(nr.integers(len(idx)))]
                    end = pool[j] - pool_n[j] * rng.uniform(0.06, 0.11)
                    bp = bend(s, end, 0.04, 0.02, 5)
                    rb = 0.55 * r0 * (1.0 - 0.4 * f)
                    tubes.append((bp, np.linspace(rb, 0.0025, 5), 5, 0.03, 0.08, phase))
                    n_branches += 1
                    # twigs from the branch's outer half out among the leaves
                    near = np.flatnonzero(np.sum((pool - end) ** 2, axis=1) < 0.18 ** 2)
                    for _t in range(rng.randint(2, 4)):
                        if not len(near):
                            break
                        j2 = near[int(nr.integers(len(near)))]
                        e2 = pool[j2] - pool_n[j2] * rng.uniform(0.01, 0.035)
                        tubes.append((bend(at(bp, rng.uniform(0.5, 0.95)), e2, 0.01, 0.01, 4),
                                      np.linspace(0.0022, 0.0008, 4), 3, 0.08, 0.18, phase))
                        n_twigs += 1

    # THE INSIDE: a sheared hedge is a thicket of fine twigs under its leaves, the ones the shears cut back and the
    # ones shaded out, many dead with brown leaves caught in them. Without it the shell is hollow, and through two gaps
    # in it the sky shows through the hedge. Short crooked twigs, 4 to 20 cm under the face, about 900 a square meter
    # of face; their caught leaves go to the litter slot.
    n_inner = int(round(area * 900.0))
    ip, inn = shape.surface(nr, n_inner, -half, half)
    n_inner = len(ip)
    ia = ip - inn * (0.04 + 0.16 * nr.random(n_inner) ** 1.5)[:, None]
    idir = _unit_rows(nr.normal(0.0, 1.0, (n_inner, 3)))
    ilen = nr.uniform(0.05, 0.14, n_inner)
    # every tip stays 1.5 cm under the face: a twig that would reach out turns back, and if that fails too, is shortened
    ib = ia + idir * ilen[:, None]
    out = shape(ib[:, 0], ib[:, 1], ib[:, 2]) > -0.015
    idir[out] = -idir[out]
    ib = ia + idir * ilen[:, None]
    for _halve in range(4):
        out = shape(ib[:, 0], ib[:, 1], ib[:, 2]) > -0.015
        if not out.any():
            break
        ilen[out] *= 0.5
        ib = ia + idir * ilen[:, None]
    caught = []
    for i in range(n_inner):
        tubes.append((bend(ia[i], ib[i], 0.0, 0.25 * ilen[i], 3), np.array((0.0018, 0.0012, 0.0006)), 3, 0.05, 0.1,
                      rng.random()))
        if rng.random() < 0.05:
            caught.append((ib[i], idir[i]))

    # THE NEW SHOOTS, standing out of the face by up to a third of a meter at shag 1
    leaves = {"P": [], "T": [], "N": [], "l": [], "w": [], "col": []}
    n_shoots = int(round(shag * area * 30.0))
    shoot_leaves = 0
    if n_shoots:
        sp, sn = shape.surface(nr, n_shoots, -half, half)
        for i in range(len(sp)):
            up = max(0.0, float(sn[i, 2]))
            d = _unit_rows(sn[i] * 0.7 + np.array((0.0, 0.0, 0.5 + 0.8 * up))
                           + np.array([rng.gauss(0.0, 0.25) for _ in range(3)]))
            length = shag * rng.uniform(0.06, 0.32)
            depth = rng.uniform(0.02, 0.05)
            pts = _arc(sp[i] - sn[i] * depth, d, length + depth, rng.uniform(0.05, 0.25), 4)
            ph = rng.random()
            tubes.append((pts, np.linspace(0.0018, 0.0007, len(pts)), 3, 0.15, 1.0, ph))
            nodes = max(1, int(length / (0.75 * leaf_len)))
            f0 = depth / (length + depth)
            light = _unit_rows(np.array((0.0, 0.0, 1.0)) + sn[i] * 0.6)
            u0 = np.cross(d, light)
            u0 = _unit_rows(u0) if np.linalg.norm(u0) > 1e-6 else np.array((1.0, 0.0, 0.0))
            for j in range(nodes):
                g = (j + 0.6) / nodes
                f = f0 + (1.0 - f0) * g
                p = at(pts, f)
                t = _unit_rows(pts[min(len(pts) - 1, int(f * (len(pts) - 1)) + 1)] - pts[int(f * (len(pts) - 1))]
                               if f < 1.0 else pts[-1] - pts[-2])
                side = _rotate_vec(u0, t, j * 0.5 * math.pi + rng.uniform(-0.3, 0.3))
                for sgn in (-1.0, 1.0):
                    a = math.radians(rng.uniform(45.0, 65.0))
                    D = _unit_rows(t * math.cos(a) + side * sgn * math.sin(a))
                    Nv = light - D * float(D @ light)
                    Nv = _unit_rows(Nv) if np.linalg.norm(Nv) > 1e-6 else _unit_rows(np.cross(D, side))
                    leaves["P"].append(p)
                    leaves["T"].append(D)
                    leaves["N"].append(Nv)
                    leaves["l"].append(leaf_len * rng.uniform(0.8, 1.1) * (1.0 - 0.4 * g ** 3))
                    leaves["w"].append(rng.uniform(0.9, 1.1))
                    # new growth is paler: R toward 1, the lighter of the two tints
                    leaves["col"].append((rng.uniform(0.75, 1.0), 0.3 + 0.6 * g, ph, 1.0))
                    shoot_leaves += 1

    # THE SHEARED FACE'S LEAVES: thinning out at the foot and in a few soft thin patches, a few centimeters deep,
    # facing out and up, each its own way; the deeper ones darker in the alpha and a little larger, as shade leaves are
    n_face = max(0, count - shoot_leaves)
    patch_seed = rng.randint(0, 2 ** 30)

    def keep(x, y, z):
        u = x + half
        pr = random.Random(patch_seed)
        f = np.zeros_like(x)
        for _w in range(6):
            i = pr.randint(1, max(1, int(Ln / 0.25)))
            f += np.sin(2.0 * math.pi * i * u / Ln + pr.gauss(0.0, 9.0) * z + pr.uniform(0.0, 6.283))
        f = f / math.sqrt(3.0) * np.sin(math.pi * u / Ln) ** 2
        return (0.1 + 0.9 * _smoothstep(0.02, shape.foot_h, z)) * np.clip(0.8 + 0.2 * f, 0.4, 1.0)

    fp, fn = shape.surface(nr, n_face, -half, half, keep)
    n_face = len(fp)
    depth = np.minimum(nr.exponential(0.035, n_face), 0.18) - shag * nr.uniform(0.0, 0.05, n_face)
    Nf = _unit_rows(fn + nr.normal(0.0, 0.5, (n_face, 3)) + np.array((0.0, 0.0, 0.35)))
    Tf = nr.normal(0.0, 1.0, (n_face, 3)) + fn * 0.25
    Tf = _unit_rows(Tf - Nf * np.sum(Tf * Nf, axis=1)[:, None])
    lf = leaf_len * nr.uniform(0.7, 1.15, n_face) * (1.0 + 0.25 * np.clip(depth / 0.1, 0.0, 1.0))
    centre = fp - fn * depth[:, None]
    Pf = centre - Tf * (0.45 * lf)[:, None]
    # tone in soft patches a hand or two across (R, which picks between the two tints), with each leaf its own way in
    # them, and paler toward the top, where the newest leaves are
    tone = 0.5 + 0.5 * np.tanh(1.5 * shape.bumps(fp[:, 0], fp[:, 1] * 1.7, fp[:, 2] * 1.7) / shape.amp / 3.0)
    tone = np.clip(0.5 * nr.random(n_face) + 0.35 * tone + 0.15 * fp[:, 2] / H, 0.0, 1.0)
    colf = np.stack((tone, 0.18 + 0.2 * np.clip(1.0 - depth / 0.12, 0.0, 1.0), nr.random(n_face),
                     _hedge_open(shape, centre, depth)), axis=1)

    # FALLEN LEAVES along the foot, under its edge and a little out from it
    n_litter = int(round(Ln * 25.0))
    lx = nr.uniform(-half, half, n_litter)
    if shape.cap_start or shape.cap_end:
        lx = np.clip(lx, lo - 0.05, hi + 0.05)
    side = np.where(nr.random(n_litter) < 0.5, -1.0, 1.0)
    ly = side * (shape.hw + nr.normal(-0.05, 0.08, n_litter))
    lz = 0.004 + nr.uniform(0.0, 0.006, n_litter)
    Nl = _unit_rows(np.stack((nr.normal(0.0, 0.35, n_litter), nr.normal(0.0, 0.35, n_litter), np.ones(n_litter)), 1))
    Tl = np.stack((nr.normal(0.0, 1.0, n_litter), nr.normal(0.0, 1.0, n_litter), np.zeros(n_litter)), axis=1)
    Tl = _unit_rows(Tl - Nl * np.sum(Tl * Nl, axis=1)[:, None])
    ll = leaf_len * nr.uniform(0.75, 1.1, n_litter)
    Pl = np.stack((lx, ly, lz), axis=1) - Tl * (0.5 * ll)[:, None]
    coll = np.stack((nr.random(n_litter), np.zeros(n_litter), nr.random(n_litter), np.full(n_litter, 0.85)), axis=1)
    if caught:
        cp = np.array([c[0] for c in caught])
        cn = _unit_rows(nr.normal(0.0, 1.0, (len(caught), 3)))
        ct = _unit_rows(np.array([c[1] for c in caught]) - cn * np.sum(np.array([c[1] for c in caught]) * cn, axis=1)[:, None])
        cl = leaf_len * nr.uniform(0.7, 1.0, len(caught))
        cc = np.stack((nr.random(len(caught)), np.full(len(caught), 0.1), nr.random(len(caught)),
                       _hedge_open(shape, cp)), axis=1)
        Pl, Tl, Nl = np.concatenate((Pl, cp)), np.concatenate((Tl, ct)), np.concatenate((Nl, cn))
        ll, coll = np.concatenate((ll, cl)), np.concatenate((coll, cc))

    # in slot order: stems, then the leaves (the face's, then the shoots'), then the litter
    _hedge_tubes(geo, tubes, shape)
    _hedge_leaves(geo, tmpl, Pf, Tf, Nf, lf, nr.uniform(0.9, 1.1, n_face), colf, 1)
    if shoot_leaves:
        ps = np.array(leaves["P"])
        cols = np.array(leaves["col"])
        cols[:, 3] = _hedge_open(shape, ps)
        _hedge_leaves(geo, tmpl, ps, np.array(leaves["T"]), np.array(leaves["N"]), np.array(leaves["l"]),
                      np.array(leaves["w"]), cols, 1)
    _hedge_leaves(geo, tmpl, Pl, Tl, Nl, ll, nr.uniform(0.9, 1.1, len(Pl)), coll, 2)
    report.update({
        "lengthCm": round(Ln * 100.0, 2), "clipHeightCm": round(H * 100.0, 2), "thicknessCm": round(Wd * 100.0, 2),
        "ends": ends, "shag": shag, "leafLengthCm": round(leaf_len * 100.0, 2),
        "faceAreaM2": round(area, 3),
        "plants": len(rows) * per_row, "stems": n_stems, "branches": n_branches, "twigs": n_twigs,
        "innerTwigs": n_inner, "shoots": n_shoots,
        "leaves": {"face": n_face, "shoots": shoot_leaves, "total": n_face + shoot_leaves, "litter": n_litter,
                   "caught": len(caught)},
        "trianglesPerLeaf": len(tt),
    })
    return n_face + shoot_leaves


def op_build_plant(params):
    """Build one piece of ground cover - a grass tuft, a fern, a broadleaf plant, a bush, a tile of lawn or a piece of
    clipped hedge - as ONE mesh object of
    real geometry blades and leaves, its pivot at the base, in centimeters asked for. The same seed and parameters give
    the same plant on a given Blender. No collision: ground cover is walked through.

    params:
      name (str)            default "Plant"
      kind (str)            grass (default), fern, plant, bush, turf (a square tile of lawn that tiles), hedge (a
                            piece of clipped hedge that tiles along X)
      seed (int)            0 or more, default 1
      height (cm)           2 to 600, the top of the plant; defaults grass 35, fern 55, plant 30, bush 110
      width (cm)            the spread - the wider plan side - 2 to 1000; defaults grass 60, fern 130, plant 60,
                            bush 130. The plant is fitted to both, centered on the pivot; the defaults are each kind's
                            grown proportions, and a height and width far from them stretch the form
      count (int)           blades (grass 1-400, default 60), fronds (fern 1-40, default 10), leaves (plant 1-60,
                            default 14; bush 50-20000, default 2000), blades on the whole tile (turf 100-100000,
                            default 3000)
      bare (0-0.9)          turf only: the share of the tile left as bare, soft-edged patches; default 0
      weeds (0-0.6)         turf only: the share of the tile its broadleaf lawn weeds cover; default 0 (none)
      weedMix ([c, p, d])   with weeds: how they split among white clover (patches), broadleaf plantain and dandelion
                            (single flat rosettes), shares of 0 or more; default [0.5, 0.3, 0.2]
      flowers (0-1)         with weeds: how many are in flower - clover heads over a patch, 1-3 yellow heads on that
                            share of the dandelions; default 0
                            For turf, width is the tile's side and height the longest blade; the tile is NOT fitted to
                            them as the other kinds are: its blade roots fill the square exactly and the blades lean
                            over its edges, as they lean over the next tile's. Tiles meet with no seam laid at any
                            quarter turn (turning tiles hides their repeat); a tile with bare patches only at one turn
      length (cm)           hedge only: the piece's run along X, 20-1000, default 100 - the step to lay pieces at
      ends (str)            hedge only: none (default; a middle piece whose leaves and twigs run on past both ends,
                            to tile), start (clipped at -X), end (clipped at +X), both (a hedge on its own)
      shag (0-1)            hedge only: the growth since the last cut - 0 freshly sheared, 1 a summer's shoots up to a
                            third of a meter out of the face; default 0.3
      leafLength (cm)       hedge only: 1-8, default 4 (privet's ellipse); about 2 gives box's small oval leaves
                            For a hedge, height is the clipped top, width the thickness at the foot (the top is cut a
                            tenth narrower) and count the leaves (default: the clipped face's area in leaves, twice
                            over - about 14,000 a meter at the defaults, 10 triangles each). It is NOT fitted: shoots
                            stand out of the clipped outline. Slots: <name>_Bark (stems), _Leaves, _Litter (fallen
                            leaves along the foot); vertex color A is how open each corner lies (an occlusion)
      leafColor, leafColor2 ([r,g,b] linear 0-1)   the two greens vertex color R picks between
      location, rotation ([x,y,z] or {x,y,z}), collection
    """
    reject_unknown(params, BUILD_PLANT_PARAMS, "build_plant", _PLANT_NOTES)
    started = time.perf_counter()
    kind = take(params, "kind", default="grass", kind=str)
    if kind not in _PLANT_KINDS:
        raise MifOpError("kind must be one of %s, got '%s'. NOTHING was created." % (", ".join(_PLANT_KINDS), kind))
    cfg = _PLANT_DEFAULTS[kind]
    name = _check_name(params, "Plant", False)
    seed = _check_seed(params)
    height = take_float(params, "height", default=cfg["height"])
    if not 2.0 <= height <= 600.0:
        raise MifOpError("height is %g cm; a plant here is 2 to 600 cm. NOTHING was created." % height)
    width = take_float(params, "width", default=cfg["width"])
    if not 2.0 <= width <= 1000.0:
        raise MifOpError("width is %g cm; it must be 2 to 1000 cm. NOTHING was created." % width)
    count = take_int(params, "count", default=cfg["count"])
    lo, hi = cfg["range"]
    if count is not None and not lo <= count <= hi:
        raise MifOpError("count is %d; a %s takes %d to %d. NOTHING was created." % (count, kind, lo, hi))
    # A HEDGE'S OWN: its run along X, its clipped ends, the growth since the last cut and its leaf
    hedge_only = [k for k in ("length", "ends", "shag", "leafLength") if params.get(k) is not None]
    if hedge_only and kind != "hedge":
        raise MifOpError("%s %s a hedge's (a clipped hedge's run, ends, growth and leaf); a %s has none. NOTHING was "
                         "created." % (", ".join(hedge_only), "is" if len(hedge_only) == 1 else "are", kind))
    length = take_float(params, "length", default=100.0)
    if not 20.0 <= length <= 1000.0:
        raise MifOpError("length is %g cm; a hedge piece runs 20 to 1000 cm. NOTHING was created." % length)
    ends = take(params, "ends", default="none", kind=str)
    if ends not in _HEDGE_ENDS:
        raise MifOpError("ends must be one of %s, got '%s'. NOTHING was created." % (", ".join(_HEDGE_ENDS), ends))
    shag = take_float(params, "shag", default=0.3)
    if not 0.0 <= shag <= 1.0:
        raise MifOpError("shag is %g; it is a hedge's growth since the last cut, 0 to 1. NOTHING was created." % shag)
    leaf_length = take_float(params, "leafLength", default=4.0)
    if not 1.0 <= leaf_length <= 8.0:
        raise MifOpError("leafLength is %g cm; a hedge's leaves are 1 to 8 cm (privet about 4, box about 2). NOTHING "
                         "was created." % leaf_length)
    if kind == "hedge":
        if not (20.0 <= height and 15.0 <= width <= 400.0):
            raise MifOpError("a hedge is 20 to 600 cm tall and 15 to 400 cm thick; got %g by %g. NOTHING was created."
                             % (height, width))
        if count is None:
            caps = {"none": 0, "start": 1, "end": 1, "both": 2}[ends]
            count = _hedge_count(height / 100.0, width / 100.0, length / 100.0, caps, leaf_length / 100.0)
            if not lo <= count <= hi:
                raise MifOpError("this hedge would carry %d leaves (its clipped face in %g cm leaves, twice over); a "
                                 "hedge takes %d to %d - give count, or a larger leafLength. NOTHING was created."
                                 % (count, leaf_length, lo, hi))
    bare = take_float(params, "bare", default=0.0)
    if not 0.0 <= bare <= 0.9:
        raise MifOpError("bare is %g; it is the share of a turf tile left bare, 0 to 0.9. NOTHING was created." % bare)
    if bare > 0.0 and kind != "turf":
        raise MifOpError("bare is turf's (the share of a lawn tile left bare); a %s has none. NOTHING was created."
                         % kind)
    weeds = take_float(params, "weeds", default=0.0)
    if not 0.0 <= weeds <= 0.6:
        raise MifOpError("weeds is %g; it is the share of a turf tile its lawn weeds cover, 0 to 0.6. NOTHING was "
                         "created." % weeds)
    if weeds > 0.0 and kind != "turf":
        raise MifOpError("weeds is turf's (the lawn weeds among a tile's blades); a %s has none. NOTHING was created."
                         % kind)
    mix = take(params, "weedMix")
    flowers = take_float(params, "flowers", default=0.0)
    if (mix is not None or flowers > 0.0) and weeds <= 0.0:
        raise MifOpError("weedMix and flowers shape a turf tile's weeds; with weeds 0 there are none. NOTHING was "
                         "created.")
    if mix is None:
        mix = _WEED_MIX
    elif (not isinstance(mix, (list, tuple)) or len(mix) != 3
          or not all(isinstance(m, (int, float)) and not isinstance(m, bool) and math.isfinite(m) and m >= 0 for m in mix)
          or sum(mix) <= 0):
        raise MifOpError("weedMix takes [clover, plantain, dandelion] shares, three numbers of 0 or more and not all 0 - "
                         "got %r. NOTHING was created." % (mix,))
    mix = tuple(float(m) for m in mix)
    if not 0.0 <= flowers <= 1.0:
        raise MifOpError("flowers is %g; it is the share of the weeds in flower, 0 to 1. NOTHING was created." % flowers)
    if flowers > 0.0 and mix[0] <= 0.0 and mix[2] <= 0.0:
        raise MifOpError("flowers are the clover's and the dandelions'; this weedMix has neither. NOTHING was created.")
    color = _color3(params, "leafColor", cfg["colors"][0])
    color2 = _color3(params, "leafColor2", cfg["colors"][1])
    try:
        placed = _place_values(params)
    except MifOpError as exc:
        raise MifOpError("%s. NOTHING was created." % str(exc).rstrip("."))
    coll = _resolve_collection(params)
    if np is None:
        raise MifOpError("this Blender's Python has no numpy, which build_plant builds with. NOTHING was created.")

    rng = random.Random(seed)
    geo = _Geo()
    H, Wd = height / 100.0, width / 100.0
    # the weeds' slots after the blades', in a fixed order: each weed of the mix with a share, then the flowers
    weed_slots = {}
    if weeds > 0.0:
        for k, w in zip(_WEED_KINDS, mix):
            if w > 0.0:
                weed_slots[k] = len(weed_slots) + 1
        if flowers > 0.0:
            weed_slots["flowers"] = len(weed_slots) + 1
    weed_report = {}
    hedge_report = {}
    if kind == "turf":
        blades = _plant_turf(geo, rng, H, Wd, count, bare, weeds, mix, flowers, weed_slots, weed_report)
        if not blades:
            raise MifOpError("bare %g left no blade on the tile. NOTHING was created." % bare)
    elif kind == "hedge":
        _plant_hedge(geo, rng, H, length / 100.0, Wd, count, ends, shag, leaf_length / 100.0, hedge_report)
    else:
        {"grass": _plant_grass, "fern": _plant_fern, "plant": _plant_broadleaf, "bush": _plant_bush}[kind](
            geo, rng, H, Wd, count)
    verts = np.array(geo.verts)
    # THE SIZE ASKED FOR IS THE PLANT'S BOUNDS: blades arch, fronds droop and outer leaves lie out, so the grown form
    # lands anywhere from half to twice the height and spread it was grown for (a tuft 32 cm tall and 57 cm across for
    # 35 by 25, measured). It is fitted: the top to `height`, the wider of the two plan sides to `width`.
    # A TURF TILE IS NOT FITTED: its roots fill the square exactly so tiles meet, and its blades lean past the edges.
    # NOR IS A HEDGE: its clipped outline is the height, thickness and length asked, and new shoots stand out of it.
    if kind not in ("turf", "hedge"):
        top = float(verts[:, 2].max())
        span = float(max(np.ptp(verts[:, 0]), np.ptp(verts[:, 1])))
        centre = np.array((0.5 * (verts[:, 0].max() + verts[:, 0].min()),
                           0.5 * (verts[:, 1].max() + verts[:, 1].min())))
        verts[:, :2] = (verts[:, :2] - centre) * (Wd / max(span, 1e-9))
        verts[:, 2] *= H / max(top, 1e-9)
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts.tolist(), [], geo.tris)
    if len(mesh.loops) != 3 * len(geo.tris):
        bpy.data.meshes.remove(mesh)
        raise MifOpError("build_plant made %d face corners for %d triangles - a defect in this op, not in the request. "
                         "NOTHING was created." % (len(mesh.loops), len(geo.tris)))
    uv_layer = mesh.uv_layers.new(name="UVMap")
    uv_layer.data.foreach_set("uv", np.array(geo.uv, dtype=np.float32).ravel())
    col = mesh.color_attributes.new(name="Col", type="BYTE_COLOR", domain="CORNER")
    # color_srgb, as build_tree writes: each byte is value x 255, which the FBX carries and Unreal reads as written
    col.data.foreach_set("color_srgb", np.clip(np.array(geo.col, dtype=np.float32), 0.0, 1.0).ravel())
    mats = np.array(geo.mat, dtype=np.int32)
    two_slots = kind in ("bush", "hedge")
    mesh.polygons.foreach_set("material_index", mats if two_slots or weed_slots else np.zeros(len(mats), np.int32))
    mesh.polygons.foreach_set("use_smooth", np.ones(len(mats), dtype=bool))
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    _link_new(obj, coll)
    _place(obj, placed)
    # THE SAME MATERIALS build_tree makes, so a plant previews the way a tree's leaves do
    bark_color = tuple(0.5 * a + 0.1 for a in (0.3, 0.2, 0.12))
    bark_m, leaf_m = _tree_materials(obj.name, bark_color, color, color2)
    if two_slots:
        mesh.materials.append(bark_m)
    else:
        bpy.data.materials.remove(bark_m)
    mesh.materials.append(leaf_m)
    for k in sorted(weed_slots, key=weed_slots.get):
        # the leaf material again, in the weed's own colors and named for it: the slot a game binds its look to
        wb, wm = _tree_materials(obj.name + "_" + k.capitalize(), bark_color, *_WEED_COLORS[k])
        bpy.data.materials.remove(wb)
        wm.name = obj.name + "_" + k.capitalize()
        mesh.materials.append(wm)
    if kind == "hedge":
        # the fallen leaves: the leaf material again in browns, named for them
        lb, lm = _tree_materials(obj.name + "_Litter", bark_color, *_HEDGE_LITTER_COLORS)
        bpy.data.materials.remove(lb)
        lm.name = obj.name + "_Litter"
        mesh.materials.append(lm)

    out = _created(obj, coll, "MESH", "build_plant", name)
    n_bark = int(np.count_nonzero(mats == 0)) if two_slots else 0
    out.update({
        "kind": kind, "seed": seed, "count": count,
        "heightCm": round(float(verts[:, 2].max()) * 100.0, 2),
        "sinkCm": round(float(-verts[:, 2].min()) * 100.0, 2),
        "widthCm": round(float(max(np.ptp(verts[:, 0]), np.ptp(verts[:, 1]))) * 100.0, 2),
        "vertices": len(mesh.vertices), "faces": len(mesh.polygons),
        "triangles": {"stems": n_bark, "leaves": len(mats) - n_bark, "total": len(mats)},
        "materialSlots": [m.name for m in mesh.materials],
        "uvMap": {"name": "UVMap", "leaves": "each blade and leaf spans the whole 0-1 square, base at v 0, tip at v 1"},
        "colorAttribute": {"name": "Col", "domain": "CORNER", "type": "BYTE_COLOR",
                           "stored": "each byte is value x 255 (sRGB space), as build_tree writes, so export_mesh's "
                                     "FBX carries the values and Unreal's VertexColor node reads them as written",
                           "R": "a random value per blade, frond or leaf, for color variation (0.5 on a bush's stems)",
                           "G": "wind weight: 0 at the base, rising to 1 at the tips",
                           "B": "phase, shared by a blade, a frond or a stem and its leaves",
                           "A": ("how open the corner lies: 1 on the clipped face, toward 0.1 deep inside and less at "
                                 "the foot - an ambient occlusion for the material" if kind == "hedge" else "1")},
        "collisionHull": None,
        "pivot": "the base, at the object's origin, Z up; the plant reaches sinkCm below it so it sits into the ground",
        "elapsedSeconds": round(time.perf_counter() - started, 2),
    })
    if kind == "hedge":
        per_slot = np.bincount(mats, minlength=3)
        hedge_report["slots"] = {"stems": int(per_slot[0]), "leaves": int(per_slot[1]), "litter": int(per_slot[2])}
        hedge_report["note"] = ("lay middle pieces (ends none) every lengthCm along X and they meet with no seam, any "
                                "seed beside any other; a start piece's clipped end is at -X, an end piece's at +X")
        out["hedge"] = hedge_report
    if kind == "turf":
        out.update({
            "tileCm": round(width, 2), "blades": blades, "bare": bare, "weeds": weeds,
            "widthCm_note": "widthCm is the blades' spread, past the tile's edges where they lean over; tileCm is the "
                            "square their roots fill, the step to lay tiles at",
        })
        if weed_slots:
            per_slot = np.bincount(mats, minlength=len(weed_slots) + 1)
            weed_report["weedMix"] = list(mix)
            weed_report["flowers"] = flowers
            weed_report["slots"] = {k: {"index": i, "material": mesh.materials[i].name, "triangles": int(per_slot[i])}
                                    for k, i in weed_slots.items()}
            weed_report["note"] = ("the weeds come after the blades in the vertex order, and each lies whole where it "
                                   "grows, reaching over the tile's edge as blades do: tiles still meet at any quarter "
                                   "turn")
            out["weedsGrown"] = weed_report
    return out


OPS = {
    "build_rock": op_build_rock,
    "build_plant": op_build_plant,
}
