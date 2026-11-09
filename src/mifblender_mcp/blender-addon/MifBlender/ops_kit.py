"""Environment pieces - build_kit_piece, set_vertex_color_by_rule, build_lod_chain.

build_kit_piece MAKES ARCHITECTURE THAT SNAPS. A modular kit is walls, floors and pillars whose sides land on one
grid, whose pivots sit where a snap starts from, and whose collision is a few boxes. The conventions it builds to:
sides in whole modules (fitted fillers allowed and reported as such), a wall 12 cm thick and a floor 7 cm by default,
the pivot at the bottom - on a corner for walls and floors (the one a floor module tiles by), at the center for a
pillar - a surface slot and a trim slot, UV0 projected in real meters so pieces laid side by side continue one texture, a
lightmap UV layer LAST, which is the channel Unreal's Lightmap Coordinate Index points at, and UCX_ box hulls that
leave an opening open.

WHAT KEEPS IT FROM READING AS A RAW BOX: every edge is bevelled (3 rounded segments, 1 cm on a wall, clamped so a
thin piece keeps its faces), shaded smooth across the bevel with the flat faces kept flat by sharp edges; a wall
carries a plinth with a band along its top, a two-step casing and a lining round its opening, a window a sill and a
frame with a mullion and a transom, a pillar a stepped base and capital. Each trim part is its own closed shell standing
proud of the slab, which is how game kits build trim.

THE GEOMETRY IS BUILT, NOT CUT. A wall with an opening is the slab's face split on a grid round the opening, with the
opening's cell left out and its reveal faces built - every face a quad, the solid closed and manifold. A boolean cut
leaves n-gons and thin slivers at the seam; this leaves neither.

set_vertex_color_by_rule WRITES A MASK A MATERIAL BLENDS WITH: moss on the faces that look up, dirt toward the foot,
grime in the cavities, a per-piece variation value. Each rule writes 0-1 into one channel (or all three) of a
BYTE_COLOR attribute, stored as value x 255 - the sRGB-space bytes the FBX export carries and Unreal's VertexColor
node reads as written (the same store build_tree uses; the citation is there).

build_lod_chain MAKES HAND LODS UNREAL'S FBX IMPORTER READS: decimated copies under an Empty whose custom property
fbx_type is "LodGroup", which Blender's FBX exporter writes as the node's type (export_fbx_bin.py 3.6:558, 4.2 and
4.4:564, 5.0 and 5.2:565). Unreal takes a LodGroup's children in order as LOD0, LOD1, ... (FbxFactory.cpp
5.8:993-1030). The source object stays LOD0 under its own name, so hulls named UCX_<name>_NN still attach
(FbxStaticMeshImport.cpp 5.8:588-598 and 2707-2745 match a hull to its mesh node's name).
"""
import math
import os
import random
import time

import bmesh
import bpy
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

from .ops_common import (remove_objects, MifOpError, check_output_path, file_before, finite_float, finite_floats, get_object, reject_unknown,
                         triangulate_ngons,
                         replaced_fields, rnd, select_only, selection_restore, selection_snapshot, take, take_bool,
                         take_float, take_int)
from .ops_create import _created, _link_new, _place, _place_values, _resolve_collection
from .ops_mesh import (FBX_EXPORT_ARGS, _UE_HULL_VERT_LIMIT, _hull_audit, _resolve_out_path, op_uv_info,
                       op_uv_unwrap)

# MODULE-LEVEL AND PLAIN LITERALS: describe_op and parity_check read an accepted-key list statically.
BUILD_KIT_PIECE_PARAMS = (
    "name", "kind", "opening", "module", "length", "height", "width", "thickness", "openingWidth", "openingHeight",
    "sillHeight", "openingOffset", "pivot", "bevel", "bevelSegments", "trim", "uvTileCm", "color", "material",
    "trimColor", "trimMaterial", "gridCm", "collision", "location", "rotation", "collection", "slatPitch",
    "hoodRadius", "hardware", "louver", "minThicknessCm",
)
VERTEX_RULE_PARAMS = (
    "object", "rule", "name", "channel", "range", "invert", "combine", "seed", "element", "distance", "rays",
    "space", "makeActive", "dirt", "group", "adjust",
)
_DIRT_KEYS = {"blurStrength", "blurIterations", "cleanAngle", "dirtAngle", "dirtOnly", "normalize"}
LOD_CHAIN_PARAMS = ("object", "ratios", "file", "overwrite")

_KIT_NOTES = {
    "size": "sides are counted in modules: length and height (a wall), length and width (a floor), height (a pillar); "
            "module sets the module in cm. A fitted filler is a fractional count, e.g. length 1.5.",
    "depth": "a wall's and a floor's thickness, and a pillar's side, are 'thickness' in cm.",
    "door": "pass kind wall with opening door (or window); openingWidth, openingHeight, sillHeight and openingOffset "
            "shape it.",
    "window": "pass kind wall with opening window.",
    "uv": "UV0 is a box projection in meters (uvTileCm per texture repeat); the lightmap layer is made and checked "
          "every time.",
    "lightmap": "the lightmap UV layer LightmapUV is always made, last, and checked with uv_info's lightmapReady.",
    "lod": "a kit piece is meant for Nanite; build_lod_chain makes hand LODs when a piece needs them.",
    "materials": "slot 0 is the surface (material or color), slot 1 the trim (trimMaterial or trimColor); a floor, "
                 "or a piece with trim false, has slot 0 only.",
    "roundness": "bevel (cm) and bevelSegments set how the edges are rounded; bevel 0 leaves hard edges.",
}
_VCOL_NOTES = {
    "by": "the rule is 'rule': up, height, cavity or random.",
    "color": "this op writes a 0-1 mask computed by a rule; set_vertex_color writes one given color.",
    "faces": "a rule covers the whole mesh; set_vertex_color colors a face list.",
    "domain": "a new attribute is made on face corners (BYTE_COLOR); an existing BYTE_COLOR attribute keeps its "
              "domain.",
    "dataType": "masks are 8-bit (BYTE_COLOR), which is what Unreal's vertex color stores.",
    "blur": "cavity is measured by rays over 'distance' meters, which is already smooth; there is no blur pass.",
}

_LOD_NOTES = {
    "screenSize": "the FBX carries no screen sizes that Unreal reads for an imported chain; set them in Unreal or let "
                  "it compute them.",
    "count": "the chain has one LOD per entry in 'ratios'.",
    "targetTris": "ratios are fractions of LOD0's triangles; decimate_mesh takes a triangle target for one mesh.",
    "name": "the LODs are named <object>_LOD1, <object>_LOD2, ... and the group <object>_LODGroup; the object itself "
            "stays LOD0 under its own name, so its UCX_ hulls still attach.",
    "mode": "the chain decimates by edge collapse, which is the mode that takes a ratio.",
}

_KINDS = ("wall", "floor", "pillar", "shutter", "door_leaf")
# WHAT FILLS AN OPENING is built in the frame of the wall it fills - the same length, height, thickness, opening,
# offset, trim and pivot - so it lands in that wall's opening on that wall actor's transform
_INFILL = ("shutter", "door_leaf")
_OPENINGS = ("none", "door", "window")
_PIVOTS = ("corner", "side", "center")
# DEFAULTS: whole-meter sides (a 400 x 300 cm wall, a 400 x 400 cm floor), a 12 cm wall and a 7 cm floor slab - the
# medians game kits are built to - and a 40 cm pillar, three modules tall.
_DEFAULTS = {
    "wall": {"length": 4.0, "height": 3.0, "thickness": 12.0, "bevel": 1.0, "pivot": "corner"},
    "floor": {"length": 4.0, "width": 4.0, "thickness": 7.0, "bevel": 1.0, "pivot": "corner"},
    "pillar": {"height": 3.0, "thickness": 40.0, "bevel": 1.5, "pivot": "center"},
    "shutter": {"length": 4.0, "height": 3.0, "thickness": 12.0, "bevel": 0.3, "pivot": "corner"},
    "door_leaf": {"length": 4.0, "height": 3.0, "thickness": 12.0, "bevel": 0.3, "pivot": "corner"},
}
_OPENING_DEFAULTS = {"door": (110.0, 220.0, 0.0), "window": (100.0, 120.0, 90.0)}   # width, height, sill (cm)
# TRIM IN CENTIMETERS. Each moulding is two steps - a board and a thicker band on its outer edge - because one flat
# board reads as a strip of tape; the second step throws the shadow line a real moulding throws.
_PLINTH_H, _PLINTH_PROUD = 12.0, 1.2            # the board along the foot of the wall
_BEAD_H, _BEAD_PROUD = 2.0, 2.0                 # the band along its top edge
_CASING_W, _CASING_PROUD = 8.0, 1.5             # the casing round an opening: its full width and the board's proudness
_BAND_W, _BAND_PROUD = 2.0, 2.6                 # the band on the casing's outer edge
_LINING_T = 2.0                                 # the boards lining the opening's reveal
_SILL_H, _SILL_PROUD, _SILL_OVER = 4.0, 4.5, 3.0
_FRAME_W, _FRAME_D, _BAR_W = 5.0, 7.0, 3.5      # a window's frame, its depth, and its mullion and transom
# HOW FAR A PLINTH STOPS SHORT OF THE WALL'S ENDS. 1 mm until 2026-10-18: its end face then lay 1 mm from the slab's end
# face, facing the same way, and coplanar_overlaps found it on every wall - 1 mm still fights at a distance, 2 mm does
# not (from use). 3 mm, as the other kit offsets.
_END_INSET = 0.3
_SUPPORT_OUT = 10.0                             # the support line a hand's width out from where trim meets the slab
_MIN_MARGIN = 10.0       # cm of wall kept between an opening (with its casing) and an edge
_LIGHTMAP_LAYER = "LightmapUV"
_MAX_SIDE_CM = 5000.0


# ------------------------------------------------------------------------------------------------ build_kit_piece
def _cells(us, vs, keep, w0, w1, to_xyz, bevel, name, slot=0):
    """A part: the extrusion from w0 to w1 of the grid cells in (u, v) whose keep flag is set. us and vs are the grid
    lines; keep[i][j] covers us[i]..us[i+1] by vs[j]..vs[j+1]. to_xyz maps (u, v, w) to (x, y, z)."""
    return {"us": list(us), "vs": list(vs), "keep": keep, "w": (w0, w1), "to_xyz": to_xyz, "bevel": bevel,
            "name": name, "slot": slot}


def _box(x0, x1, y0, y1, z0, z1, bevel, name, slot=0):
    return _cells((x0, x1), (y0, y1), [[True]], z0, z1, lambda u, v, w: (u, v, w), bevel, name, slot)


def _face_boards(x0, x1, y0, y1, z0, z1, bevel, name, t, slot=1):
    """Trim on both faces of a slab t thick: two boards that BUTT the slab - y0 to 0 and t to y1 - with square backs (no
    bevel on the edges against the slab, so no groove shows where they meet it), not one box run through it.

    ONE BOX THROUGH THE SLAB SHARED THE SLAB'S PLANES. coplanar_overlaps on every wall with trim, 2026-10-18: the plinth's
    and the door casing's bottom faces lay in the slab's bottom plane, facing the same way, overlapping it across the
    slab's whole thickness - 14 such overlaps on a plain wall, 4,173 cm2 of them. Pieces butt; they never overlap (from
    use)."""
    if not (y0 < 0.0 < t < y1):
        return [_box(x0, x1, y0, y1, z0, z1, bevel, name, slot)]
    front = _box(x0, x1, y0, 0.0, z0, z1, bevel, name, slot)
    back = _box(x0, x1, t, y1, z0, z1, bevel, name, slot)
    front["square"], back["square"] = ((1, 0.0),), ((1, t),)
    return [front, back]


# THREE SHAPES FOR WHAT FILLS AN OPENING, every one a closed all-quad solid. A part made here carries its own vertex
# and face lists ("mesh"); _part_bmesh takes them as they are and recalculates the normals outward.
def _sheet(path, thick, w0, w1, to_xyz, name, slot=1):
    """A thin formed sheet: the 2D centerline `path` [(u, v), ...] offset half `thick` to each side and extruded from
    w0 to w1. ITS END CAPS ARE QUADS BETWEEN THE TWO RUNS, one per path segment - a thin shell capped with one n-gon
    triangulates into fans of slivers across the whole profile. A sharp turn's miter is capped at 4x the thickness."""
    n = len(path)
    h = thick / 2.0
    left, right = [], []
    for i, (u, v) in enumerate(path):
        dirs = []
        for a, b in ((i - 1, i), (i, i + 1)):
            if 0 <= a and b < n:
                du, dv = path[b][0] - path[a][0], path[b][1] - path[a][1]
                ln = math.hypot(du, dv)
                dirs.append((du / ln, dv / ln))
        nu = sum(-d[1] for d in dirs)
        nv = sum(d[0] for d in dirs)
        ln = math.hypot(nu, nv) or 1.0
        nu, nv = nu / ln, nv / ln
        # the miter: the offset along the averaged normal that keeps the wall `thick` on both segments
        cos_half = abs(nu * -dirs[0][1] + nv * dirs[0][0])
        s = h / max(cos_half, 0.25)
        left.append((u + nu * s, v + nv * s))
        right.append((u - nu * s, v - nv * s))
    verts = []
    for w in (w0, w1):
        verts += [to_xyz(u, v, w) for u, v in left] + [to_xyz(u, v, w) for u, v in right]

    def L(i, side):
        return side * 2 * n + i

    def R(i, side):
        return side * 2 * n + n + i

    faces = []
    for i in range(n - 1):
        faces.append([L(i, 0), L(i + 1, 0), L(i + 1, 1), L(i, 1)])
        faces.append([R(i, 0), R(i, 1), R(i + 1, 1), R(i + 1, 0)])
        faces.append([L(i, 0), R(i, 0), R(i + 1, 0), L(i + 1, 0)])
        faces.append([L(i, 1), L(i + 1, 1), R(i + 1, 1), R(i, 1)])
    faces.append([L(0, 0), L(0, 1), R(0, 1), R(0, 0)])
    faces.append([L(n - 1, 0), R(n - 1, 0), R(n - 1, 1), L(n - 1, 1)])
    return {"mesh": (verts, faces), "bevel": 0.0, "name": name, "slot": slot}


def _cyl(cu, cv, radius, w0, w1, sides, to_xyz, name, slot=1):
    """A round bar from w0 to w1 round (cu, cv), `sides` even. Each end is capped by a ladder of quads between the
    ring's two halves - no n-gon, no fan."""
    verts = []
    for w in (w0, w1):
        for k in range(sides):
            a = 2.0 * math.pi * k / sides
            verts.append(to_xyz(cu + radius * math.cos(a), cv + radius * math.sin(a), w))
    faces = [[k, (k + 1) % sides, sides + (k + 1) % sides, sides + k] for k in range(sides)]
    for base in (0, sides):
        for k in range(sides // 2 - 1):
            q = [base + k, base + k + 1, base + sides - 2 - k, base + sides - 1 - k]
            # ONE WINDING OVER THE WHOLE BAR: the w0 cap runs its ring the other way from the sides
            faces.append(q[::-1] if base == 0 else q)
    return {"mesh": (verts, faces), "bevel": 0.0, "name": name, "slot": slot}


def _arc(cu, cv, r, degs):
    return [(cu + r * math.cos(math.radians(d)), cv + r * math.sin(math.radians(d))) for d in degs]


def _along_x(u, v, w):      # a profile in (y, z) run along x
    return (w, u, v)


def _along_y(u, v, w):      # a section in (x, z) run along y
    return (u, w, v)


def _along_z(u, v, w):      # a section in (x, y) run up z
    return (u, v, w)


# A ROLL-UP SHUTTER, in cm. The coil is a 9-1/2 in barrel with the curtain's last wraps on it; the bottom bar a
# 2 x 1-1/2 in angle on a PVC bulb astragal; the guides channels on jamb angles, bolted.
_SHUTTER_BARREL_R, _SHUTTER_WRAP = 12.07, 1.6
_SHUTTER_MOUNT_GAP = 0.1            # the guides' angles stand this far off the wall (or its casing) face
_SHUTTER_CURTAIN_Y = 2.0            # the curtain's plane, in front of the mount plane
_SHUTTER_SLAT_T, _SHUTTER_HOOD_T = 0.1, 0.12
_SHUTTER_HOOD_OVER = 10.0           # the hood runs this far past the opening on each side, over the guides
_SHUTTER_HOOD_CLEAR = 2.5
_SHUTTER_PITCH = (3.0, 15.0, 6.7)   # min, max, default slat pitch (a 2-5/8 in roll-formed slat)
_SHUTTER_HOOD_R = (0.0, 15.0, 8.0)
# A HOLLOW-METAL DOOR LEAF, in cm: 1-3/4 in thick, 1/8 in clear of the jambs and the head, a 5/8 in undercut
_LEAF_T, _LEAF_GAP, _LEAF_UNDERCUT = 4.45, 0.3, 1.5
_LOUVER_W, _LOUVER_H, _LOUVER_FOOT, _LOUVER_PITCH = 61.0, 30.5, 30.0, 3.0


def _shutter_dims(spec):
    """Where the shutter's pieces stand (cm, the wall's slab-corner space): computed once for the refusals and the
    build, so a refusal and the geometry cannot disagree."""
    ca, cb, z0, cz1 = _clear_opening(spec)
    t, pitch = spec["thickness"], spec["slatPitch"]
    ym = t + (_BAND_PROUD if spec["trim"] else 0.0) + _SHUTTER_MOUNT_GAP
    yc = ym + _SHUTTER_CURTAIN_Y
    rc = _SHUTTER_BARREL_R + _SHUTTER_WRAP
    zh0 = cz1 + 1.0
    coil = (yc + rc, zh0 + 4.0 + pitch + rc)          # the curtain leaves the coil's back, in its own plane
    yf = coil[0] + rc + _SHUTTER_HOOD_CLEAR
    zt = coil[1] + rc + _SHUTTER_HOOD_CLEAR
    return {"ca": ca, "cb": cb, "z0": z0, "cz1": cz1, "ym": ym, "yc": yc, "rc": rc, "zh0": zh0, "coil": coil,
            "yf": yf, "zt": zt}


def _slat_profile(pitch):
    """A roll-formed slat's centerline in (y, z) from its base: the hook that hangs in the slat below's curl, a leg up
    to the face, the face bowed 4 mm out, and the curl open at the top that the slat above hooks into. The hook (r 2 mm)
    sits on the curl's (r 4.5 mm) center one pitch down, so the two interlock without touching."""
    hook = _arc(-0.45, -0.9, 0.2, (330, 270, 210, 150, 90))
    face = [(0.4 * math.sin(math.pi * k / 4.0), 0.6 + (pitch - 1.5) * k / 4.0) for k in range(5)]
    curl = _arc(-0.45, pitch - 0.9, 0.45, (-40, -80, -120, -160, -200, -240))
    return hook + face + curl


def _shutter_parts(spec):
    """A roll-up shutter closed over the wall's opening, in the wall's frame: the curtain of interlocking slats, the
    bottom angle bar on its astragal with a lift handle, channel guides on jamb angles with bolts, and a hood with a
    rounded front over the coil, closed by end brackets."""
    d = _shutter_dims(spec)
    ca, cb, z0, ym, yc, zh0, rc, yf, zt = (d[k] for k in ("ca", "cb", "z0", "ym", "yc", "zh0", "rc", "yf", "zt"))
    pitch, hw = spec["slatPitch"], spec["hardware"]
    parts, hulls = [], []
    mid = (ca + cb) / 2.0
    # ---- the bottom bar on its bulb astragal; the curtain's slats from it up into the hood
    zb = z0 + 2.4 + 0.18
    parts.append(_cyl(yc + 1.9, z0 + 1.2, 1.2, ca - 1.2, cb + 1.2, 12, _along_x, "astragal", 1))
    parts.append(_sheet([(yc, zb + 5.1), (yc, zb), (yc + 3.8, zb)], 0.32, ca - 1.2, cb + 1.2, _along_x,
                        "bottom bar", 0))
    if hw:
        hz0, hz1 = zb + 2.0, zb + 3.2
        for x0, x1 in ((mid - 6.0, mid - 4.8), (mid + 4.8, mid + 6.0)):
            parts.append(_box(x0, x1, yc + 0.18, yc + 2.2, hz0, hz1, 0.0, "lift handle", 1))
        parts.append(_box(mid - 7.0, mid + 7.0, yc + 2.2, yc + 3.4, hz0, hz1, 0.0, "lift handle", 1))
    zs0 = zb + 5.1 + 1.5
    n = max(1, int(math.ceil((zh0 + 2.0 - zs0) / pitch)))
    prof = _slat_profile(pitch)
    for k in range(n):
        base = zs0 + k * pitch
        parts.append(_sheet([(yc + y, base + z) for y, z in prof], _SHUTTER_SLAT_T, ca - 1.2, cb + 1.2, _along_x,
                            "slat", 0))
    # ---- the guides: a channel open to the opening, on a jamb angle; mirrored on the right
    for side in (-1.0, 1.0):
        def mx(x, s=side):
            return x if s < 0 else ca + cb - x
        chan = [(mx(ca + 2.5), ym + 0.6), (mx(ca - 1.83), ym + 0.6), (mx(ca - 1.83), ym + 7.0), (mx(ca + 2.5), ym + 7.0)]
        ang = [(mx(ca - 9.0), ym + 0.25), (mx(ca - 2.25), ym + 0.25), (mx(ca - 2.25), ym + 7.0)]
        parts.append(_sheet(chan, 0.3, z0, zh0, _along_z, "guide", 1))
        parts.append(_sheet(ang, 0.5, z0, zh0, _along_z, "jamb angle", 1))
        if hw:
            nb = max(2, int((zh0 - z0) / 60.0) + 1)
            for k in range(nb):
                zk = z0 + 8.0 + (zh0 - z0 - 16.0) * k / (nb - 1)
                parts.append(_cyl(mx(ca - 6.0), zk, 0.9, ym + 0.5, ym + 1.1, 6, _along_y, "bolt", 1))
        lo, hi = sorted((mx(ca - 9.0), mx(ca + 2.65)))
        hulls.append((lo, hi, ym, ym + 7.15, z0, zh0))
    # ---- the hood over the coil: a sheet from the mount plane over the top, round the front, down to a lip
    r = spec["hoodRadius"]
    hood = [(ym + 0.1, zt)]
    if r > 0.0:
        hood += _arc(yf - r, zt - r, r, (90, 67.5, 45, 22.5, 0))
    else:
        hood.append((yf, zt))
    hood += [(yf, zh0), (yf - 2.5, zh0)]
    hx0, hx1 = ca - _SHUTTER_HOOD_OVER, cb + _SHUTTER_HOOD_OVER
    parts.append(_sheet(hood, _SHUTTER_HOOD_T, hx0, hx1, _along_x, "hood", 1))
    for x0, x1 in ((hx0 - 0.6, hx0), (hx1, hx1 + 0.6)):
        parts.append(_box(x0, x1, ym, yf + 0.06, zh0 - 0.06, zt + 0.06, 0.0, "end bracket", 1))
    parts.append(_cyl(d["coil"][0], d["coil"][1], rc, hx0 + 0.1, hx1 - 0.1, 24, _along_x, "coil", 1))
    hulls.append((hx0 - 0.6, hx1 + 0.6, ym, yf + 0.06, zh0 - 0.06, zt + 0.06))
    hulls.append((ca - 1.2, cb + 1.2, yc - 1.0, yc + 4.0, z0, zh0))
    return parts, hulls


def _leaf_dims(spec):
    ca, cb, z0, cz1 = _clear_opening(spec)
    t = spec["thickness"]
    lx0, lx1, lz0, lz1 = ca + _LEAF_GAP, cb - _LEAF_GAP, z0 + _LEAF_UNDERCUT, cz1 - _LEAF_GAP
    ly0, ly1 = t / 2.0 - _LEAF_T / 2.0, t / 2.0 + _LEAF_T / 2.0
    hole = None
    if spec["louver"]:
        hw = min(_LOUVER_W, lx1 - lx0 - 20.0)
        mid = (lx0 + lx1) / 2.0
        hole = (mid - hw / 2.0, mid + hw / 2.0, lz0 + _LOUVER_FOOT, lz0 + _LOUVER_FOOT + _LOUVER_H)
    return {"x": (lx0, lx1), "y": (ly0, ly1), "z": (lz0, lz1), "hole": hole}


def _leaf_parts(spec):
    """A 1-3/4 in hollow-metal door leaf hung in the wall's clear opening, in the wall's frame: the leaf with hemmed
    (rounded) edges, and with hardware three butt hinges, a lever and a deadbolt on both faces, a kick plate, an
    astragal over the latch-side gap and a drip cap over the opening on the wall's face; with louver a stamped
    chevron louver through its lower half."""
    d = _leaf_dims(spec)
    (lx0, lx1), (ly0, ly1), (lz0, lz1), hole = d["x"], d["y"], d["z"], d["hole"]
    t, grid = spec["thickness"], spec["gridCm"]
    parts = []
    xs, zs = [lx0, lx1], [lz0, lz1]
    if hole:
        xs, zs = [lx0, hole[0], hole[1], lx1], [lz0, hole[2], hole[3], lz1]
    xs, zs = _with_grid(xs, grid), _with_grid(zs, grid)
    parts.append(_cells(xs, zs, _keep_outside(xs, zs, hole), ly0, ly1, _along_y, min(spec["bevel"], 1.0), "leaf", 0))
    if hole:
        hx0, hx1, hz0, hz1 = hole
        z = hz0 + 0.8
        while z + 3.6 <= hz1 - 0.6:
            parts.append(_sheet([(ly0 + 0.4, z), (t / 2.0, z + 3.6), (ly1 - 0.4, z)], 0.12, hx0 + 0.05, hx1 - 0.05,
                                _along_x, "louver blade", 1))
            z += _LOUVER_PITCH
        for y0, y1 in ((ly1, ly1 + 0.3), (ly0 - 0.3, ly0)):
            parts += [_box(hx0 - 2.0, hx0, y0, y1, hz0 - 2.0, hz1 + 2.0, 0.0, "louver frame", 1),
                      _box(hx1, hx1 + 2.0, y0, y1, hz0 - 2.0, hz1 + 2.0, 0.0, "louver frame", 1),
                      _box(hx0, hx1, y0, y1, hz1, hz1 + 2.0, 0.0, "louver frame", 1),
                      _box(hx0, hx1, y0, y1, hz0 - 2.0, hz0, 0.0, "louver frame", 1)]
    if spec["hardware"]:
        # BUTT HINGES on the front face's hinge edge: a knuckle and its leaf, 4-1/2 in tall, three to a door
        hh = 11.4
        for zc in (lz1 - 12.7 - hh / 2.0, (lz0 + lz1) / 2.0 + 5.0, lz0 + 25.4 + hh / 2.0):
            parts.append(_cyl(lx0 + 0.5, ly1 + 0.65, 0.65, zc - hh / 2.0, zc + hh / 2.0, 12, _along_z, "hinge", 1))
            parts.append(_box(lx0 + 1.2, lx0 + 4.0, ly1, ly1 + 0.25, zc - hh / 2.0, zc + hh / 2.0, 0.0, "hinge", 1))
        # THE LEVER AND THE DEADBOLT at a 2-3/4 in backset, on both faces
        cx = lx1 - 7.0
        lz = lz0 + min(96.0, 0.45 * (lz1 - lz0))
        dz = lz + 14.0
        for face, sgn in ((ly1, 1.0), (ly0, -1.0)):
            def yy(a, b, f=face, s=sgn):
                return (f + a, f + b) if s > 0 else (f - b, f - a)
            y0, y1 = yy(0.0, 0.9)
            parts.append(_cyl(cx, lz, 3.3, y0, y1, 16, _along_y, "lever", 1))
            y0, y1 = yy(0.9, 3.4)
            parts.append(_cyl(cx, lz, 0.9, y0, y1, 12, _along_y, "lever", 1))
            y0, y1 = yy(3.4, 5.0)
            parts.append(_box(cx - 12.0, cx + 1.0, y0, y1, lz - 0.8, lz + 0.8, 0.3, "lever", 1))
            y0, y1 = yy(0.0, 0.9)
            parts.append(_cyl(cx, dz, 3.0, y0, y1, 16, _along_y, "deadbolt", 1))
            if sgn > 0:
                y0, y1 = yy(0.9, 1.9)
                parts.append(_cyl(cx, dz, 1.4, y0, y1, 12, _along_y, "deadbolt", 1))
            else:
                y0, y1 = yy(0.9, 2.5)
                parts.append(_box(cx - 0.6, cx + 0.6, y0, y1, dz - 2.0, dz + 2.0, 0.2, "deadbolt", 1))
        # THE KICK PLATE STANDS 2 MM PROUD of the leaf: 1.3 mm (a real plate's gauge) put its face within the 1.5 mm in
        # which coplanar_overlaps calls two same-facing faces one plane, and 1 mm still fights at a distance (from use)
        parts.append(_box(lx0 + 2.5, lx1 - 3.6, ly0 - 0.2, ly0, lz0 + 0.6, lz0 + 25.4, 0.0, "kick plate", 1))
        parts.append(_box(lx1 - 2.7, lx1 + _LEAF_GAP - 0.05, ly1, ly1 + 0.32, lz0, lz1, 0.0, "astragal", 1))
        # THE DRIP CAP on the wall's face over the opening (its casing, when there is trim)
        a, b, z1 = _rough_opening(spec)
        cw = _CASING_W if spec["trim"] else 0.0
        zc = z1 + cw + 0.1
        out = (_BAND_PROUD if spec["trim"] else 0.0) + 1.6
        # ITS LEG STANDS OFF THE WALL: centered 2.5 mm out, its face is 3 mm proud of the wall's face. At 0.6 mm out the
        # face was 1.1 mm proud and coplanar_overlaps found it against the wall it hangs on (2026-10-18)
        parts.append(_sheet([(t + 0.25, zc + 3.5), (t + 0.25, zc + 0.6), (t + out, zc + 0.1), (t + out, zc - 1.2)],
                            0.1, a - cw - 2.0, b + cw + 2.0, _along_x, "drip cap", 1))
    return parts, [(lx0, lx1, ly0, ly1, lz0, lz1)]


def _with_grid(breaks, step):
    """The grid lines of a face with lines added every `step` cm between the ones it has, so a vertex-color mask has
    vertices to vary across: a 3 m wall face with corners only would stretch a 60 cm dirt band over its whole height."""
    if step <= 0.0:
        return list(breaks)
    out = [breaks[0]]
    for a, b in zip(breaks, breaks[1:]):
        n = max(1, int(round((b - a) / step)))
        out.extend(a + (b - a) * k / n for k in range(1, n + 1))
    return out


def _with_lines(breaks, extra):
    """breaks with the lines in `extra` that fall strictly inside its span added, sorted, none closer than 1 mm."""
    lo, hi = breaks[0], breaks[-1]
    out = []
    for v in sorted(list(breaks) + [e for e in extra if lo < e < hi]):
        if not out or v - out[-1] > 0.1:
            out.append(v)
    # The far end stays, whatever stands near it: a list in meters lost it to the 0.1 gap, so a dumpster's lid stopped 4 cm short of its own front
    # skirt and its grip hung in the air, seen first by gameReady's board rule
    if len(out) > 1 and out[-1] != hi:
        out[-1] = hi
    return out


def _keep_outside(xs, zs, hole):
    """keep[i][j] for the grid cells, False for a cell inside the hole (x0, x1, z0, z1); every cell with no hole."""
    if hole is None:
        return [[True] * (len(zs) - 1) for _ in range(len(xs) - 1)]
    x0, x1, z0, z1 = hole
    return [[not (x0 <= (xs[i] + xs[i + 1]) / 2.0 <= x1 and z0 <= (zs[j] + zs[j + 1]) / 2.0 <= z1)
             for j in range(len(zs) - 1)] for i in range(len(xs) - 1)]


def _part_bmesh(part):
    """One closed, all-quad solid for a part: the kept cells' faces at w0 and w1, and a side face on every cell edge
    that borders nothing kept. Normals are recalculated outward, so winding is not hand-kept. A part made by _sheet or
    _prism carries its own closed vertex and face lists under "mesh"."""
    bm = bmesh.new()
    if "mesh" in part:
        pverts, pfaces = part["mesh"]
        made = [bm.verts.new(co) for co in pverts]
        for f in pfaces:
            bm.faces.new([made[i] for i in f])
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
        return bm
    us, vs, keep = part["us"], part["vs"], part["keep"]
    w0, w1 = part["w"]
    to_xyz = part["to_xyz"]
    verts = {}

    def vert(i, j, side):
        key = (i, j, side)
        v = verts.get(key)
        if v is None:
            v = bm.verts.new(to_xyz(us[i], vs[j], (w0, w1)[side]))
            verts[key] = v
        return v

    def kept(i, j):
        return 0 <= i < len(us) - 1 and 0 <= j < len(vs) - 1 and keep[i][j]

    for i in range(len(us) - 1):
        for j in range(len(vs) - 1):
            if not keep[i][j]:
                continue
            for side in (0, 1):
                bm.faces.new([vert(i, j, side), vert(i + 1, j, side), vert(i + 1, j + 1, side), vert(i, j + 1, side)])
            # the four cell edges: (a, b) grid points and the neighbor across them
            for (a, b), (ni, nj) in ((((i, j), (i + 1, j)), (i, j - 1)), (((i + 1, j), (i + 1, j + 1)), (i + 1, j)),
                                     (((i + 1, j + 1), (i, j + 1)), (i, j + 1)), (((i, j + 1), (i, j)), (i - 1, j))):
                if not kept(ni, nj):
                    bm.faces.new([vert(a[0], a[1], 0), vert(b[0], b[1], 0), vert(b[0], b[1], 1), vert(a[0], a[1], 1)])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    return bm


def _bevel_part(bm, offset, segments, square=()):
    """Bevel every edge where two faces meet at an angle (a grid line across a flat face has none). Returns the set of
    faces the bevel made. clamp_overlap keeps a thin piece's faces: an offset the geometry has no room for is capped.
    `square` names planes (axis index, value in cm) whose edges stay square: a board's back where it butts the slab."""
    if offset <= 0.0:
        return set()

    def on_square(e):
        return any(all(abs(v.co[ax] - val) < 1e-6 for v in e.verts) for ax, val in square)
    edges = [e for e in bm.edges if len(e.link_faces) == 2 and e.calc_face_angle(0.0) > math.radians(1.0)
             and not on_square(e)]
    if not edges:
        return set()
    out = bmesh.ops.bevel(bm, geom=edges, offset=offset, offset_type="OFFSET", profile_type="SUPERELLIPSE",
                          segments=segments, profile=0.5, affect="EDGES", clamp_overlap=True)
    if square:
        # A SQUARE BACK GATHERS THE BEVELLED SIDE EDGES' ENDS, so it comes out an n-gon (KP103 found 4 on a plain wall):
        # it is split into triangles - it lies against the slab, where nothing sees how it is cut
        ngons = [f for f in bm.faces if len(f.verts) > 4 and any(
            all(abs(v.co[ax] - val) < 1e-6 for v in f.verts) for ax, val in square)]
        if ngons:
            bmesh.ops.triangulate(bm, faces=ngons, quad_method="BEAUTY", ngon_method="BEAUTY")
    return set(out.get("faces") or [])


def _kit_parts(spec):
    """Every part of the piece in centimeters, in the space where the slab's bottom corner is the origin: x along its
    length, y through its thickness, z up. Returns (parts, slab regions for the hulls as boxes)."""
    kind, t, bev = spec["kind"], spec["thickness"], spec["bevel"]
    if kind == "shutter":
        return _shutter_parts(spec)
    if kind == "door_leaf":
        return _leaf_parts(spec)
    trim_bev = min(bev, 0.5)
    parts, hulls = [], []
    if kind == "wall":
        length, height = spec["lengthCm"], spec["heightCm"]
        opening = spec["opening"]
        grid = spec["gridCm"]
        # SUPPORT LINES WHERE TRIM MEETS THE SLAB, and one a hand's width out: a mask computed per vertex (grime where
        # the plinth meets the wall) then fades over 10 cm, not over the whole next grid cell. Only with a grid - gridCm
        # 0 is a request for the bare slab.
        near = (0.0, _SUPPORT_OUT) if (spec["trim"] and grid > 0.0) else ()
        z_extra = [_PLINTH_H + d for d in near]
        x_extra = []
        if opening == "none":
            xs = _with_grid((0.0, length), grid)
            zs = _with_grid(_with_lines((0.0, height), z_extra), grid)
            keep = _keep_outside(xs, zs, None)
            gaps = []
            hulls.append((0.0, length, 0.0, t, 0.0, height))
        else:
            # THE REQUEST IS THE CLEAR OPENING, the one a character walks through. The lining is set into the slab's
            # reveal OUTSIDE it, so the slab's rough opening is the clear one plus a lining board on each side and on
            # top. Until 2026-10-02 the lining stood inside the requested opening: asked 120 x 230, the doorway was
            # 116 x 228 while the reply and the hulls said 120 x 230 (benchmark round ten B, X2 - 3 of 5 runs wrong).
            ca, cb, z0, cz1 = _clear_opening(spec)
            a, b, z1 = _rough_opening(spec)
            x_extra = [a - _CASING_W - d for d in near] + [b + _CASING_W + d for d in near]
            z_extra += [z1 + _CASING_W + d for d in near]
            if z0 > 0.0:
                z_extra += [z0 - _SILL_H - _CASING_W - d for d in near]
            xs = _with_grid(_with_lines((0.0, a, b, length), x_extra), grid)
            zs = _with_grid(_with_lines([0.0] + ([z0] if z0 > 0.0 else []) + [z1, height], z_extra), grid)
            keep = _keep_outside(xs, zs, (a, b, z0, z1))
            gaps = [(a - _CASING_W, b + _CASING_W)] if opening == "door" else []
            # THE HULLS BOUND THE CLEAR OPENING: the lining is solid, so the collision stops where the mesh does
            hulls += [(0.0, ca, 0.0, t, 0.0, height), (cb, length, 0.0, t, 0.0, height), (ca, cb, 0.0, t, cz1, height)]
            if z0 > 0.0:
                hulls.append((ca, cb, 0.0, t, 0.0, z0))
        parts.append(_cells(xs, zs, keep, 0.0, t, lambda u, v, w: (u, w, v), bev, "slab", 0))
        if spec["trim"]:
            # THE PLINTH runs the wall's length on both faces, broken by a door's casing. It stops 1 mm short of each
            # end of the wall: flush, its end face would lie in the slab's end plane and flicker there
            runs, x = [], _END_INSET
            for g0, g1 in gaps:
                runs.append((x, g0))
                x = g1
            runs.append((x, length - _END_INSET))
            for r0, r1 in runs:
                if r1 - r0 > 1.0:
                    parts += _face_boards(r0, r1, -_PLINTH_PROUD, t + _PLINTH_PROUD, 0.0, _PLINTH_H - _BEAD_H, trim_bev,
                                          "plinth", t)
                    parts += _face_boards(r0, r1, -_BEAD_PROUD, t + _BEAD_PROUD, _PLINTH_H - _BEAD_H, _PLINTH_H,
                                          trim_bev, "plinth", t)
            if opening != "none":
                cw, bw = _CASING_W, _BAND_W
                # the casing's board and, on its outer edge, the band - jambs to the head, the head across both. NO TWO
                # PARTS SHARE A FACE'S PLANE WHERE THEY OVERLAP: coplanar overlapping faces render as flicker and black
                # specks, so every part abuts the next instead of running through it
                for lo, hi, proud in ((0.0, cw - bw, _CASING_PROUD), (cw - bw, cw, _BAND_PROUD)):
                    y0, y1 = -proud, t + proud
                    parts += _face_boards(a - hi, a - lo, y0, y1, z0, z1 + lo, trim_bev, "casing", t)
                    parts += _face_boards(b + lo, b + hi, y0, y1, z0, z1 + lo, trim_bev, "casing", t)
                    parts += _face_boards(a - hi, b + hi, y0, y1, z1 + lo, z1 + hi, trim_bev, "casing", t)
                    if opening == "window":
                        parts += _face_boards(a - hi, b + hi, y0, y1, z0 - _SILL_H - hi, z0 - _SILL_H - lo, trim_bev,
                                              "casing", t)
                # THE LINING covers the reveal from one casing to the other
                ly0, ly1 = -_CASING_PROUD + 0.2, t + _CASING_PROUD - 0.2
                parts.append(_box(a, a + _LINING_T, ly0, ly1, z0, z1, trim_bev, "lining", 1))
                parts.append(_box(b - _LINING_T, b, ly0, ly1, z0, z1, trim_bev, "lining", 1))
                parts.append(_box(a + _LINING_T, b - _LINING_T, ly0, ly1, z1 - _LINING_T, z1, trim_bev, "lining", 1))
                if opening == "window":
                    parts.append(_box(a - cw - _SILL_OVER, b + cw + _SILL_OVER, -_SILL_PROUD, t + _SILL_PROUD,
                                      z0 - _SILL_H, z0, trim_bev, "sill", 1))
                    # THE WINDOW'S FRAME: a ring inside the lining at the middle of the wall, a mullion and a transom
                    fa, fb = a + _LINING_T, b - _LINING_T
                    fz0, fz1 = z0, z1 - _LINING_T
                    fy0, fy1 = t / 2.0 - _FRAME_D / 2.0, t / 2.0 + _FRAME_D / 2.0
                    parts.append(_box(fa, fa + _FRAME_W, fy0, fy1, fz0, fz1, trim_bev, "frame", 1))
                    parts.append(_box(fb - _FRAME_W, fb, fy0, fy1, fz0, fz1, trim_bev, "frame", 1))
                    parts.append(_box(fa + _FRAME_W, fb - _FRAME_W, fy0, fy1, fz0, fz0 + _FRAME_W, trim_bev, "frame", 1))
                    parts.append(_box(fa + _FRAME_W, fb - _FRAME_W, fy0, fy1, fz1 - _FRAME_W, fz1, trim_bev, "frame", 1))
                    mx = (fa + fb) / 2.0
                    by0, by1 = t / 2.0 - _FRAME_D / 2.0 + 1.0, t / 2.0 + _FRAME_D / 2.0 - 1.0
                    parts.append(_box(mx - _BAR_W / 2.0, mx + _BAR_W / 2.0, by0, by1, fz0 + _FRAME_W, fz1 - _FRAME_W,
                                      trim_bev, "frame", 1))
                    tz = fz0 + _FRAME_W + 0.62 * (fz1 - fz0 - 2.0 * _FRAME_W)
                    for t0, t1 in ((fa + _FRAME_W, mx - _BAR_W / 2.0), (mx + _BAR_W / 2.0, fb - _FRAME_W)):
                        parts.append(_box(t0, t1, by0, by1, tz - _BAR_W / 2.0, tz + _BAR_W / 2.0, trim_bev, "frame", 1))
    elif kind == "floor":
        length, width = spec["lengthCm"], spec["widthCm"]
        xs, ys = _with_grid((0.0, length), spec["gridCm"]), _with_grid((0.0, width), spec["gridCm"])
        parts.append(_cells(xs, ys, [[True] * (len(ys) - 1) for _ in range(len(xs) - 1)], 0.0, t,
                            lambda u, v, w: (u, v, w), bev, "slab", 0))
        hulls.append((0.0, length, 0.0, width, 0.0, t))
    else:
        height = spec["heightCm"]
        base_h, cap_h = max(8.0, 0.45 * t), max(6.0, 0.35 * t)
        base_w, cap_w = t + 2.0 * max(3.0, 0.15 * t), t + 2.0 * max(2.5, 0.12 * t)
        c = t / 2.0
        # WITH TRIM THE SHAFT STOPS INSIDE THE BASE AND THE CAPITAL: run to the pillar's foot and top, its end faces would
        # lie in the base's bottom and the capital's top and render black there
        s0, s1 = (base_h / 2.0, height - cap_h / 2.0) if spec["trim"] else (0.0, height)
        step_h, cstep_h = 0.35 * base_h, 0.4 * cap_h
        extra = []
        if spec["trim"] and spec["gridCm"] > 0.0:
            # support lines where the base and the capital meet the shaft, and a hand's width out - as on a wall
            foot, head = base_h + step_h, height - cap_h - cstep_h
            extra = [foot, foot + _SUPPORT_OUT, head, head - _SUPPORT_OUT]
        zs = _with_grid(_with_lines((s0, s1), extra), spec["gridCm"])
        parts.append(_cells((0.0, t), zs, [[True] * (len(zs) - 1)], 0.0, t, lambda u, v, w: (u, w, v), bev, "shaft", 0))
        if spec["trim"]:
            # TWO STEPS EACH: a block and a narrower step toward the shaft; the capital mirrors the base
            parts.append(_box(c - base_w / 2, c + base_w / 2, c - base_w / 2, c + base_w / 2, 0.0, base_h, bev,
                              "base", 1))
            step_w = (base_w + t) / 2.0
            parts.append(_box(c - step_w / 2, c + step_w / 2, c - step_w / 2, c + step_w / 2, base_h, base_h + step_h,
                              bev, "base", 1))
            parts.append(_box(c - cap_w / 2, c + cap_w / 2, c - cap_w / 2, c + cap_w / 2, height - cap_h, height, bev,
                              "capital", 1))
            cstep_w = (cap_w + t) / 2.0
            parts.append(_box(c - cstep_w / 2, c + cstep_w / 2, c - cstep_w / 2, c + cstep_w / 2,
                              height - cap_h - cstep_h, height - cap_h, bev, "capital", 1))
        hulls.append((0.0, t, 0.0, t, 0.0, height))
    return parts, hulls


def _clear_opening(spec):
    """The clear opening asked for, in the slab-corner space of _kit_parts (cm): (x0, x1, z0, z1)."""
    ow, oh, sill, off = spec["openingWidth"], spec["openingHeight"], spec["sillHeight"], spec["openingOffset"]
    x0 = spec["lengthCm"] / 2.0 + off - ow / 2.0
    return x0, x0 + ow, sill, sill + oh


def _rough_opening(spec):
    """The slab's cut round the clear opening (cm): (x0, x1, z1) - wider and taller by the lining when there is trim.
    A window's foot is not lined (its sill sits below it), so the rough and clear feet are the same z0."""
    x0, x1, _z0, z1 = _clear_opening(spec)
    lin = _LINING_T if spec["trim"] else 0.0
    return x0 - lin, x1 + lin, z1 + lin


def _measure_opening(mesh, spec, shift):
    """THE CLEAR OPENING AS BUILT, measured on the render mesh by rays - not the request echoed back. Rays run from
    the opening's middle along -X and +X at three heights and across the reveal's depth, and up (and, for a window,
    down) at three points across it; the nearest hit on each side is the edge. Returns cm, or None when a ray that
    must hit found nothing (a defect in this op)."""
    ca, cb, z0, cz1 = _clear_opening(spec)
    t = spec["thickness"]
    if spec["trim"]:
        # past the lining's 0.5 cm bevel; a window's are in front of and behind its frame, which stands in the middle
        ly0, ly1 = -_CASING_PROUD + 0.2, t + _CASING_PROUD - 0.2
        ys = (ly0 + 0.6, ly1 - 0.6) if spec["opening"] == "window" else (ly0 + 0.6, t / 2.0, ly1 - 0.6)
    else:
        ys = (0.1 * t, 0.5 * t, 0.9 * t)
    tree = BVHTree.FromPolygons([v.co.copy() for v in mesh.vertices], [tuple(p.vertices) for p in mesh.polygons])
    sx, sy, sz = shift

    def cast(x, y, z, d):
        loc = tree.ray_cast(Vector(((x - sx) / 100.0, (y - sy) / 100.0, (z - sz) / 100.0)), Vector(d), 50.0)[0]
        return None if loc is None else Vector((loc.x * 100.0 + sx, loc.y * 100.0 + sy, loc.z * 100.0 + sz))

    cx, h, w = (ca + cb) / 2.0, cz1 - z0, cb - ca
    left, right, top, bottom = [], [], [], []
    for y in ys:
        for f in (0.25, 0.5, 0.75):
            z = z0 + f * h
            hl, hr = cast(cx, y, z, (-1.0, 0.0, 0.0)), cast(cx, y, z, (1.0, 0.0, 0.0))
            left.append(None if hl is None else hl.x)
            right.append(None if hr is None else hr.x)
        for f in (-0.25, 0.0, 0.25):
            hu = cast(cx + f * w, y, z0 + 0.5, (0.0, 0.0, 1.0))
            top.append(None if hu is None else hu.z)
            if z0 > 0.0:
                hd = cast(cx + f * w, y, z0 + 0.5 * h, (0.0, 0.0, -1.0))
                bottom.append(None if hd is None else hd.z)
    if None in left or None in right or None in top or None in bottom:
        return None
    x0, x1, zt = max(left), min(right), min(top)
    zb = max(bottom) if bottom else 0.0
    return {"width": round(x1 - x0, 3), "height": round(zt - zb, 3), "left": round(x0 - ca, 3),
            "right": round(x1 - cb, 3), "foot": round(zb, 3), "top": round(zt, 3),
            "measuredBy": "rays on the built mesh from the opening's middle to the nearest face on each side"}


def _pivot_shift(spec):
    """The offset (cm) subtracted from every coordinate so the chosen pivot lands on the object's origin."""
    kind, pivot, t = spec["kind"], spec["pivot"], spec["thickness"]
    if kind in ("wall",) + _INFILL:
        length = spec["lengthCm"]
        return {"corner": (0.0, 0.0, 0.0), "side": (length / 2.0, 0.0, 0.0), "center": (length / 2.0, t / 2.0, 0.0)}[pivot]
    if kind == "floor":
        length, width = spec["lengthCm"], spec["widthCm"]
        return {"corner": (0.0, 0.0, 0.0), "side": (length / 2.0, 0.0, 0.0),
                "center": (length / 2.0, width / 2.0, 0.0)}[pivot]
    return {"corner": (0.0, 0.0, 0.0), "side": (t / 2.0, 0.0, 0.0), "center": (t / 2.0, t / 2.0, 0.0)}[pivot]


def _assemble(parts, shift, segments):
    """Every part built, bevelled and merged into one bmesh in METERS with the pivot at the origin. Faces are smooth;
    an edge between a bevel face and a flat face is marked sharp, so a flat face shades flat and the rounding shades
    round. Returns the bmesh; the caller frees it."""
    out = bmesh.new()
    sx, sy, sz = shift
    for part in parts:
        pb = _part_bmesh(part)
        try:
            made = _bevel_part(pb, part["bevel"], segments, part.get("square", ()))
            vmap = {}
            for v in pb.verts:
                vmap[v] = out.verts.new(((v.co.x - sx) / 100.0, (v.co.y - sy) / 100.0, (v.co.z - sz) / 100.0))
            fmap = {}
            for f in pb.faces:
                nf = out.faces.new([vmap[v] for v in f.verts])
                nf.smooth = True
                nf.material_index = part["slot"]
                fmap[f] = nf
            for e in pb.edges:
                faces = list(e.link_faces)
                if len(faces) != 2:
                    continue
                sharp = (faces[0] in made) != (faces[1] in made)
                if not made and e.calc_face_angle(0.0) > math.radians(30.0):
                    sharp = True
                if sharp:
                    ne = out.edges.get([vmap[e.verts[0]], vmap[e.verts[1]]])
                    if ne is not None:
                        ne.smooth = False
        finally:
            pb.free()
    out.normal_update()
    return out


def _box_uvs(mesh, tile_m):
    """UV0 as a box projection in meters: each face takes the two axes across its normal's largest component, so a
    wall's face maps x and z, its top x and y, and pieces laid on the module continue one texture."""
    uv = mesh.uv_layers.new(name="UVMap")
    if uv is None:
        raise MifOpError("uv_layers.new() returned None on a new mesh - a defect in this op, not in the request.")
    data = uv.data
    for poly in mesh.polygons:
        n = poly.normal
        ax = max(range(3), key=lambda k: abs(n[k]))
        a, b = {0: (1, 2), 1: (0, 2), 2: (0, 1)}[ax]
        # MIRRORED ON THE BACK FACES so text or grain reads the right way round from either side
        flip = -1.0 if (ax == 0 and n[0] < 0) or (ax == 1 and n[1] > 0) or (ax == 2 and n[2] < 0) else 1.0
        for li in poly.loop_indices:
            co = mesh.vertices[mesh.loops[li].vertex_index].co
            data[li].uv = (flip * co[a] / tile_m, co[b] / tile_m)
    return uv


def _find_type(nodes, type_name):
    return next((n for n in nodes if n.type == type_name), None)


def _sock(sockets, *names):
    for name in names:
        s = sockets.get(name)
        if s is not None:
            return s
    raise MifOpError("a shader socket build_kit_piece needs is missing on this Blender: wanted %s, the node has %s."
                     % (" or ".join(names), ", ".join(s.name for s in sockets)))


def _kit_material(name, color, trim=False):
    """A look for judging the piece in Blender - in Unreal each slot takes the kit's own material. The surface is a
    matte plaster: broad and mid-scale tonal variation and a fine trowel bump; the trim a satin paint, smoother and
    lighter. Noise runs in object space, so it keeps its size whatever the piece's size."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = _find_type(nt.nodes, "BSDF_PRINCIPLED")
    coord = nt.nodes.new("ShaderNodeTexCoord")
    obj_space = _sock(coord.outputs, "Object")

    def noise(scale, detail):
        n = nt.nodes.new("ShaderNodeTexNoise")
        _sock(n.inputs, "Scale").default_value = scale
        _sock(n.inputs, "Detail").default_value = detail
        nt.links.new(obj_space, _sock(n.inputs, "Vector"))
        return _sock(n.outputs, "Fac", "Factor")

    lo, hi = (0.93, 1.02) if trim else (0.80, 1.06)
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.28
    ramp.color_ramp.elements[0].color = tuple(c * lo for c in color) + (1.0,)
    ramp.color_ramp.elements[1].position = 0.72
    ramp.color_ramp.elements[1].color = tuple(min(1.0, c * hi) for c in color) + (1.0,)
    broad = noise(0.7, 4.0)
    mid = noise(5.0, 6.0)
    mix = nt.nodes.new("ShaderNodeMath")
    mix.operation = "MULTIPLY_ADD"
    mix.inputs[1].default_value = 0.35 if trim else 0.6
    nt.links.new(mid, mix.inputs[0])
    nt.links.new(broad, mix.inputs[2])
    nt.links.new(mix.outputs[0], ramp.inputs[0])
    nt.links.new(ramp.outputs[0], _sock(bsdf.inputs, "Base Color"))
    rough = nt.nodes.new("ShaderNodeMapRange")
    _sock(rough.inputs, "To Min").default_value = 0.38 if trim else 0.78
    _sock(rough.inputs, "To Max").default_value = 0.48 if trim else 0.95
    nt.links.new(mid, _sock(rough.inputs, "Value"))
    nt.links.new(_sock(rough.outputs, "Result"), _sock(bsdf.inputs, "Roughness"))
    bump = nt.nodes.new("ShaderNodeBump")
    _sock(bump.inputs, "Strength").default_value = 0.05 if trim else 0.3
    _sock(bump.inputs, "Distance").default_value = 0.002
    nt.links.new(noise(60.0 if trim else 38.0, 10.0), _sock(bump.inputs, "Height"))
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


def _hull_box(name, box_cm, shift, coll, obj):
    """One UCX_ box hull from a box in cm, audited as create_collision_hull audits. Returns (row, None) or (None,
    reason); a hull failing its audit is removed, never kept."""
    x0, x1, y0, y1, z0, z1 = box_cm
    sx, sy, sz = shift
    bm = bmesh.new()
    try:
        vs = [bm.verts.new(((x - sx) / 100.0, (y - sy) / 100.0, (z - sz) / 100.0))
              for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]
        for q in ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)):
            bm.faces.new([vs[i] for i in q])
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
        mesh = bpy.data.meshes.new(name)
        bm.to_mesh(mesh)
    finally:
        bm.free()
    audit = _hull_audit(mesh)
    bad = [k for k in ("boundaryEdges", "nonManifoldEdges", "looseVertices", "convexityViolations") if audit[k]]
    if audit["eulerCharacteristic"] != 2 or audit["volume"] <= 1e-9 or audit["vertices"] > _UE_HULL_VERT_LIMIT:
        bad.append("eulerCharacteristic, volume or vertex count")
    if bad:
        bpy.data.meshes.remove(mesh)
        return None, "the hull %s failed its audit (%s): %s" % (name, ", ".join(bad), audit)
    hull = bpy.data.objects.new(name, mesh)
    hull.display_type = "WIRE"
    hull.hide_render = True
    coll.objects.link(hull)
    hull.matrix_world = obj.matrix_world.copy()
    return {"object": hull.name, "boxCm": [round(v, 3) for v in box_cm], "vertices": audit["vertices"],
            "volumeM3": audit["volume"], "audit": "closed, manifold, convex"}, None


def _kit_name(spec):
    kind = spec["kind"]
    if kind == "wall":
        label = {"none": "Wall", "door": "WallDoor", "window": "WallWindow"}[spec["opening"]]
        size = (spec["lengthCm"], spec["heightCm"])
    elif kind in _INFILL:
        label = "Shutter" if kind == "shutter" else "DoorLeaf"
        size = (spec["openingWidth"], spec["openingHeight"])
    elif kind == "floor":
        label, size = "Floor", (spec["lengthCm"], spec["widthCm"])
    else:
        label, size = "Pillar", (spec["thickness"], spec["heightCm"])
    dims = "x".join(("%d" % round(s)) if abs(s - round(s)) < 1e-6 else ("%g" % s).replace(".", "p") for s in size)
    # ONE PASS OVER THE OBJECTS, then set lookups: a membership test on bpy.data.objects walks the whole list
    names = {o.name for o in bpy.data.objects}
    for n in range(1, 100):
        name = "SM_%s_%s_%02da" % (label, dims, n)
        if name not in names and not any(("UCX_%s_%02d" % (name, k)) in names for k in range(8)):
            return name
    raise MifOpError("SM_%s_%s_01a to _99a are all taken. Pass 'name'. NOTHING was created." % (label, dims))


def _lightmap_res(largest_m):
    """A lightmap resolution by size: 128 for a piece 3 m or more across, 64 for 1-3 m, 32 below."""
    return 128 if largest_m >= 3.0 else (64 if largest_m >= 1.0 else 32)


def _infill_spec(params, spec):
    """Read and check a shutter's or a door leaf's own keys into spec. Every refusal ends NOTHING was created."""
    kind = spec["kind"]
    spec["hardware"] = take_bool(params, "hardware", default=True)
    spec["louver"] = take_bool(params, "louver", default=False) if kind == "door_leaf" else False
    if kind == "shutter":
        lo, hi, dflt = _SHUTTER_PITCH
        pitch = take_float(params, "slatPitch", default=dflt)
        if not lo <= pitch <= hi:
            raise MifOpError("slatPitch is %g cm; a roll-formed slat's pitch must be %g to %g cm. NOTHING was created."
                             % (pitch, lo, hi))
        spec["slatPitch"] = pitch
        lo, hi, dflt = _SHUTTER_HOOD_R
        rad = take_float(params, "hoodRadius", default=dflt)
        if not lo <= rad <= hi:
            raise MifOpError("hoodRadius is %g cm; the hood's front can be rounded 0 (square) to %g cm. NOTHING was "
                             "created." % (rad, hi))
        spec["hoodRadius"] = rad
        top = _shutter_dims(spec)["zt"]
        if top > spec["heightCm"]:
            raise MifOpError("the shutter's hood over a %g cm opening (its coil and a %g cm slat pitch) would stand "
                             "to %g cm, above the %g cm wall. Lower the opening or raise the wall. NOTHING was created."
                             % (spec["openingHeight"], pitch, round(top, 2), spec["heightCm"]))
        return
    d = _leaf_dims(spec)
    lw, lh = d["x"][1] - d["x"][0], d["z"][1] - d["z"][0]
    if spec["hardware"] and (lw < 40.0 or lh < 60.0):
        raise MifOpError("a leaf %g x %g cm has no room for its hinges, lever and deadbolt (at least 40 x 60 cm); "
                         "pass hardware false or a larger opening. NOTHING was created." % (round(lw, 2), round(lh, 2)))
    if spec["louver"] and (lw < 40.0 or lh < _LOUVER_FOOT + _LOUVER_H + 20.0):
        raise MifOpError("a louver needs a leaf at least 40 x %g cm; this one is %g x %g cm. NOTHING was created."
                         % (_LOUVER_FOOT + _LOUVER_H + 20.0, round(lw, 2), round(lh, 2)))


def _slot_bounds(mesh, slot, shift):
    """The bounds (cm, in the wall's slab-corner space) of the vertices of the faces in material slot `slot`, as
    built - what the reply measures an infill by."""
    idx = {i for p in mesh.polygons if p.material_index == slot for i in p.vertices}
    if not idx:
        return None
    co = [mesh.vertices[i].co for i in idx]
    return ([min(c[k] for c in co) * 100.0 + shift[k] for k in range(3)],
            [max(c[k] for c in co) * 100.0 + shift[k] for k in range(3)])


def _infill_report(mesh, spec, shift, parts):
    """The shutter's or leaf's own read-back, measured on the built mesh where it can be."""
    ca, cb, z0, cz1 = _clear_opening(spec)
    counts = {}
    for p in parts:
        counts[p["name"]] = counts.get(p["name"], 0) + 1
    out = {"partCounts": counts, "fills": {"opening": spec["opening"], "clearOpeningCm": {
        "x": [round(ca, 3), round(cb, 3)], "z": [round(z0, 3), round(cz1, 3)]},
        "frame": "the wall's: build the wall with the same length, height, thickness, opening, offset, trim and "
                 "pivot, and put this on the same location and rotation"}}
    b = _slot_bounds(mesh, 0, shift)
    if spec["kind"] == "shutter":
        d = _shutter_dims(spec)
        out["curtain"] = {"slats": counts.get("slat", 0), "slatPitchCm": spec["slatPitch"],
                          "planeYCm": round(d["yc"], 3), "measuredBy": "the surface slot's vertices (slats and bar)"}
        if b:
            out["curtain"]["pastOpeningCm"] = {"left": round(ca - b[0][0], 3), "right": round(b[1][0] - cb, 3)}
            out["curtain"]["spanZCm"] = [round(b[0][2], 3), round(b[1][2], 3)]
        out["hood"] = {"depthCm": round(d["yf"] - d["ym"], 3), "heightCm": round(d["zt"] - d["zh0"], 3),
                       "frontRadiusCm": spec["hoodRadius"], "topCm": round(d["zt"], 3),
                       "coilDiameterCm": round(2.0 * d["rc"], 3), "mountPlaneYCm": round(d["ym"], 3)}
    else:
        out["leaf"] = {"hardware": spec["hardware"], "louver": spec["louver"],
                       "measuredBy": "the surface slot's vertices (the leaf)"}
        if b:
            out["leaf"].update({"widthCm": round(b[1][0] - b[0][0], 3), "heightCm": round(b[1][2] - b[0][2], 3),
                                "thicknessCm": round(b[1][1] - b[0][1], 3),
                                "clearanceCm": {"hingeSide": round(b[0][0] - ca, 3), "latchSide": round(cb - b[1][0], 3),
                                                "head": round(cz1 - b[1][2], 3), "undercut": round(b[0][2] - z0, 3)}})
        hole = _leaf_dims(spec)["hole"]
        if hole:
            out["leaf"]["louverCm"] = {"x": [round(hole[0], 3), round(hole[1], 3)],
                                       "z": [round(hole[2], 3), round(hole[3], 3)]}
    return out


def op_build_kit_piece(params):
    """Build one modular kit piece - a wall, a wall with a door or window opening, a floor or a pillar - on a module
    grid: sides in whole modules, the pivot at the bottom (a corner by default; a pillar's bottom center), every edge
    bevelled, profiled trim (a plinth, a casing and lining round an opening, a window's sill and frame, a pillar's base
    and capital), a surface slot and a trim slot, UV0 in meters, a lightmap UV layer last and UCX_ box hulls that leave
    the opening open. Sizes are real: the object is in meters, built from centimeters. Rotation is in radians.

    WHAT FILLS THE OPENING (2026-10-06): kind shutter or door_leaf builds it in the frame of the wall it fills - pass
    that wall's length, height, thickness, opening, openingWidth/Height, sillHeight, openingOffset, trim and pivot, and
    the same location and rotation, and it lands in that wall's opening. THE GAP: the openings were filled with
    stacked boxes (a box hood, box slats, a slab door with box panels), and v2 came from plain-Python scripts. A
    shutter is a curtain of roll-formed slats with an interlock curl, a bottom angle bar on a PVC bulb astragal with a
    lift handle, channel guides on jamb angles with bolts, a sheet hood with a rounded front over the coil and end
    brackets - closed, on the wall's face (in front of its casing when there is trim). A door_leaf is a 1-3/4 in
    hollow-metal leaf with hemmed edges, 3 mm clear of the jambs and head with a 15 mm undercut; hardware adds butt
    hinges, a lever and a deadbolt on both faces, a kick plate, an astragal and a drip cap over the opening; louver a
    stamped chevron louver. Every part is a closed solid; a thin formed sheet is capped with quads between its two
    runs, never an n-gon that triangulates into fans. Slot 0 is the curtain or leaf, slot 1 the hardware.

    params:
      kind (str)            wall (default), floor, pillar, shutter, door_leaf
      opening (str)         none (default), door, window - a wall; a shutter door (default) or window; a door_leaf door
      slatPitch (cm)        a shutter's slat pitch, default 6.7; 3 to 15
      hoodRadius (cm)       the rounding of a shutter hood's front, default 8; 0 (square) to 15
      hardware (bool)       a shutter's bolts and lift handle, a leaf's hinges, lever, deadbolt, kick plate, astragal
                            and drip cap; default true
      louver (bool)         a door leaf's stamped louver, default false
      module (cm)           the grid, default 100; 10 to 1000
      length (modules)      a wall's or a floor's length along X, default 4
      height (modules)      a wall's or a pillar's height, default 3
      width (modules)       a floor's depth along Y, default 4
      thickness (cm)        a wall's (default 12) or a floor's (default 7) slab, or a pillar's side (default 40)
      openingWidth, openingHeight (cm)   the CLEAR opening, default 110 x 220 (door) or 100 x 120 (window); the
                            lining sits outside it in the slab's reveal. The reply's clearOpeningCm is it measured on
                            the mesh, roughOpeningCm the slab's cut
      sillHeight (cm)       a window's sill above the floor, default 90; a door's is 0
      openingOffset (cm)    the opening's center from the wall's center along X, default 0
      pivot (str)           corner, side (the middle of the -Y side) or center; walls and floors default corner, a
                            pillar center; always at the bottom, which is where floor modules tile from too
      bevel (cm)            edge rounding, default 1 (1.5 on a pillar); 0 leaves hard edges
      bevelSegments (int)   1 to 8, default 3
      trim (bool)           plinth, casing and sill on a wall, base and capital on a pillar; default true
      uvTileCm (cm)         meters of surface per UV0 repeat, in cm; default 200
      gridCm (cm)           extra grid lines across the slab's (a pillar's shaft's) faces every this many cm, so a
                            vertex-color mask has vertices to vary across; default 50, 0 for none
      color ([r,g,b])       slot 0's made material (the surface), linear 0-1; default a warm light gray, a
                            concrete gray on a floor
      material (str)        an existing material for slot 0 instead of making one
      trimColor ([r,g,b])   slot 1's made material (plinth, casing, lining, sill, window frame, a pillar's base and
                            capital); default an off-white
      trimMaterial (str)    an existing material for slot 1
      collision (bool)      UCX_<name>_NN box hulls; default true
      name (str)            default SM_<Kind>_<size in cm>_01a, the first number free
      location, rotation ([x,y,z] or {x,y,z}), collection
      minThicknessCm (cm)   a thickness the caller holds the piece to (2026-10-11): the reply says belowMinThickness
                            and adds a note when the piece is thinner. Never a refusal and never a changed default -
                            the kit defaults stay as they are; the figure depends on the renderer's tracing mode
    """
    reject_unknown(params, BUILD_KIT_PIECE_PARAMS, "build_kit_piece", _KIT_NOTES)
    min_t = take_float(params, "minThicknessCm")
    if min_t is not None and not (math.isfinite(min_t) and min_t > 0.0):
        raise MifOpError("minThicknessCm is a thickness in CENTIMETERS above 0, not %r. NOTHING was created." % min_t)
    started = time.perf_counter()
    # EVERY VALUE IS READ AND CHECKED BEFORE ANYTHING IS CREATED
    kind = take(params, "kind", default="wall", kind=str)
    if kind not in _KINDS:
        raise MifOpError("kind must be one of %s, got '%s'. NOTHING was created." % (", ".join(_KINDS), kind))
    d = _DEFAULTS[kind]
    walled = kind == "wall" or kind in _INFILL
    opening = take(params, "opening", default="door" if kind in _INFILL else "none", kind=str)
    if opening not in _OPENINGS:
        raise MifOpError("opening must be one of %s, got '%s'. NOTHING was created." % (", ".join(_OPENINGS), opening))
    if opening != "none" and not walled:
        raise MifOpError("an opening is cut in a wall; kind is '%s'. NOTHING was created." % kind)
    if kind in _INFILL and opening == "none":
        raise MifOpError("a %s fills a wall's opening; opening is none. Pass the opening (door or window) and the "
                         "sizes of the wall it fills. NOTHING was created." % kind)
    if kind == "door_leaf" and opening != "door":
        raise MifOpError("a door leaf hangs in a door opening; opening is '%s'. NOTHING was created." % opening)
    for key, kinds in (("slatPitch", ("shutter",)), ("hoodRadius", ("shutter",)), ("hardware", _INFILL),
                       ("louver", ("door_leaf",))):
        if params.get(key) is not None and kind not in kinds:
            raise MifOpError("'%s' shapes a %s; kind is '%s'. NOTHING was created." % (key, " or ".join(kinds), kind))
    if kind == "shutter" and params.get("gridCm") is not None:
        raise MifOpError("gridCm adds lines across a slab's face; a shutter's slats have no slab face. NOTHING was "
                         "created.")
    module = take_float(params, "module", default=100.0)
    if not 10.0 <= module <= 1000.0:
        raise MifOpError("module is %g cm; it must be 10 to 1000 cm. NOTHING was created." % module)
    used = {"floor": ("length", "width"), "pillar": ("height",)}.get(kind, ("length", "height"))
    for key in ("length", "height", "width"):
        if key not in used and params.get(key) is not None:
            raise MifOpError("'%s' does not size a %s (it takes %s). NOTHING was created."
                             % (key, kind, " and ".join(used)))
    spec = {"kind": kind, "opening": opening, "module": module}
    for key in used:
        count = take_float(params, key, default=d[key])
        cm = count * module
        if count <= 0.0 or not 10.0 <= cm <= _MAX_SIDE_CM:
            raise MifOpError("%s is %g modules (%g cm); a side must be 10 to %g cm. NOTHING was created."
                             % (key, count, cm, _MAX_SIDE_CM))
        spec[key + "Cm"] = cm
        spec[key + "Modules"] = count
    t = take_float(params, "thickness", default=d["thickness"])
    t_max = {"floor": 100.0, "pillar": 300.0}.get(kind, 200.0)
    if not 1.0 <= t <= t_max:
        raise MifOpError("thickness is %g cm; a %s's must be 1 to %g cm. NOTHING was created." % (t, kind, t_max))
    if walled and t > min(spec["lengthCm"], spec["heightCm"]) / 2.0:
        raise MifOpError("thickness %g cm is more than half the wall's shorter side; that is a block, not a wall. "
                         "NOTHING was created." % t)
    if kind == "pillar" and t > spec["heightCm"]:
        raise MifOpError("a pillar %g cm thick and %g cm tall is wider than it is tall. NOTHING was created."
                         % (t, spec["heightCm"]))
    spec["thickness"] = t
    for key in ("openingWidth", "openingHeight", "sillHeight", "openingOffset"):
        if opening == "none" and params.get(key) is not None:
            raise MifOpError("'%s' shapes an opening and opening is none. NOTHING was created." % key)
    trim = take_bool(params, "trim", default=True)
    spec["trim"] = trim
    if opening != "none":
        dw, dh, ds = _OPENING_DEFAULTS[opening]
        ow = take_float(params, "openingWidth", default=dw)
        oh = take_float(params, "openingHeight", default=dh)
        sill = take_float(params, "sillHeight", default=ds)
        off = take_float(params, "openingOffset", default=0.0)
        if opening == "door" and sill != 0.0:
            raise MifOpError("a door reaches the floor, so its sillHeight is 0; got %g cm. Use opening window for a "
                             "raised opening. NOTHING was created." % sill)
        length, height = spec["lengthCm"], spec["heightCm"]
        casing = _CASING_W if trim else 0.0
        margin = _MIN_MARGIN + casing
        if ow < 20.0 or oh < 20.0:
            raise MifOpError("the opening is %g x %g cm; each side must be at least 20 cm. NOTHING was created."
                             % (ow, oh))
        spec.update({"openingWidth": ow, "openingHeight": oh, "sillHeight": sill, "openingOffset": off})
        # THE MARGINS ARE KEPT ROUND THE ROUGH OPENING: the clear opening plus its lining, the casing outside that
        lined = (" (with its %g cm lining)" % _LINING_T) if trim else ""
        a, b, top = _rough_opening(spec)
        if a < margin or b > length - margin:
            raise MifOpError("an opening %g cm wide%s at offset %g cm runs from %g to %g cm along a %g cm wall; it "
                             "must keep %g cm of wall (with its casing) at each end. NOTHING was created."
                             % (ow, lined, off, a, b, length, margin))
        if sill < 0.0 or (opening == "window" and sill < margin + (_SILL_H if trim else 0.0)):
            raise MifOpError("sillHeight %g cm leaves too little wall below the window: at least %g cm. NOTHING was "
                             "created." % (sill, margin + (_SILL_H if trim else 0.0)))
        if top > height - margin:
            raise MifOpError("the opening's top%s is at %g cm in a %g cm wall; it must keep %g cm of wall above it. "
                             "NOTHING was created." % (lined, top, height, margin))
    pivot = take(params, "pivot", default=d["pivot"], kind=str)
    if pivot not in _PIVOTS:
        raise MifOpError("pivot must be one of %s, got '%s'. NOTHING was created." % (", ".join(_PIVOTS), pivot))
    spec["pivot"] = pivot
    bev = take_float(params, "bevel", default=d["bevel"])
    bev_max = min(t * 0.3, 1.0) if kind in _INFILL else t * 0.3
    if not 0.0 <= bev <= bev_max:
        raise MifOpError("bevel is %g cm; it must be 0 to %g cm (30%% of the thickness%s), or the rounding eats the "
                         "piece. NOTHING was created." % (bev, bev_max, ", and 1 cm on a leaf's hem"
                                                          if kind in _INFILL else ""))
    spec["bevel"] = bev
    segments = take_int(params, "bevelSegments", default=3)
    if not 1 <= segments <= 8:
        raise MifOpError("bevelSegments is %d; it must be 1 to 8. NOTHING was created." % segments)
    grid = take_float(params, "gridCm", default=50.0)
    if grid != 0.0 and not 5.0 <= grid <= 1000.0:
        raise MifOpError("gridCm is %g; it must be 0 (no extra lines) or 5 to 1000 cm. NOTHING was created." % grid)
    spec["gridCm"] = grid
    if kind in _INFILL:
        _infill_spec(params, spec)
    tile = take_float(params, "uvTileCm", default=200.0)
    if not 10.0 <= tile <= 10000.0:
        raise MifOpError("uvTileCm is %g; it must be 10 to 10000 cm per repeat. NOTHING was created." % tile)
    color = _color3(params, "color", {"floor": (0.40, 0.39, 0.37), "shutter": (0.55, 0.56, 0.57),
                                      "door_leaf": (0.30, 0.36, 0.42)}.get(kind, (0.62, 0.60, 0.56)))
    trim_color = _color3(params, "trimColor", (0.45, 0.46, 0.47) if kind in _INFILL else (0.80, 0.79, 0.75))
    has_trim_slot = trim and kind not in ("floor",) + _INFILL
    if kind in _INFILL:
        # SLOT 1 IS THE HARDWARE: a shutter's guides, hood and coil are always there; a leaf has it with hardware
        # or a louver
        has_trim_slot = kind == "shutter" or spec["hardware"] or spec["louver"]
    materials = []
    for mkey, ckey, want in (("material", "color", True), ("trimMaterial", "trimColor", has_trim_slot)):
        mat_name = take(params, mkey, kind=str)
        if not want and (mat_name is not None or params.get(ckey) is not None):
            raise MifOpError("'%s' colors the trim slot, and this %s has no trim%s. NOTHING was created."
                             % (mkey if mat_name is not None else ckey, kind,
                                " (no hardware and no louver)" if kind in _INFILL else ""))
        if mat_name is not None and params.get(ckey) is not None:
            raise MifOpError("'%s' reuses an existing material and '%s' colors the one made here; pass one. NOTHING "
                             "was created." % (mkey, ckey))
        if mat_name is not None and bpy.data.materials.get(mat_name) is None:
            raise MifOpError("no material named '%s'. NOTHING was created." % mat_name)
        if want:
            materials.append(bpy.data.materials.get(mat_name) if mat_name is not None else None)
    want_hulls = take_bool(params, "collision", default=True)
    name = take(params, "name", kind=str)
    if name is None:
        name = _kit_name(spec)
    else:
        if not name.strip() or any(ord(ch) < 32 for ch in name):
            raise MifOpError("name cannot be empty or hold a control character, got %r. NOTHING was created." % name)
        if name in bpy.data.objects:
            raise MifOpError("an object named '%s' already exists; Blender would name this one '%s.001', and a kit "
                             "piece's name is its asset name. NOTHING was created." % (name, name))
    if want_hulls and len(name) > 56:
        raise MifOpError("name is %d characters; with collision on it can be at most 56, so UCX_<name>_NN fits the "
                         "63 characters Blender keeps in a name. NOTHING was created." % len(name))
    try:
        placed = _place_values(params)
    except MifOpError as exc:
        raise MifOpError("%s. NOTHING was created." % str(exc).rstrip("."))
    coll = _resolve_collection(params)

    parts, hull_boxes = _kit_parts(spec)
    if want_hulls:
        names = {o.name for o in bpy.data.objects}
        taken = [("UCX_%s_%02d" % (name, k)) for k in range(len(hull_boxes)) if ("UCX_%s_%02d" % (name, k)) in names]
        if taken:
            raise MifOpError("%s already exist%s, so the hulls would be suffixed .001 and attach to nothing. Remove "
                             "them, pass another name, or collision false. NOTHING was created."
                             % (", ".join(taken), "s" if len(taken) == 1 else ""))
    shift = _pivot_shift(spec)

    # ---- the mesh -----------------------------------------------------------------------------------------------
    bm = _assemble(parts, shift, segments)
    mesh = bpy.data.meshes.new(name)
    try:
        bm.to_mesh(mesh)
    finally:
        bm.free()
    # UP TO 4.0 SHARP EDGES SHADE ONLY WITH AUTO SMOOTH ON; from 4.1 the property is gone and sharp edges always shade
    if hasattr(mesh, "use_auto_smooth"):
        mesh.use_auto_smooth = True
        mesh.auto_smooth_angle = math.pi
    _box_uvs(mesh, tile / 100.0)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    _link_new(obj, coll)
    _place(obj, placed)
    slots = []
    for i, mat in enumerate(materials):
        made_mat = mat is None
        if made_mat:
            mat = _kit_material("M_%s%s" % (name, "_Trim" if i else ""), trim_color if i else color, trim=bool(i))
        mesh.materials.append(mat)
        slots.append({"slot": i, "part": "trim" if i else "surface", "material": mat.name, "made": made_mat})

    # ---- the lightmap layer, LAST, made by uv_unwrap and judged by uv_info --------------------------------------
    sel = selection_snapshot()
    try:
        lm = op_uv_unwrap({"object": obj.name, "method": "SMART", "uvLayer": _LIGHTMAP_LAYER, "islandMargin": 0.02})
        layers = {r.get("name"): r for r in (op_uv_info({"object": obj.name, "layer": _LIGHTMAP_LAYER}).get("layers")
                                             or [])}
        lm_row = layers.get(_LIGHTMAP_LAYER) or {}
        method = "SMART"
        if not lm_row.get("lightmapReady"):
            # A SMART LAYOUT WITH TOUCHING ISLANDS IS REPACKED one face group per island, which never touches
            op_uv_unwrap({"object": obj.name, "method": "LIGHTMAP", "uvLayer": _LIGHTMAP_LAYER, "replace": True,
                          "islandMargin": 0.05})
            layers = {r.get("name"): r for r in
                      (op_uv_info({"object": obj.name, "layer": _LIGHTMAP_LAYER}).get("layers") or [])}
            lm_row = layers.get(_LIGHTMAP_LAYER) or {}
            method = "LIGHTMAP"
    finally:
        selection_restore(sel)
    mesh.uv_layers.active = mesh.uv_layers["UVMap"]
    try:
        mesh.uv_layers["UVMap"].active_render = True
    except (AttributeError, TypeError):
        pass

    hulls, hull_notes = [], []
    if want_hulls:
        for k, box in enumerate(hull_boxes):
            row, note = _hull_box("UCX_%s_%02d" % (name, k), box, shift, coll, obj)
            if row:
                hulls.append(row)
            else:
                hull_notes.append(note)

    out = _created(obj, coll, "MESH", "build_kit_piece", params.get("name"))
    co = [v.co for v in mesh.vertices]
    lo = [min(c[i] for c in co) for i in range(3)]
    hi = [max(c[i] for c in co) for i in range(3)]
    tris = sum(len(p.vertices) - 2 for p in mesh.polygons)
    sides = {k: round(spec[k + "Cm"], 3) for k in used}
    on_module = {k: abs(spec[k + "Modules"] - round(spec[k + "Modules"])) < 1e-6 for k in used}
    largest = max(top - bottom for top, bottom in zip(hi, lo))
    out.update({
        "kind": kind, "opening": opening, "moduleCm": module, "sidesCm": sides, "onModule": on_module,
        "fittedFiller": not all(on_module.values()),
        "thicknessCm": t, "pivot": pivot,
        "pivotAt": {"wall": {"corner": "the slab's bottom corner (x 0, y 0 at its back face, z 0)",
                             "side": "the middle of the slab's back face, at the bottom",
                             "center": "the slab's bottom center"},
                    "floor": {"corner": "a bottom corner of the slab", "side": "the middle of its -Y edge, at the bottom",
                              "center": "the center of its underside"},
                    "pillar": {"corner": "the shaft's bottom corner", "side": "the middle of the shaft's -Y side, at "
                               "the bottom", "center": "the shaft's bottom center"}}.get(kind, {
                        "corner": "the bottom corner of the wall it fills (x 0, y 0 at the wall's back face, z 0)",
                        "side": "the middle of that wall's back face, at the bottom",
                        "center": "that wall's bottom center"})[pivot],
        "boundsLocalCm": {"min": [round(v * 100.0, 3) for v in lo], "max": [round(v * 100.0, 3) for v in hi]},
        "parts": ([p["name"] for p in parts] if kind not in _INFILL
                  else list(dict.fromkeys(p["name"] for p in parts))), "bevelCm": bev, "bevelSegments": segments, "trim": trim, "gridCm": grid,
        "vertices": len(mesh.vertices), "faces": len(mesh.polygons), "triangles": tris,
        "ngons": sum(1 for p in mesh.polygons if len(p.vertices) > 4),
        "uvLayers": [u.name for u in mesh.uv_layers],
        "uv0": {"name": "UVMap", "projection": "box, in meters", "cmPerRepeat": tile,
                "continuesAcrossModule": bool(abs((module / tile) - round(module / tile)) < 1e-6 or
                                              abs((tile / module) - round(tile / module)) < 1e-6)},
        "lightmap": {"layer": _LIGHTMAP_LAYER, "index": len(mesh.uv_layers) - 1, "method": method,
                     "lightmapReady": bool(lm_row.get("lightmapReady")),
                     "reason": lm_row.get("lightmapReadyReason"), "islands": lm_row.get("islandCount"),
                     "resolutionSuggested": _lightmap_res(largest),
                     "unwrapWarnings": (lm or {}).get("warnings") or None},
        "materialSlots": slots,
        "collisionHulls": hulls, "collisionNotes": hull_notes or None,
        "elapsedSeconds": round(time.perf_counter() - started, 2),
    })
    if opening != "none":
        out["openingCm"] = {"width": spec["openingWidth"], "height": spec["openingHeight"],
                            "sill": spec["sillHeight"], "offset": spec["openingOffset"]}
    if kind in _INFILL:
        out.update(_infill_report(mesh, spec, shift, parts))
    elif opening != "none":
        ra, rb, rtop = _rough_opening(spec)
        out["roughOpeningCm"] = {
            "width": round(rb - ra, 3), "height": round(rtop - spec["sillHeight"], 3),
            "note": ("the slab's cut: the clear opening plus the %g cm lining on each side and on top" % _LINING_T)
            if trim else "the slab's cut; with trim false there is no lining, so it is the clear opening"}
        measured = _measure_opening(mesh, spec, shift)
        out["clearOpeningCm"] = measured
        if measured is None or abs(measured["width"] - spec["openingWidth"]) > 0.05 \
                or abs(measured["height"] - spec["openingHeight"]) > 0.05:
            out.setdefault("warnings", []).append(
                "the clear opening measured on the mesh is %s, not the %g x %g cm asked for - a defect in this op; "
                "measure it before relying on it"
                % ("unmeasurable" if measured is None else "%g x %g cm" % (measured["width"], measured["height"]),
                   spec["openingWidth"], spec["openingHeight"]))
    if min_t is not None:
        out["minThicknessCm"] = min_t
        out["belowMinThickness"] = out.get("thicknessCm") is not None and out["thicknessCm"] < min_t
        if out["belowMinThickness"]:
            out.setdefault("notes", []).append(
                "the piece is %g cm thick, under the %g cm asked for - nothing was changed; build it with thickness "
                "%g or more where that figure matters" % (out["thicknessCm"], min_t, min_t))
    return out


# ------------------------------------------------------------------------------------------ set_vertex_color_by_rule
_RULES = ("up", "height", "cavity", "random", "dirt", "fromWeight", "adjust")
_ADJUST_KEYS = {"levels", "hsv", "brightnessContrast"}


def _dirt_tones(mesh, blur_strength, blur_iterations, clean, dirt, dirt_only, normalize):
    """Blender's Dirty Vertex Colors, per vertex (5.2 bl_operators/vertexpaint_dirt.py:136, 3.6 :133): the angle between
    the normal and the mean direction to the connected vertices (under 90 degrees in a crease), clamped, blurred over the
    edges, then mapped to 0-1 between the lowest and highest tone (normalize) or the two clamp angles. 0 is dirt."""
    nv = len(mesh.vertices)
    con = [[] for _ in range(nv)]
    for e in mesh.edges:
        a, b = e.vertices
        con[a].append(b)
        con[b].append(a)
    tone = [0.0] * nv
    for i, v in enumerate(mesh.vertices):
        if not con[i]:
            ang = math.pi / 2.0
        else:
            vec = Vector()
            for c in con[i]:
                vec += (mesh.vertices[c].co - v.co).normalized()
            vec /= len(con[i])
            ang = math.acos(max(-1.0, min(1.0, v.normal.dot(vec))))
        ang = max(dirt, ang)
        if not dirt_only:
            ang = min(clean, ang)
        tone[i] = ang
    for _ in range(blur_iterations):
        orig = tone[:]
        for j, c in enumerate(con):
            for k in c:
                tone[j] += blur_strength * orig[k]
            tone[j] /= len(c) * blur_strength + 1
    lo, hi = (min(tone), max(tone)) if normalize else (dirt, clean)
    span = hi - lo
    scale = 0.0 if span < 1e-4 else 1.0 / span
    out = []
    for t in tone:
        x = (t - lo) * scale
        if dirt_only:
            x = min(x, 0.5) * 2.0
        out.append(x)
    return out


def _adjust_plan(spec):
    """adjust {levels {offset, gain}, hsv {hue, saturation, value}, brightnessContrast {brightness, contrast}}, checked."""
    if not isinstance(spec, dict) or not spec or set(spec) - _ADJUST_KEYS:
        raise MifOpError("adjust is {levels: {offset, gain}, hsv: {hue, saturation, value}, brightnessContrast: "
                         "{brightness, contrast}} - one or more; got %r. NOTHING was changed." % (spec,))
    plan = {}
    for key, fields, defaults in (("levels", ("offset", "gain"), (0.0, 1.0)),
                                  ("hsv", ("hue", "saturation", "value"), (0.5, 1.0, 1.0)),
                                  ("brightnessContrast", ("brightness", "contrast"), (0.0, 0.0))):
        sub = spec.get(key)
        if sub is None:
            continue
        if not isinstance(sub, dict) or set(sub) - set(fields):
            raise MifOpError("adjust.%s takes %s, got %r. NOTHING was changed." % (key, ", ".join(fields), sub))
        plan[key] = tuple(finite_float(sub.get(f, d), "adjust.%s.%s" % (key, f)) for f, d in zip(fields, defaults))
    if "brightnessContrast" in plan and not all(-100.0 <= x <= 100.0 for x in plan["brightnessContrast"]):
        raise MifOpError("brightness and contrast are -100 to 100, as Blender's Brightness/Contrast. NOTHING was "
                         "changed.")
    if "hsv" in plan and not 0.0 <= plan["hsv"][0] <= 1.0:
        raise MifOpError("adjust.hsv.hue is 0 to 1 (0.5 leaves it), as Blender's Hue/Saturation/Value. NOTHING was "
                         "changed.")
    return plan


def _bc_coeffs(brightness, contrast):
    """Blender's vertex_color_brightness_contrast: new = a x old + b (paint_vertex_color_ops.cc)."""
    brightness /= 100.0
    delta = contrast / 200.0
    if contrast > 0:
        a = 1.0 / max(1e-6, 1.0 - delta * 2.0)
        b = a * (brightness - delta)
    else:
        delta = -delta
        a = max(0.0, 1.0 - delta * 2.0)
        b = a * brightness + delta
    return a, b
_CHANNELS = {"R": (0,), "G": (1,), "B": (2,), "A": (3,), "RGB": (0, 1, 2)}
_COMBINES = ("replace", "multiply", "max", "min", "add")
_RULE_RANGE = {"up": (0.5, 0.9), "cavity": (0.1, 0.5), "random": (0.0, 1.0)}


def _range(params, rule):
    raw = params.get("range")
    if raw is None:
        return None
    if (not isinstance(raw, (list, tuple)) or len(raw) != 2
            or any(isinstance(c, bool) or not isinstance(c, (int, float)) or not math.isfinite(c) for c in raw)):
        raise MifOpError("'range' takes [from, to] as two finite numbers, got %r. NOTHING was changed." % (raw,))
    a, b = finite_floats(raw, "range")
    if a == b:
        raise MifOpError("'range' [%g, %g] has no width, so every value would be 0 or 1. NOTHING was changed."
                         % (a, b))
    if rule == "random" and not (0.0 <= a <= 1.0 and 0.0 <= b <= 1.0):
        raise MifOpError("for rule random, 'range' is the values written, each 0 to 1; got %r. NOTHING was changed."
                         % (raw,))
    return a, b


def _components(n_verts, edges):
    """A component id per vertex (connected by edges), numbered by each component's lowest vertex index."""
    parent = list(range(n_verts))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(0, len(edges), 2):
        ra, rb = find(edges[i]), find(edges[i + 1])
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb
    return [find(v) for v in range(n_verts)]


def _corner_normals(mesh):
    """Each face corner's shading normal, in the object's space - the one sharp edges split. From 4.1 Blender keeps
    them in mesh.corner_normals; up to 4.0 calc_normals_split() fills loop.normal (and needs auto smooth on for sharp
    edges to split them, which build_kit_piece switches on)."""
    n = len(mesh.loops)
    flat = [0.0] * (3 * n)
    # 3.6 HAS A corner_normals TOO, and reading it raised "internal error setting the array" there (3.6.23,
    # 2026-09-29): it is not the 4.1 array. So the split-normal path runs wherever the 4.1 one fails or is short.
    cn = getattr(mesh, "corner_normals", None)
    try:
        if cn is None or len(cn) != n:
            raise RuntimeError("no 4.1 corner normals")
        cn.foreach_get("vector", flat)
    except (RuntimeError, TypeError, AttributeError):
        mesh.calc_normals_split()
        mesh.loops.foreach_get("normal", flat)
    return [Vector(flat[3 * i:3 * i + 3]) for i in range(n)]


def _fibonacci_hemisphere(n):
    """n directions spread evenly over the +Z hemisphere, cosine-weighted, the same every call."""
    out = []
    golden = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n):
        z = math.sqrt(1.0 - (i + 0.5) / n)
        r = math.sqrt(max(0.0, 1.0 - z * z))
        a = golden * i
        out.append(Vector((r * math.cos(a), r * math.sin(a), z)))
    return out


def _cavity(obj, mesh, distance, rays):
    """Per vertex, the share of short rays over its normal's hemisphere that hit the mesh within `distance` meters -
    1 in a crevice or where two parts meet, 0 on an open face or an outside corner. Rays are cast in the object's
    world space, so scale counts."""
    mw = obj.matrix_world
    nm = mw.to_3x3().inverted_safe().transposed()
    world = [mw @ v.co for v in mesh.vertices]
    normals = [(nm @ v.normal).normalized() for v in mesh.vertices]
    polys = [tuple(p.vertices) for p in mesh.polygons]
    tree = BVHTree.FromPolygons(world, polys, epsilon=0.0)
    dirs = _fibonacci_hemisphere(rays)
    out = []
    eps = max(1e-5, distance * 1e-3)
    for p, n in zip(world, normals):
        if n.length < 0.5:
            out.append(0.0)
            continue
        rot = n.to_track_quat("Z", "Y").to_matrix()
        origin = p + n * eps
        hits = 0
        for d in dirs:
            loc, _nrm, _idx, _dist = tree.ray_cast(origin, rot @ d, distance)
            if loc is not None:
                hits += 1
        out.append(hits / float(rays))
    return out


def op_set_vertex_color_by_rule(params):
    """Write a 0-1 mask into a color attribute by a rule - what a material blends moss, dirt, grime or per-piece
    variation with. The value is stored as value x 255 in an 8-bit (BYTE_COLOR) attribute, which the FBX export
    carries and Unreal's VertexColor node reads as written.

    rules:
      up        how much the surface faces world +Z: the shading normal's Z (per face corner, so sharp edges keep a
                top face up to its edge; per vertex on a POINT attribute) mapped through range (default 0.5 to 0.9)
                - moss on tops
      height    the vertex's height, from the object's lowest point (0) to its highest (1) by default, or through
                range given in meters of world Z - with invert, dirt at the foot
      cavity    how enclosed each vertex is: the share of `rays` short rays over its hemisphere that meet the mesh
                within `distance` meters, mapped through range (default 0.1 to 0.5) - grime in crevices and where parts
                meet
      random    one value per element, uniform in range (default 0 to 1) - per-piece variation
      dirt      Blender's Dirty Vertex Colors (second 1.3 audit, 2026-10-25): 1 in creases, 0 on open and worn surface -
                dirt {blurStrength (0.01-1, default 1), blurIterations (0-40, default 1), cleanAngle / dirtAngle
                (degrees, default 180 / 0), dirtOnly, normalize (default true)}; occlusion-like grime without rays
      fromWeight  a vertex group's weights (group) into the channel, through range (default 0 to 1) - a wind or wetness
                mask authored as weights, moved to where Unreal's VertexColor node reads it
      adjust    change what the attribute already holds, no mask computed: adjust {levels {offset, gain}: gain x (value +
                offset); hsv {hue (0.5 leaves it), saturation, value}: RGB only; brightnessContrast {brightness,
                contrast} -100 to 100} - Blender's Levels, Hue/Saturation/Value and Brightness/Contrast, in that order,
                then invert; the attribute must exist

    params:
      object (str)          required, a mesh in OBJECT mode
      rule (str)            required: up, height, cavity, random
      name (str)            the color attribute, default Col; made on face corners if it does not exist
      channel (str)         R (default), G, B, A or RGB (all three)
      range ([from, to])    what maps to 0 and to 1 (for random, the values written); from > to reverses it
      invert (bool)         write 1 - value
      combine (str)         replace (default), multiply, max, min, add - with what the channel holds
      seed (int)            random: default 1
      element (str)         random: part (each connected piece, default) or face - face needs a CORNER attribute
      distance (m)          cavity: default 5% of the object's largest world side
      rays (int)            cavity: 4 to 256, default 24
      space (str)           up and height: world (default) or local
      makeActive (bool)     make the attribute the active and render color, default true
    """
    reject_unknown(params, VERTEX_RULE_PARAMS, "set_vertex_color_by_rule", _VCOL_NOTES)
    started = time.perf_counter()
    # EVERY VALUE IS READ AND CHECKED BEFORE THE OBJECT IS LOOKED UP, and the object before anything is written
    obj_name = take(params, "object", required=True, kind=str)
    rule = take(params, "rule", required=True, kind=str)
    if rule not in _RULES:
        raise MifOpError("rule must be one of %s, got '%s'. NOTHING was changed." % (", ".join(_RULES), rule))
    name = take(params, "name", default="Col", kind=str)
    if not name or len(name) > 63:
        raise MifOpError("the attribute name must be 1 to 63 characters, got %r. NOTHING was changed." % name)
    channel = take(params, "channel", default="R", kind=str).upper()
    if channel not in _CHANNELS:
        raise MifOpError("channel must be R, G, B, A or RGB, got '%s'. NOTHING was changed." % channel)
    rng_in = _range(params, rule)
    invert = take_bool(params, "invert", default=False)
    combine = take(params, "combine", default="replace", kind=str)
    if combine not in _COMBINES:
        raise MifOpError("combine must be one of %s, got '%s'. NOTHING was changed." % (", ".join(_COMBINES), combine))
    make_active = take_bool(params, "makeActive", default=True)
    only = {"seed": "random", "element": "random", "distance": "cavity", "rays": "cavity", "space": "up or height",
            "dirt": "dirt", "group": "fromWeight", "adjust": "adjust"}
    for key, owner in only.items():
        if params.get(key) is not None and rule not in owner.split(" or "):
            raise MifOpError("'%s' is read by rule %s only; rule is %s. It is refused rather than ignored. NOTHING "
                             "was changed." % (key, owner, rule))
    seed = take_int(params, "seed", default=1)
    element = take(params, "element", default="part", kind=str)
    if element not in ("part", "face"):
        raise MifOpError("element must be part or face, got '%s'. NOTHING was changed." % element)
    rays = take_int(params, "rays", default=24)
    if not 4 <= rays <= 256:
        raise MifOpError("rays is %d; it must be 4 to 256. NOTHING was changed." % rays)
    space = take(params, "space", default="world", kind=str)
    if space not in ("world", "local"):
        raise MifOpError("space must be world or local, got '%s'. NOTHING was changed." % space)
    distance = take_float(params, "distance")
    if distance is not None and not distance > 0.0:
        raise MifOpError("distance must be more than 0 m, got %g. NOTHING was changed." % distance)
    dspec = params.get("dirt") or {}
    if not isinstance(dspec, dict) or set(dspec) - _DIRT_KEYS:
        raise MifOpError("dirt is {%s}, got %r. NOTHING was changed." % (", ".join(sorted(_DIRT_KEYS)), dspec))
    blur_s = take_float(dspec, "blurStrength", default=1.0)
    blur_n = take_int(dspec, "blurIterations", default=1)
    clean_a = take_float(dspec, "cleanAngle", default=180.0)
    dirt_a = take_float(dspec, "dirtAngle", default=0.0)
    if not 0.01 <= blur_s <= 1.0 or not 0 <= blur_n <= 40 or not 0.0 <= dirt_a <= clean_a <= 180.0:
        raise MifOpError("dirt: blurStrength 0.01-1, blurIterations 0-40, and 0 <= dirtAngle <= cleanAngle <= 180. "
                         "NOTHING was changed.")
    dirt_only = take_bool(dspec, "dirtOnly", default=False)
    normalize = take_bool(dspec, "normalize", default=True)
    group = take(params, "group", kind=str)
    if rule == "fromWeight" and not group:
        raise MifOpError("rule fromWeight needs group: the vertex group to read. NOTHING was changed.")
    adjust = None
    if rule == "adjust":
        if params.get("adjust") is None:
            raise MifOpError("rule adjust needs adjust: {levels, hsv, brightnessContrast}. NOTHING was changed.")
        adjust = _adjust_plan(params["adjust"])
        if "hsv" in adjust and channel != "RGB":
            raise MifOpError("hsv changes a color, so it needs channel RGB (got %s). NOTHING was changed." % channel)
        if params.get("range") is not None or params.get("combine") is not None:
            raise MifOpError("range and combine shape a computed mask; adjust changes the values already there. NOTHING "
                             "was changed.")
    obj = get_object(obj_name, want_mesh=True)
    if rule == "fromWeight" and group not in obj.vertex_groups:
        raise MifOpError("'%s' has no vertex group %r (it has %s). NOTHING was changed."
                         % (obj.name, group, ", ".join(g.name for g in obj.vertex_groups) or "none"))
    mesh = obj.data
    if not mesh.polygons:
        raise MifOpError("'%s' has no faces, so there is nothing to color. NOTHING was changed." % obj.name)

    attr = mesh.color_attributes.get(name)
    if attr is not None:
        if attr.data_type != "BYTE_COLOR" or attr.domain not in ("CORNER", "POINT"):
            raise MifOpError("'%s' is %s on %s; this op writes 8-bit masks, which is what Unreal's vertex color "
                             "stores, into a BYTE_COLOR attribute on face corners or points. Use another name. "
                             "NOTHING was changed." % (name, attr.data_type, attr.domain))
        domain = attr.domain
    else:
        domain = "CORNER"
    if rule == "adjust" and attr is None:
        raise MifOpError("'%s' has no color attribute '%s' to adjust (it has %s). NOTHING was changed."
                         % (obj.name, name, ", ".join(a.name for a in mesh.color_attributes) or "none"))
    if rule == "random" and element == "face" and domain != "CORNER":
        raise MifOpError("a value per face needs a face-corner attribute and '%s' is on points. NOTHING was changed."
                         % name)
    mw = obj.matrix_world if space == "world" else Matrix.Identity(4)
    nv = len(mesh.vertices)
    if rule == "cavity" and distance is None:
        world = [obj.matrix_world @ v.co for v in mesh.vertices]
        distance = 0.05 * max(max(w[i] for w in world) - min(w[i] for w in world) for i in range(3))
        if not distance > 0.0:
            raise MifOpError("'%s' has no size, so a cavity distance cannot be taken from it. Pass 'distance'. NOTHING "
                             "was changed." % obj.name)

    # ---- the value per vertex (or per face) ---------------------------------------------------------------------
    per_face = None
    resolved = None
    per_corner = None
    if rule == "up":
        nm = mw.to_3x3().inverted_safe().transposed()
        # ON FACE CORNERS THE SHADING NORMAL IS READ, which a sharp edge splits: a wall's top face is up to its edge.
        # A vertex normal averages every face round the vertex, so a hard-edged box's top corners read 0.58, not 1.
        if domain == "CORNER":
            per_corner = [(nm @ c).normalized().z for c in _corner_normals(mesh)]
            raw = None
        else:
            raw = [(nm @ v.normal).normalized().z for v in mesh.vertices]
        resolved = rng_in or _RULE_RANGE["up"]
    elif rule == "height":
        raw = [(mw @ v.co).z for v in mesh.vertices]
        resolved = rng_in or (min(raw), max(raw))
        if resolved[0] == resolved[1]:
            raise MifOpError("'%s' is flat - every vertex at z %g - so height has no range. Pass 'range'. NOTHING was "
                             "changed." % (obj.name, resolved[0]))
    elif rule == "cavity":
        raw = _cavity(obj, mesh, distance, rays)
        resolved = rng_in or _RULE_RANGE["cavity"]
    elif rule == "dirt":
        tones = _dirt_tones(mesh, blur_s, blur_n, math.radians(clean_a), math.radians(dirt_a), dirt_only, normalize)
        raw = [1.0 - t for t in tones]
        resolved = rng_in or (0.0, 1.0)
    elif rule == "fromWeight":
        gi = obj.vertex_groups[group].index
        raw = [next((g.weight for g in v.groups if g.group == gi), 0.0) for v in mesh.vertices]
        resolved = rng_in or (0.0, 1.0)
    elif rule == "adjust":
        raw = None
        resolved = (0.0, 1.0)
    else:
        lo_v, hi_v = rng_in or _RULE_RANGE["random"]
        gen = random.Random(seed)
        if element == "face":
            per_face = [lo_v + (hi_v - lo_v) * gen.random() for _ in mesh.polygons]
            raw = None
        else:
            edges = [0] * (2 * len(mesh.edges))
            mesh.edges.foreach_get("vertices", edges)
            comp = _components(nv, edges)
            roots = sorted(set(comp))
            value = {r: lo_v + (hi_v - lo_v) * gen.random() for r in roots}
            raw = [value[c] for c in comp]
        resolved = (lo_v, hi_v)

    def mapped(x):
        if rule == "random":
            v = x
        else:
            a, b = resolved
            v = min(1.0, max(0.0, (x - a) / (b - a)))
        return 1.0 - v if invert else v

    # ---- write ---------------------------------------------------------------------------------------------------
    if attr is None:
        attr = mesh.color_attributes.new(name=name, type="BYTE_COLOR", domain="CORNER")
        if attr is None:
            raise MifOpError("color_attributes.new() returned None for '%s' - Blender signals failure that way. "
                             "NOTHING was changed." % name)
        # A NEW ATTRIBUTE STARTS BLACK WITH ALPHA 1, so the channels this call does not write hold 0
        n = len(attr.data)
        attr.data.foreach_set("color_srgb", [0.0, 0.0, 0.0, 1.0] * n)
        made = True
    else:
        made = False
    n = len(attr.data)
    cur = [0.0] * (4 * n)
    attr.data.foreach_get("color_srgb", cur)
    if rule == "adjust":
        import colorsys                                              # noqa: PLC0415
        chans = _CHANNELS[channel]
        lv, hsv, bc = adjust.get("levels"), adjust.get("hsv"), adjust.get("brightnessContrast")
        ab = _bc_coeffs(*bc) if bc else None
        for i in range(n):
            px = cur[4 * i:4 * i + 4]
            for c in chans:
                if lv:
                    px[c] = lv[1] * (px[c] + lv[0])
            if hsv:
                h, s, v = colorsys.rgb_to_hsv(*[min(1.0, max(0.0, x)) for x in px[:3]])
                h = (h + hsv[0] - 0.5) % 1.0
                px[:3] = list(colorsys.hsv_to_rgb(h, min(1.0, s * hsv[1]), v * hsv[2]))
            for c in chans:
                if ab:
                    px[c] = ab[0] * px[c] + ab[1]
                if invert:
                    px[c] = 1.0 - px[c]
                px[c] = min(1.0, max(0.0, px[c]))
            cur[4 * i:4 * i + 4] = px
        vals = [None] * n
    elif domain == "CORNER":
        loop_v = [0] * len(mesh.loops)
        mesh.loops.foreach_get("vertex_index", loop_v)
        if per_corner is not None:
            vals = [mapped(x) for x in per_corner]
        elif per_face is not None:
            vals = [0.0] * len(mesh.loops)
            for p, fv in zip(mesh.polygons, per_face):
                for li in range(p.loop_start, p.loop_start + p.loop_total):
                    vals[li] = mapped(fv)
        else:
            per_vert = [mapped(x) for x in raw]
            vals = [per_vert[vi] for vi in loop_v]
    else:
        vals = [mapped(x) for x in raw]
    chans = _CHANNELS[channel]
    for i, v in enumerate(vals if rule != "adjust" else ()):
        for c in chans:
            old = cur[4 * i + c]
            if combine == "replace":
                new = v
            elif combine == "multiply":
                new = old * v
            elif combine == "max":
                new = max(old, v)
            elif combine == "min":
                new = min(old, v)
            else:
                new = min(1.0, old + v)
            cur[4 * i + c] = new
    attr.data.foreach_set("color_srgb", cur)
    if make_active:
        mesh.color_attributes.active_color = attr
        try:
            mesh.color_attributes.render_color_index = mesh.color_attributes.find(name)
        except (AttributeError, TypeError):
            pass
    mesh.update()

    # A MASK IS ONLY AS FINE AS THE MESH UNDER IT: values live on vertices and blend across each face, so a band
    # narrower than the faces it crosses becomes a ramp over whole faces. Said, with the numbers, rather than left to a
    # render to show.
    coarse = None
    if rule in ("height", "cavity"):
        ev = [0] * (2 * len(mesh.edges))
        mesh.edges.foreach_get("vertices", ev)
        wco = [obj.matrix_world @ v.co for v in mesh.vertices]
        if rule == "height":
            span = max((abs(wco[ev[i]].z - wco[ev[i + 1]].z) for i in range(0, len(ev), 2)), default=0.0)
            feature = abs(resolved[1] - resolved[0])
            if span > feature:
                coarse = ("edges here span up to %.3f m of height and the range is %.3f m, so the mask ramps across "
                          "whole faces instead of banding - add edge loops (build_kit_piece gridCm) or widen the "
                          "range." % (span, feature))
        else:
            span = max(((wco[ev[i]] - wco[ev[i + 1]]).length for i in range(0, len(ev), 2)), default=0.0)
            if span > 8.0 * distance:
                coarse = ("edges here run up to %.3f m, more than 8 times the %.3f m cavity distance, so a crevice's "
                          "value spreads along whole faces - add edge loops (build_kit_piece gridCm) or raise distance."
                          % (span, distance))

    # READ BACK FROM THE ATTRIBUTE, not from what was computed: the store is 8-bit
    back = [0.0] * (4 * n)
    attr.data.foreach_get("color_srgb", back)
    stats = {}
    for c in chans:
        col = back[c::4]
        stats["RGBA"[c]] = {"min": round(min(col), 4), "max": round(max(col), 4),
                            "mean": round(sum(col) / len(col), 4),
                            "atLeastHalf": round(sum(1 for x in col if x >= 0.5) / len(col), 4)}
    out = {
        "ok": True, "object": obj.name, "rule": rule, "attribute": name, "made": made, "domain": domain,
        "dataType": "BYTE_COLOR", "channel": channel, "combine": combine, "invert": invert,
        "range": [round(resolved[0], 6), round(resolved[1], 6)],
        "written": len(vals),
        "readBack": stats,
        "resolutionNote": coarse,
        "stored": "each byte is value x 255 (sRGB space), what export_mesh's FBX carries and Unreal's VertexColor node "
                  "reads; Blender's shading and mesh_stats read the same bytes linearized",
        "elapsedSeconds": round(time.perf_counter() - started, 2),
    }
    if rule == "cavity":
        out.update({"distanceM": round(distance, 6), "rays": rays})
    if rule == "random":
        out.update({"seed": seed, "element": element,
                    "elements": len(per_face) if per_face is not None else len(set(comp))})
    if rule in ("up", "height"):
        out["space"] = space
    if rule == "dirt":
        out["dirt"] = {"blurStrength": blur_s, "blurIterations": blur_n, "cleanAngle": clean_a, "dirtAngle": dirt_a,
                       "dirtOnly": dirt_only, "normalize": normalize,
                       "note": "1 in creases, 0 on open or worn surface - invert gives Blender's own Dirty Vertex Colors "
                               "tone"}
    if rule == "fromWeight":
        out["group"] = group
    if rule == "adjust":
        out["adjust"] = {k: list(v) for k, v in adjust.items()}
        out.pop("range", None)
        out.pop("combine", None)
    return out


# --------------------------------------------------------------------------------------------------- build_lod_chain
# THE DEFAULT RATIOS are the triangle fractions engine-reduced LODs keep at their medians: 0.8, 0.7, 0.6 and 0.45 of
# LOD0 for LOD1 to LOD4.
_DEFAULT_RATIOS = (0.8, 0.7, 0.6, 0.45)
_MAX_LODS = 8               # MAX_STATIC_MESH_LODS; LOD0 is the object itself


_LOD_SHARP_DEG = 30.0


def _resharpen(mesh, degrees=_LOD_SHARP_DEG):
    """Mark sharp every edge where two faces meet at more than `degrees`, and return how many are. A collapse merges
    a flat face into long triangles that meet the rounding along NEW edges, which carry no sharp mark, so a smooth-
    shaded LOD smeared its flat faces into gradients (measured on a kit wall's 6% LOD, 2026-09-29). Up to 4.0 the marks
    shade only with auto smooth on; from 4.1 they always do."""
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        limit = math.radians(degrees)
        sharp = 0
        for e in bm.edges:
            if len(e.link_faces) == 2 and e.calc_face_angle(0.0) > limit:
                e.smooth = False
                sharp += 1
        bm.to_mesh(mesh)
    finally:
        bm.free()
    if hasattr(mesh, "use_auto_smooth"):
        mesh.use_auto_smooth = True
        mesh.auto_smooth_angle = math.pi
    mesh.update()
    return sharp


def _tris(mesh):
    return sum(len(p.vertices) - 2 for p in mesh.polygons)


def _skin_armature(obj):
    """The armature that deforms `obj` - its Armature modifier's object, else an armature parent - or None."""
    for md in obj.modifiers:
        if md.type == "ARMATURE" and md.object is not None:
            return md.object
    if obj.parent is not None and obj.parent.type == "ARMATURE":
        return obj.parent
    return None


def _basis_mesh(obj):
    """A scratch mesh of `obj`'s rest shape: the basis key's positions, no shape keys, no modifiers."""
    me = obj.data.copy()
    if me.shape_keys is not None:
        basis = me.shape_keys.reference_key
        co = [0.0] * (len(me.vertices) * 3)
        basis.data.foreach_get("co", co)
        tmp = bpy.data.objects.new("MifLODBasis", me)
        tmp.shape_key_clear()
        remove_objects([tmp])
        me.vertices.foreach_set("co", co)
    me.update()
    return me


def _bary_rows(src_me, lod_me):
    """For each LOD vertex: (three source vertex indices, barycentric weights) of the nearest point on the source."""
    src_me.calc_loop_triangles()
    verts = [v.co.copy() for v in src_me.vertices]
    tris = [tuple(t.vertices) for t in src_me.loop_triangles]
    bvh = BVHTree.FromPolygons(verts, tris, all_triangles=True)
    rows = []
    for v in lod_me.vertices:
        loc, _n, ti, _d = bvh.find_nearest(v.co)
        a, b, c = tris[ti]
        pa, pb, pc = verts[a], verts[b], verts[c]
        v0, v1, v2 = pb - pa, pc - pa, loc - pa
        d00, d01, d11 = v0.dot(v0), v0.dot(v1), v1.dot(v1)
        d20, d21 = v2.dot(v0), v2.dot(v1)
        den = d00 * d11 - d01 * d01
        if abs(den) < 1e-20:
            w = (1.0, 0.0, 0.0)
        else:
            wb = (d11 * d20 - d01 * d21) / den
            wc = (d00 * d21 - d01 * d20) / den
            w = (1.0 - wb - wc, wb, wc)
        rows.append(((a, b, c), w))
    return rows


def _skinned_lod_chain(obj, arm, ratios, names, path, started):
    """build_lod_chain for a skinned mesh: each LOD decimated from the REST shape, then given the armature, every
    vertex group's weights and every shape key by barycentric interpolation from the nearest point on LOD0's rest
    surface, parented and deformed as LOD0 is. Unreal imports a skeletal mesh's LODs one file each, so with `file`
    each LOD is written to <file>_LOD<n>.fbx with the armature."""
    coll = (list(obj.users_collection) or [bpy.context.scene.collection])[0]
    src = _basis_mesh(obj)
    keys = obj.data.shape_keys.key_blocks if obj.data.shape_keys is not None else []
    key_rows = [(k.name, [d.co.copy() for d in k.data], k.slider_min, k.slider_max, k.relative_key.name, k.vertex_group)
                for k in keys]
    key_state = {k.name: (k.value, k.mute) for k in keys}
    weights = [dict() for _ in src.vertices]
    for v in obj.data.vertices:
        for g in v.groups:
            weights[v.index][g.group] = g.weight
    arm_mod = next((m for m in obj.modifiers if m.type == "ARMATURE"), None)
    base_tris = _tris(src)
    lods, made_meshes = [], []
    dg = bpy.context.evaluated_depsgraph_get()
    try:
        for n, r in zip(names, ratios):
            tmp = bpy.data.objects.new("MifLODTmp", src)
            coll.objects.link(tmp)
            try:
                mod = tmp.modifiers.new("MifLOD", "DECIMATE")
                mod.decimate_type = "COLLAPSE"
                mod.ratio = r
                dg.update()
                mesh = bpy.data.meshes.new_from_object(tmp.evaluated_get(dg))
            finally:
                remove_objects([tmp])
            made_meshes.append(mesh)
            mesh.name = n
            _resharpen(mesh)
            rows = _bary_rows(src, mesh)
            lod = bpy.data.objects.new(n, mesh)
            coll.objects.link(lod)
            lods.append(lod)
            # Mesh-level group names (3.0+) can come through the decimate already: one group per name, emptied, then
            # written from the interpolation below.
            for vg in obj.vertex_groups:
                if lod.vertex_groups.get(vg.name) is None:
                    lod.vertex_groups.new(name=vg.name)
            every = list(range(len(mesh.vertices)))
            for vg in lod.vertex_groups:
                vg.remove(every)
            by_index = {vg.index: lod.vertex_groups[vg.name] for vg in obj.vertex_groups}
            for i, ((a, b, c), w) in enumerate(rows):
                acc = {}
                for vi, wi in zip((a, b, c), w):
                    for g, gw in weights[vi].items():
                        acc[g] = acc.get(g, 0.0) + gw * wi
                for g, gw in acc.items():
                    if gw > 1e-6:
                        by_index[g].add([i], min(1.0, gw), "REPLACE")
            if key_rows:
                for kname, cos, smin, smax, rel, kvg in key_rows:
                    kb = lod.shape_key_add(name=kname, from_mix=False)
                    kb.slider_min, kb.slider_max = smin, smax
                    kb.value, kb.mute = key_state[kname]     # a new key comes at value 1 - LOD0's value is kept
                    if kvg:
                        kb.vertex_group = kvg
                    if kname == key_rows[0][0]:
                        continue
                    basis = key_rows[0][1]
                    for i, ((a, b, c), w) in enumerate(rows):
                        delta = sum(((cos[vi] - basis[vi]) * wi for vi, wi in zip((a, b, c), w)), Vector())
                        kb.data[i].co = mesh.vertices[i].co + delta
                for kname, _c, _s, _m, rel, _g in key_rows[1:]:
                    lod.data.shape_keys.key_blocks[kname].relative_key = lod.data.shape_keys.key_blocks.get(rel) or \
                        lod.data.shape_keys.reference_key
            lod.parent = obj.parent
            lod.parent_type = obj.parent_type
            lod.matrix_parent_inverse = obj.matrix_parent_inverse.copy()
            lod.matrix_basis = obj.matrix_basis.copy()
            if arm_mod is not None:
                m2 = lod.modifiers.new(arm_mod.name, "ARMATURE")
                for attr in ("object", "use_vertex_groups", "use_bone_envelopes", "use_deform_preserve_volume",
                             "use_multi_modifier"):
                    setattr(m2, attr, getattr(arm_mod, attr))
    except Exception:
        for o in lods:
            try:
                remove_objects([o])
            except ReferenceError:
                pass
        bpy.data.meshes.remove(src)
        raise
    # HOW WELL EACH LOD FOLLOWS THE SKIN, measured in the current pose: every evaluated (posed) LOD vertex's distance
    # from LOD0's evaluated surface.
    dg.update()
    ev0 = obj.evaluated_get(dg)
    me0 = ev0.to_mesh()
    me0.calc_loop_triangles()
    w0 = ev0.matrix_world
    bvh0 = BVHTree.FromPolygons([w0 @ v.co for v in me0.vertices], [tuple(t.vertices) for t in me0.loop_triangles],
                                all_triangles=True)
    ev0.to_mesh_clear()
    rows_out = [{"lod": 0, "object": obj.name, "ratioAsked": 1.0, "triangles": base_tris, "ratioAchieved": 1.0,
                 "vertexGroups": len(obj.vertex_groups), "shapeKeys": [k[0] for k in key_rows],
                 "armature": arm.name}]
    for i, (lod, r) in enumerate(zip(lods, ratios)):
        ev = lod.evaluated_get(dg)
        me = ev.to_mesh()
        dev = max((bvh0.find_nearest(ev.matrix_world @ v.co)[3] or 0.0 for v in me.vertices), default=0.0)
        ev.to_mesh_clear()
        unweighted = sum(1 for v in lod.data.vertices if not v.groups) if obj.vertex_groups else 0
        rows_out.append({"lod": i + 1, "object": lod.name, "ratioAsked": r, "triangles": _tris(lod.data),
                         "ratioAchieved": round(_tris(lod.data) / float(max(1, base_tris)), 4),
                         "vertexGroups": len(lod.vertex_groups), "unweightedVertices": unweighted,
                         "shapeKeys": [k.name for k in lod.data.shape_keys.key_blocks] if lod.data.shape_keys else [],
                         "armature": arm.name, "poseDeviation": round(dev, 6),
                         "uvLayers": [u.name for u in lod.data.uv_layers]})
    bpy.data.meshes.remove(src)
    out = {"ok": True, "object": obj.name, "skinned": True, "armature": arm.name, "group": None, "lods": rows_out,
           "order": "LOD0 then %s" % ", ".join(n for n in names),
           "poseNote": "poseDeviation is each LOD's largest distance, in world units, from LOD0's surface in the pose "
                       "the armature holds now; weights and shape keys came by barycentric interpolation from the "
                       "nearest point on LOD0's rest surface."}
    if path is not None:
        files = []
        stem = path[:-4]
        sel = selection_snapshot()
        try:
            for k, o in enumerate([obj] + lods):
                fp = "%s_LOD%d.fbx" % (stem, k)
                select_only([arm, o])
                args = dict(FBX_EXPORT_ARGS)
                args["object_types"] = {"MESH", "ARMATURE"}
                bpy.ops.export_scene.fbx(filepath=fp, use_selection=True, **args)
                files.append({"lod": k, "path": fp, "bytes": os.path.getsize(fp) if os.path.exists(fp) else None})
        finally:
            selection_restore(sel)
        if any(not f["bytes"] for f in files):
            raise MifOpError("the FBX exporter returned and %s is missing or empty. The chain WAS built in the scene."
                             % [f["path"] for f in files if not f["bytes"]])
        out["files"] = files
        out["filesNote"] = ("Unreal imports a skeletal mesh's LODs one file each (Skeletal Mesh editor: LOD Import), so "
                            "each LOD went to its own file with the armature.")
    out["elapsedSeconds"] = round(time.perf_counter() - started, 2)
    return out


def op_build_lod_chain(params):
    """Make a chain of LODs for one mesh that Unreal's FBX importer takes as the static mesh's LODs: decimated copies
    <object>_LOD1, <object>_LOD2, ... and the object itself as LOD0, all under an Empty <object>_LODGroup whose
    fbx_type is LodGroup. With 'file', the group, its LODs and the object's UCX_/UBX_ hulls are written to that FBX,
    ready for import with Import Mesh LODs on. Each LOD's edges sharper than 30 degrees are marked sharp, so its flat
    faces still shade flat after the collapse.

    params:
      object (str)          required, a mesh in OBJECT mode with no parent
      ratios (list)         the triangle fraction of LOD0 each LOD keeps, falling, each above 0 and below 1; default
                            [0.8, 0.7, 0.6, 0.45]; at most 7
      file (str)            write the chain to this .fbx (centimeters, Z up, the export_mesh settings)
      overwrite (bool)      replace an existing file, default true
    """
    reject_unknown(params, LOD_CHAIN_PARAMS, "build_lod_chain", _LOD_NOTES)
    started = time.perf_counter()
    # EVERY VALUE IS CHECKED BEFORE THE OBJECT IS LOOKED UP, and everything about the object before anything is made
    obj_name = take(params, "object", required=True, kind=str)
    raw = params.get("ratios")
    if raw is None:
        ratios = list(_DEFAULT_RATIOS)
    else:
        if not isinstance(raw, (list, tuple)) or not raw:
            raise MifOpError("'ratios' takes a list of fractions, e.g. [0.5, 0.25], got %r. NOTHING was created."
                             % (raw,))
        if any(isinstance(r, bool) or not isinstance(r, (int, float)) or not math.isfinite(r) for r in raw):
            raise MifOpError("'ratios' must be numbers, got %r. NOTHING was created." % (raw,))
        ratios = finite_floats(raw, "ratios")
    if len(ratios) > _MAX_LODS - 1:
        raise MifOpError("%d ratios make %d LODs; Unreal keeps at most %d, LOD0 included. NOTHING was created."
                         % (len(ratios), len(ratios) + 1, _MAX_LODS))
    if any(not 0.0 < r < 1.0 for r in ratios):
        raise MifOpError("every ratio must be above 0 and below 1 (a fraction of LOD0's triangles), got %s. NOTHING "
                         "was created." % ratios)
    if any(b >= a for a, b in zip(ratios, ratios[1:])):
        raise MifOpError("ratios must fall from one LOD to the next, got %s. NOTHING was created." % ratios)
    overwrite = take_bool(params, "overwrite", default=True)
    raw_path = take(params, "file", kind=str)
    if raw_path is not None and not raw_path.lower().endswith(".fbx"):
        raise MifOpError("'file' must end in .fbx - a LOD group is an FBX node type. NOTHING was created.")
    obj = get_object(obj_name, want_mesh=True)
    # A SKINNED MESH (2026-10-06): an Armature modifier or an armature parent makes a skinned chain - the armature,
    # every vertex group's weights and every shape key kept on each LOD (_skinned_lod_chain).
    arm = _skin_armature(obj)
    if arm is not None and obj.parent is not None and obj.parent is not arm:
        raise MifOpError("'%s' is skinned to '%s' but parented to '%s'; a skinned chain keeps LOD0's parent, which must "
                         "be its armature or nothing. NOTHING was created." % (obj.name, arm.name, obj.parent.name))
    if arm is None and obj.parent is not None:
        raise MifOpError("'%s' is parented to '%s'; the chain parents it under its LOD group, which would replace "
                         "that. Clear the parent first. NOTHING was created." % (obj.name, obj.parent.name))
    if obj.children:
        # A LOD NODE'S WHOLE SUBTREE IS THAT LOD in Unreal (FindAllLODGroupNode collects every mesh under it), so a
        # child would be merged into LOD0
        raise MifOpError("'%s' has children (%s); under a LOD group they would be merged into LOD0 on import. Unparent "
                         "them first. NOTHING was created." % (obj.name, ", ".join(c.name for c in obj.children)))
    base = obj.name
    group_name = base + "_LODGroup"
    names = ["%s_LOD%d" % (base, i + 1) for i in range(len(ratios))]
    present = {o.name for o in bpy.data.objects}
    taken = [n for n in ([] if arm is not None else [group_name]) + names if n in present]
    if taken:
        raise MifOpError("%s already exist%s; Blender would suffix the new one .001. Remove %s first. NOTHING was "
                         "created." % (", ".join(taken), "s" if len(taken) == 1 else "", "it" if len(taken) == 1
                                       else "them"))
    if len(names[-1]) > 63 or len(group_name) > 63:
        raise MifOpError("'%s' is too long for its LOD names to fit Blender's 63 characters. NOTHING was created."
                         % base)
    path = None
    if raw_path is not None:
        path = _resolve_out_path(raw_path)
        check_output_path(raw_path, path, "exported")
        before_file = file_before(path)
        if before_file and not overwrite and arm is None:
            raise MifOpError("%s exists and overwrite is false. NOTHING was created." % path)
    else:
        before_file = None
    # THE TRIANGLES THAT REACH UNREAL are the evaluated ones - LOD0 exports through its modifiers, and every LOD is
    # reduced from the same evaluated mesh - so ratios are measured against those
    dg = bpy.context.evaluated_depsgraph_get()
    if obj.modifiers:
        evaluated = bpy.data.meshes.new_from_object(obj.evaluated_get(dg))
        try:
            base_tris = _tris(evaluated)
        finally:
            bpy.data.meshes.remove(evaluated)
    else:
        base_tris = _tris(obj.data)
    if base_tris < 8:
        raise MifOpError("'%s' has %d triangles; there is nothing to reduce. NOTHING was created." % (base, base_tris))

    if arm is not None:
        if path is not None:
            for k in range(len(names) + 1):
                fp = "%s_LOD%d.fbx" % (path[:-4], k)
                check_output_path(fp, fp, "exported")
                if os.path.exists(fp) and not overwrite:
                    raise MifOpError("%s exists and overwrite is false - a skinned chain writes one file per LOD. "
                                     "NOTHING was created." % fp)
        return _skinned_lod_chain(obj, arm, ratios, names, path, started)
    colls = list(obj.users_collection) or [bpy.context.scene.collection]
    coll = colls[0]
    lods = []
    made = []
    try:
        for n, r in zip(names, ratios):
            tmp = obj.copy()
            coll.objects.link(tmp)
            made.append(tmp)
            mod = tmp.modifiers.new("MifLOD", "DECIMATE")
            mod.decimate_type = "COLLAPSE"
            mod.ratio = r
            dg.update()
            mesh = bpy.data.meshes.new_from_object(tmp.evaluated_get(dg))
            remove_objects([tmp])
            made.remove(tmp)
            mesh.name = n
            _resharpen(mesh)
            lod = bpy.data.objects.new(n, mesh)
            coll.objects.link(lod)
            made.append(lod)
            lods.append(lod)
        group = bpy.data.objects.new(group_name, None)
        group["fbx_type"] = "LodGroup"
        group.empty_display_type = "PLAIN_AXES"
        coll.objects.link(group)
        made.append(group)
    except Exception:
        # WHAT THIS CALL MADE IS TAKEN BACK before the error goes on: the temporary copy, the LODs, the group
        for o in made:
            try:
                remove_objects([o])
            except ReferenceError:
                pass
        raise
    # THE GROUP TAKES THE OBJECT'S PLACE: it sits where the object sat, and every LOD rides on it with no offset of
    # its own - so every LOD sits exactly on LOD0
    world = obj.matrix_world.copy()
    group.matrix_world = world
    for child in [obj] + lods:
        child.parent = group
        child.matrix_parent_inverse = Matrix.Identity(4)
        child.matrix_basis = Matrix.Identity(4)
    # the object keeps its own modifiers; the LODs were made from its evaluated result
    rows = [{"lod": 0, "object": obj.name, "ratioAsked": 1.0, "triangles": base_tris, "ratioAchieved": 1.0,
             "uvLayers": [u.name for u in obj.data.uv_layers],
             "colorAttributes": [c.name for c in obj.data.color_attributes]}]
    for i, (lod, r) in enumerate(zip(lods, ratios)):
        t = _tris(lod.data)
        rows.append({"lod": i + 1, "object": lod.name, "ratioAsked": r, "triangles": t,
                     "ratioAchieved": round(t / float(max(1, base_tris)), 4),
                     "uvLayers": [u.name for u in lod.data.uv_layers],
                     "colorAttributes": [c.name for c in lod.data.color_attributes],
                     "sharpEdges": sum(1 for e in lod.data.edges if e.use_edge_sharp)})
    notes = []
    if obj.modifiers:
        notes.append("'%s' has %d modifier(s): LOD0 exports through them and the LODs were reduced from the same "
                     "evaluated result, so LOD0 reaches Unreal with %d triangles, not the %d of its mesh data."
                     % (obj.name, len(obj.modifiers), base_tris, _tris(obj.data)))
    hulls = sorted(o.name for o in bpy.data.objects if o.type == "MESH"
                   and any(o.name.startswith(p + base + "_") for p in ("UCX_", "UBX_", "USP_", "UCP_")))
    out = {"ok": True, "object": obj.name, "group": group.name, "fbxType": group.get("fbx_type"),
           "lods": rows, "collisionHulls": hulls,
           "order": "Unreal takes the LodGroup's children in order: %s" % ", ".join(r["object"] for r in rows),
           "elapsedSeconds": None}
    if path is not None:
        sel = selection_snapshot()
        try:
            if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
                bpy.ops.object.mode_set(mode="OBJECT")
            export_set = [group, obj] + lods + [bpy.data.objects[h] for h in hulls]
            select_only(export_set)
            args = dict(FBX_EXPORT_ARGS)
            args["object_types"] = {"MESH", "EMPTY"}
            bpy.ops.export_scene.fbx(filepath=path, use_selection=True, **args)
        finally:
            selection_restore(sel)
        out["file"] = dict({"path": path, "bytes": os.path.getsize(path) if os.path.exists(path) else None,
                            "objects": [o.name for o in export_set]}, **replaced_fields(before_file))
        if not out["file"]["bytes"]:
            raise MifOpError("the FBX exporter returned and %s is missing or empty. The chain WAS built in the scene."
                             % path)
    out["notes"] = notes or None
    out["elapsedSeconds"] = round(time.perf_counter() - started, 2)
    return out


# =============================================================================================================
# LATHE AND TUBE (2026-10-06, spec "Prop builders learned from the alley")
# =============================================================================================================
# Both build vertex, face and UV lists and hand them to create_mesh, so naming, collections and the read-back are
# create_mesh's. Each part is a closed solid when its ends are closed, and its winding is checked by signed volume.

_LATHE_KEYS = ("name", "profile", "segments", "axis", "caps", "smooth", "uvScale", "location", "rotation", "scale",
               "collection", "takeOverName", "material")
_TUBE_KEYS = ("name", "points", "start", "end", "sag", "samples", "radius", "sides", "closed", "caps", "smooth",
              "uvScale", "location", "rotation", "scale", "collection", "takeOverName", "material", "surface", "lift",
              "step", "hangEnds", "endCaps", "stoppers")
_AXES = {"Z": ((1, 0, 0), (0, 1, 0), (0, 0, 1)), "Y": ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
         "X": ((0, 1, 0), (0, 0, 1), (1, 0, 0))}


def _signed_volume(verts, faces):
    vol = 0.0
    for f in faces:
        a = Vector(verts[f[0]])
        for k in range(1, len(f) - 1):
            vol += a.dot(Vector(verts[f[k]]).cross(Vector(verts[f[k + 1]])))
    return vol / 6.0


def _build_through_create_mesh(params, verts, faces, uvs, smooth, default_name, op):
    from .ops_create import op_create_mesh   # noqa: PLC0415 - create_mesh owns naming and the read-back
    sub = {"name": params.get("name") or default_name, "vertices": verts, "faces": faces, "uvs": {"UVMap": uvs},
           "smooth": bool(smooth)}
    for k in ("location", "rotation", "scale", "collection", "takeOverName"):
        if params.get(k) is not None:
            sub[k] = params[k]
    if params.get("material") is not None:
        sub["materials"] = [params["material"]]
        sub["materialIndices"] = [0] * len(faces)
    out = op_create_mesh(sub)
    out["op"] = op
    return out


def op_build_lathe(params):
    """Revolve a profile [[r, h], ...] round an axis into a closed or open solid of revolution - a can, a bottle, a post.

    THE GAP (2026-10-04): a trash can's profile was revolved in local Python, and that revolve turned each face AWAY
    FROM THE PROFILE'S CENTROID: on a tapered can the lower wall's radius was under the centroid's, so 800 of 3,880
    outer faces (20 %) faced inward and Unreal drew that band see-through. Here every face winds by the profile
    edge's own direction - one consistent orientation over the whole surface - and the finished part is then turned
    so it faces out: by its signed volume when it is closed, by the area-weighted radial direction when it is open.
    `outwardCheck` reports both.

    profile: two or more [r, h] points, r (distance from the axis) at least 0, h along it, in order. A point at r 0 is
    a pole (one vertex); an end with r above 0 is closed by a cap face when caps is true (default). segments (default
    32, 3-512) round the axis; axis Z (default), Y or X. smooth (default true) shades smooth. uvScale (default 1) is
    metres per UV unit: u round the circumference at the largest radius, v along the profile. material names an
    existing material for every face. name, location, rotation, scale, collection, takeOverName are create_mesh's.

    Refused before anything exists: fewer than two points, a negative r, two consecutive points on one spot, every
    point on the axis, segments outside 3-512.
    """
    tail = "NOTHING was created."
    reject_unknown(params, _LATHE_KEYS, "build_lathe")
    raw = params.get("profile")
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        raise MifOpError("profile must be two or more [r, h] points. %s" % tail)
    prof = []
    for p in raw:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            raise MifOpError("each profile point must be [r, h], got %r. %s" % (p, tail))
        prof.append(tuple(finite_floats(list(p), "profile")))
    if any(r < 0 for r, _ in prof):
        raise MifOpError("a profile radius is below 0: r is the distance from the axis. %s" % tail)
    for a, b in zip(prof, prof[1:]):
        if math.dist(a, b) < 1e-9:
            raise MifOpError("profile repeats the point %r - a zero-length edge. %s" % (a, tail))
    if all(r < 1e-9 for r, _ in prof):
        raise MifOpError("every profile point is on the axis - nothing to revolve. %s" % tail)
    segs = take_int(params, "segments", default=32)
    if not 3 <= segs <= 512:
        raise MifOpError("segments must be 3-512, got %d. %s" % (segs, tail))
    axis = str(take(params, "axis", default="Z", kind=str)).upper()
    if axis not in _AXES:
        raise MifOpError("axis must be X, Y or Z, got %r. %s" % (axis, tail))
    caps = take_bool(params, "caps", default=True)
    smooth = take_bool(params, "smooth", default=True)
    uv_scale = take_float(params, "uvScale", default=1.0)
    if not uv_scale > 0:
        raise MifOpError("uvScale must be above 0, got %g. %s" % (uv_scale, tail))
    if params.get("material") is not None and params["material"] not in bpy.data.materials:
        raise MifOpError("no material named %r. %s" % (params["material"], tail))
    ex, ey, ez = (Vector(v) for v in _AXES[axis])
    verts, rows = [], []
    for r, h in prof:
        if r < 1e-9:
            rows.append([len(verts)])
            verts.append(list(ez * h))
            continue
        row = []
        for k in range(segs):
            t = 2 * math.pi * k / segs
            row.append(len(verts))
            verts.append(list(ex * (r * math.cos(t)) + ey * (r * math.sin(t)) + ez * h))
        rows.append(row)
    circ = 2 * math.pi * max(r for r, _ in prof)
    along = [0.0]
    for a, b in zip(prof, prof[1:]):
        along.append(along[-1] + math.dist(a, b))
    faces, uvs = [], []
    for j in range(len(rows) - 1):
        a, b = rows[j], rows[j + 1]
        v0, v1 = along[j] / uv_scale, along[j + 1] / uv_scale
        for k in range(segs):
            k1 = (k + 1) % segs
            u0, u1 = circ * k / segs / uv_scale, circ * (k + 1) / segs / uv_scale
            if len(a) == 1 and len(b) == 1:
                continue
            if len(a) == 1:
                faces.append([a[0], b[k1], b[k]])     # ring b walked k1 -> k, as the band above walks it
                uvs.append([[(u0 + u1) / 2, v0], [u1, v1], [u0, v1]])
            elif len(b) == 1:
                faces.append([a[k], a[k1], b[0]])
                uvs.append([[u0, v0], [u1, v0], [(u0 + u1) / 2, v1]])
            else:
                faces.append([a[k], a[k1], b[k1], b[k]])
                uvs.append([[u0, v0], [u1, v0], [u1, v1], [u0, v1]])
    capped = []
    if caps:
        for end, row in (("start", rows[0]), ("end", rows[-1])):
            if len(row) > 1:
                _cap_fan(verts, faces, uvs, list(row), end != "end")
                capped.append(end)
    closed = all(len(r) == 1 or caps for r in (rows[0], rows[-1]))
    if not faces:
        raise MifOpError("the profile makes no face. %s" % tail)
    vol = _signed_volume(verts, faces)
    flipped = False
    if closed:
        flipped = vol < 0
    else:
        radial = 0.0
        for f in faces:
            c = sum((Vector(verts[i]) for i in f), Vector()) / len(f)
            n = (Vector(verts[f[1]]) - Vector(verts[f[0]])).cross(Vector(verts[f[-1]]) - Vector(verts[f[0]]))
            out_dir = c - ez * c.dot(ez)
            radial += n.dot(out_dir.normalized()) if out_dir.length > 1e-9 else 0.0
        flipped = radial < 0
    if flipped:
        faces = [list(reversed(f)) for f in faces]
        uvs = [list(reversed(u)) for u in uvs]
        vol = -vol
    out = _build_through_create_mesh(params, verts, faces, uvs, smooth, "Lathe", "build_lathe")
    out.update({"segments": segs, "axis": axis, "capped": capped, "closed": closed,
                "outwardCheck": {"signedVolume": round(vol, 9), "closed": closed,
                                 "turnedOver": flipped,
                                 "note": ("one winding over the whole surface, then turned so the %s is positive"
                                          % ("signed volume" if closed else "area-weighted radial direction"))}})
    return out


def _catenary(a, b, sag, n):
    """n + 1 points from a to b hanging `sag` metres below the chord at its middle (a parabola through the three is
    the catenary's shape at these sags; the exact catenary is solved for the same mid drop)."""
    a, b = Vector(a), Vector(b)
    pts = []
    if sag <= 0:
        return [a.lerp(b, k / n) for k in range(n + 1)]
    span = (b - a).xy.length if (b - a).xy.length > 1e-9 else (b - a).length
    # cosh form y = c*cosh(x/c) - c with c from the sag over half the span (bisection on c)
    half = span / 2.0
    lo_c, hi_c = 1e-6, 1e9
    for _ in range(200):
        mid = math.sqrt(lo_c * hi_c)
        d = mid * (math.cosh(min(half / mid, 700)) - 1.0)
        if d > sag:
            lo_c = mid
        else:
            hi_c = mid
    c = math.sqrt(lo_c * hi_c)
    for k in range(n + 1):
        t = k / n
        x = (t - 0.5) * span
        drop = sag - c * (math.cosh(x / c) - 1.0)
        p = a.lerp(b, t)
        p.z -= drop
        pts.append(p)
    return pts


def _cap_fan(verts, faces, uvs, ring, reverse):
    """Close a round ring with a fan of triangles from a new vertex at its middle, UVs on a disc: one n-gon there fails
    mesh_quality gameReady, and an engine would split it its own way. Same flat cap, and on a smooth part the fan shades
    from the center out where an arbitrary split would streak."""
    n = len(ring)
    mid = len(verts)
    verts.append([sum(verts[i][k] for i in ring) / n for k in range(3)])
    disc = [[0.5 + 0.5 * math.cos(2 * math.pi * k / n), 0.5 + 0.5 * math.sin(2 * math.pi * k / n)] for k in range(n)]
    for k in range(n):
        k1 = (k + 1) % n
        if reverse:
            faces.append([mid, ring[k1], ring[k]])
            uvs.append([[0.5, 0.5], disc[k1], disc[k]])
        else:
            faces.append([mid, ring[k], ring[k1]])
            uvs.append([[0.5, 0.5], disc[k], disc[k1]])


def op_build_tube(params):
    """A round tube along points or hanging between two of them - a cable, a pipe, a neon tube, a downspout's run.

    THE GAP (2026-10-04): neon signs, cables and pipes were swept in local scripts, and a cable between two points was
    hung by hand. points [[x, y, z], ...] (two or more) gives the path; closed true joins the last back to the first
    (a neon loop); or start and end with sag (metres the middle hangs below the straight line, along -Z) give a catenary
    sampled at samples (default 16) segments. The cross-section is a ring of sides (default 12, 3-128) at radius,
    carried along the path by parallel transport, so it does not twist at a bend; each corner ring is widened by the
    bend so the wall keeps its thickness. caps (default true) close an open tube's ends, so it is a closed solid.
    uvScale (default 1): u round the tube, v along it, in metres per UV unit. The part's winding is checked by signed
    volume and turned out if needed (outwardCheck). material, smooth (default true), name, location, rotation, scale,
    collection and takeOverName are as build_lathe's.

    Refused before anything exists: points with start/end, neither, fewer than two points, two consecutive points on one
    spot, radius not above 0, sag below 0, sides or samples out of range, sag with points.
    """
    tail = "NOTHING was created."
    reject_unknown(params, _TUBE_KEYS, "build_tube")
    has_pts = params.get("points") is not None
    has_ends = params.get("start") is not None or params.get("end") is not None
    if has_pts == has_ends:
        raise MifOpError("pass points [[x, y, z], ...] OR start and end (with sag). %s" % tail)
    radius = take_float(params, "radius", default=None)
    if radius is None or not radius > 0:
        raise MifOpError("radius is required and must be above 0, got %r. %s" % (radius, tail))
    sides = take_int(params, "sides", default=12)
    if not 3 <= sides <= 128:
        raise MifOpError("sides must be 3-128, got %d. %s" % (sides, tail))
    closed = take_bool(params, "closed", default=False)
    caps = take_bool(params, "caps", default=True)
    smooth = take_bool(params, "smooth", default=True)
    uv_scale = take_float(params, "uvScale", default=1.0)
    if not uv_scale > 0:
        raise MifOpError("uvScale must be above 0, got %g. %s" % (uv_scale, tail))
    if params.get("material") is not None and params["material"] not in bpy.data.materials:
        raise MifOpError("no material named %r. %s" % (params["material"], tail))
    if has_pts:
        if params.get("sag") is not None or params.get("samples") is not None:
            raise MifOpError("sag and samples hang a cable between start and end; points give the path itself. %s" % tail)
        raw = params["points"]
        if not isinstance(raw, (list, tuple)) or len(raw) < 2 or any(
                not isinstance(p, (list, tuple)) or len(p) != 3 for p in raw):
            raise MifOpError("points must be two or more [x, y, z]. %s" % tail)
        pts = [Vector(finite_floats(list(p), "points")) for p in raw]
        if closed and len(pts) < 3:
            raise MifOpError("a closed tube needs at least three points. %s" % tail)
    else:
        if params.get("start") is None or params.get("end") is None:
            raise MifOpError("a hanging tube needs both start and end. %s" % tail)
        if closed:
            raise MifOpError("closed joins a path of points; a cable from start to end has two ends. %s" % tail)
        a = finite_floats(list(params["start"]), "start")
        b = finite_floats(list(params["end"]), "end")
        sag = take_float(params, "sag", default=0.0)
        n = take_int(params, "samples", default=16)
        if sag < 0:
            raise MifOpError("sag must be 0 or above (metres below the straight line), got %g. %s" % (sag, tail))
        if not 1 <= n <= 1000:
            raise MifOpError("samples must be 1-1000, got %d. %s" % (n, tail))
        pts = _catenary(a, b, sag, n)
    # ON A SURFACE (2026-10-06, spec "Eyelets ... and a cord threaded through them"): a drawcord runs round a hood's
    # channel from one eyelet to the other, lift off the cloth, and hangs past each end. surface names the mesh the
    # path is laid on: points are resampled every step (default: as given), each moved to the nearest point of the
    # evaluated surface and lift along its normal; hangEnds [a, b] adds a straight drop of a and b metres (-Z) at the
    # start and the end, so the two ends can hang unevenly.
    surf_name = take(params, "surface", default=None, kind=str)
    lift = take_float(params, "lift", default=None)
    step = take_float(params, "step", default=None)
    hang = params.get("hangEnds")
    if surf_name is None and (lift is not None or step is not None):
        raise MifOpError("lift and step lay the path on a surface - pass surface too. %s" % tail)
    if surf_name is not None and not has_pts:
        raise MifOpError("surface lays a path of points on a mesh; a hanging cable (start, end, sag) is not laid on "
                         "one. %s" % tail)
    if step is not None and not step > 0:
        raise MifOpError("step must be above 0, got %g. %s" % (step, tail))
    if lift is not None and lift < 0:
        raise MifOpError("lift must be 0 or above, got %g. %s" % (lift, tail))
    if hang is not None:
        if closed or not isinstance(hang, (list, tuple)) or len(hang) != 2:
            raise MifOpError("hangEnds is [start drop, end drop] in metres on an open path. %s" % tail)
        hang = finite_floats(list(hang), "hangEnds")
        if min(hang) < 0:
            raise MifOpError("hangEnds must be 0 or above, got %s. %s" % (hang, tail))
    # CORD ENDS (2026-10-06): endCaps NONE | KNOT | AGLET | STOPPER, one for both ends or [start, end]; stoppers
    # [{at, object, boreRadius, boreAxis}] slide a stopper onto the cord at `at` metres along it - a model already in
    # the file (its origin on its bore's centre, the bore along boreAxis, default Z), scaled so its bore matches the
    # cord's radius, or without object a built-in barrel whose bore is the cord's radius.
    end_caps = _cord_caps(params.get("endCaps"), closed, tail)
    stoppers = _cord_stoppers(params.get("stoppers"), closed, tail)
    surface_gap = channel_len = None
    if surf_name is not None:
        from mathutils.bvhtree import BVHTree   # noqa: PLC0415
        surf = get_object(surf_name, want_mesh=True)
        lift = lift or 0.0
        if step:
            dense = []
            for i in range(len(pts) - (0 if closed else 1)):
                a, b = pts[i], pts[(i + 1) % len(pts)]
                n = max(1, int(math.ceil((b - a).length / step - 1e-9)))
                dense.extend(a.lerp(b, k / n) for k in range(n))
            if not closed:
                dense.append(pts[-1])
            pts = dense
        dg = bpy.context.evaluated_depsgraph_get()
        ev = surf.evaluated_get(dg)
        me = ev.to_mesh()
        me.calc_loop_triangles()
        mw = ev.matrix_world
        wv = [mw @ v.co for v in me.vertices]
        tris = [tuple(t.vertices) for t in me.loop_triangles]
        ev.to_mesh_clear()
        if not tris:
            raise MifOpError("surface '%s' has no faces. %s" % (surf.name, tail))
        bvh = BVHTree.FromPolygons(wv, tris, all_triangles=True)
        laid, gaps = [], []
        for p in pts:
            loc, nrm, _i, _d = bvh.find_nearest(p)
            q = loc + nrm.normalized() * lift
            laid.append(q)
        for q in laid:
            gaps.append(bvh.find_nearest(q)[3])
        pts = laid
        surface_gap = {"min": round(min(gaps), 6), "max": round(max(gaps), 6), "lift": lift}
        channel_len = sum((pts[(i + 1) % len(pts)] - pts[i]).length for i in range(len(pts) - (0 if closed else 1)))
    if hang is not None:
        if channel_len is None:
            channel_len = sum((pts[i + 1] - pts[i]).length for i in range(len(pts) - 1))
        down = Vector((0, 0, -1))
        head = [pts[0] + down * hang[0]] if hang[0] > 0 else []
        tail_pts = [pts[-1] + down * hang[1]] if hang[1] > 0 else []
        pts = head + pts + tail_pts
    for i in range(len(pts) - (0 if closed else 1)):
        if (pts[(i + 1) % len(pts)] - pts[i]).length < 1e-9:
            raise MifOpError("points %d and %d are on one spot. %s" % (i, (i + 1) % len(pts), tail))
    m = len(pts)
    tangents = []
    for i in range(m):
        if closed:
            t = (pts[(i + 1) % m] - pts[i]).normalized() + (pts[i] - pts[i - 1]).normalized()
        elif i == 0:
            t = pts[1] - pts[0]
        elif i == m - 1:
            t = pts[-1] - pts[-2]
        else:
            t = (pts[i + 1] - pts[i]).normalized() + (pts[i] - pts[i - 1]).normalized()
        if t.length < 1e-9:
            t = pts[min(i + 1, m - 1)] - pts[max(i - 1, 0)]
        tangents.append(t.normalized())
    t0 = tangents[0]
    ref = Vector((0, 0, 1)) if abs(t0.z) < 0.9 else Vector((1, 0, 0))
    nrm = (ref - t0 * ref.dot(t0)).normalized()
    frames = []
    for i, t in enumerate(tangents):
        if i:
            prev = tangents[i - 1]
            rot = prev.rotation_difference(t)
            nrm = (rot @ nrm)
            nrm = (nrm - t * nrm.dot(t)).normalized()
        frames.append((nrm.copy(), t.cross(nrm).normalized()))
    verts, rings = [], []
    for i, (p, t, (nx, bx)) in enumerate(zip(pts, tangents, frames)):
        widen = 1.0
        if 0 < i < m - 1 or closed:
            seg = (pts[(i + 1) % m] - p).normalized()
            c = abs(seg.dot(t))
            widen = 1.0 / max(c, 0.2)
        ring = []
        for k in range(sides):
            ang = 2 * math.pi * k / sides
            off = nx * math.cos(ang) + bx * math.sin(ang)
            # widen only across the bend (the component in the bend's plane)
            if widen != 1.0:
                seg = (pts[(i + 1) % m] - p).normalized()
                bend = (seg - t * seg.dot(t))
                if bend.length > 1e-9:
                    bend.normalize()
                    off = off + bend * off.dot(bend) * (widen - 1.0)
            ring.append(len(verts))
            verts.append(list(p + off * radius))
        rings.append(ring)
    along = [0.0]
    for i in range(1, m + (1 if closed else 0)):
        along.append(along[-1] + (pts[i % m] - pts[i - 1]).length)
    circ = 2 * math.pi * radius
    faces, uvs = [], []
    for i in range(m - 1 + (1 if closed else 0)):
        a, b = rings[i], rings[(i + 1) % m]
        v0, v1 = along[i] / uv_scale, along[i + 1] / uv_scale
        for k in range(sides):
            k1 = (k + 1) % sides
            u0, u1 = circ * k / sides / uv_scale, circ * (k + 1) / sides / uv_scale
            faces.append([a[k], a[k1], b[k1], b[k]])
            uvs.append([[u0, v0], [u1, v0], [u1, v1], [u0, v1]])
    if caps and not closed:
        for ring, rev in ((rings[0], True), (rings[-1], False)):
            _cap_fan(verts, faces, uvs, ring, rev)
    solid = closed or caps
    vol = _signed_volume(verts, faces)
    flipped = solid and vol < 0
    if flipped:
        faces = [list(reversed(f)) for f in faces]
        uvs = [list(reversed(u)) for u in uvs]
        vol = -vol
    cap_rows, stop_rows, placed = [], [], []
    if end_caps or stoppers:
        for st in stoppers:
            if st["at"] > along[-1] + 1e-9:
                raise MifOpError("a stopper at %g m is past the cord's end (%g m long). %s" % (st["at"], along[-1], tail))
        for which, kind in end_caps:
            pv, pf, row = _cord_cap(kind, which, pts, along, radius, sides)
            base = len(verts)
            verts += pv
            faces += [[i + base for i in f] for f in pf]
            uvs += _face_uvs(pv, pf)
            cap_rows.append(row)
        for k, st in enumerate(stoppers):
            pos, tan = _along_path(pts, along, st["at"])
            row = {"at": st["at"], "location": rnd(list(pos)), "tangent": rnd(list(tan))}
            if st["object"] is None:
                length, outer = 8 * radius, 3 * radius
                pv, pf = _revolve([(radius, 0), (outer * 0.8, 0), (outer, length * 0.2), (outer, length * 0.8),
                                   (outer * 0.8, length), (radius, length)], pos - tan * (length / 2), tan, sides,
                                  loop=True)
                base = len(verts)
                verts += pv
                faces += [[i + base for i in f] for f in pf]
                uvs += _face_uvs(pv, pf)
                row.update({"object": None, "boreRadius": radius, "length": length})
            else:
                placed.append((k, st, pos, tan))
            stop_rows.append(row)
    out = _build_through_create_mesh(params, verts, faces, uvs, smooth, "Tube", "build_tube")
    if placed:
        tube = bpy.data.objects[out["name"]]
        for k, st, pos, tan in placed:
            model = st["object"]
            dup = model.copy()
            dup.name = "%s_Stopper%d" % (tube.name, k + 1)
            for col in tube.users_collection:
                col.objects.link(dup)
            scale = radius / st["boreRadius"]
            rot = Vector(_AXIS_VEC[st["boreAxis"]]).rotation_difference(tan).to_matrix().to_4x4()
            dup.parent = tube
            dup.matrix_parent_inverse = Matrix.Identity(4)
            dup.matrix_basis = Matrix.Translation(pos) @ rot @ Matrix.Diagonal((scale, scale, scale, 1.0))
            stop_rows[k].update({"object": dup.name, "model": model.name, "boreRadius": st["boreRadius"],
                                 "scale": round(scale, 6), "boreAxis": st["boreAxis"]})
    out.update({"pathPoints": m, "pathLength": round(along[-1], 6), "closedLoop": closed,
                "capped": bool(caps and not closed), "radius": radius, "sides": sides,
                "lowestPoint": rnd(list(min(pts, key=lambda q: q.z))),
                "surface": surf_name, "surfaceGap": surface_gap,
                "channelLength": round(channel_len, 6) if channel_len is not None else None,
                "hangEnds": hang, "endCaps": cap_rows or None, "stoppers": stop_rows or None,
                "outwardCheck": {"signedVolume": round(vol, 9), "closed": solid, "turnedOver": flipped}})
    return out


_CORD_CAPS = ("NONE", "KNOT", "AGLET", "STOPPER")
_AXIS_VEC = {"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}


def _cord_caps(raw, closed, tail):
    if raw is None:
        return []
    pair = [raw, raw] if isinstance(raw, str) else raw
    if not isinstance(pair, (list, tuple)) or len(pair) != 2 or not all(isinstance(c, str) for c in pair):
        raise MifOpError("endCaps is one of %s for both ends, or [start, end]. %s" % (", ".join(_CORD_CAPS), tail))
    pair = [c.upper() for c in pair]
    bad = [c for c in pair if c not in _CORD_CAPS]
    if bad:
        raise MifOpError("endCaps %s is not one of %s. %s" % (bad[0], ", ".join(_CORD_CAPS), tail))
    if closed and any(c != "NONE" for c in pair):
        raise MifOpError("endCaps finish a cord's two ends, and a closed tube has none. %s" % tail)
    return [(w, c) for w, c in zip(("START", "END"), pair) if c != "NONE"]


def _cord_stoppers(raw, closed, tail):
    if raw is None:
        return []
    rows = [raw] if isinstance(raw, dict) else raw
    if not isinstance(rows, (list, tuple)) or not rows:
        raise MifOpError("stoppers is a list of {at, object, boreRadius, boreAxis}. %s" % tail)
    if closed:
        raise MifOpError("stoppers slide onto an open cord, measured from its start. %s" % tail)
    out = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) - {"at", "object", "boreRadius", "boreAxis"} or "at" not in row:
            raise MifOpError("stoppers[%d] must be {at, object, boreRadius, boreAxis}, got %r. %s" % (i, row, tail))
        at = finite_float(row["at"], "stoppers.at")
        if at < 0:
            raise MifOpError("stoppers[%d].at must be 0 or above (metres from the cord's start). %s" % (i, tail))
        model = None
        if row.get("object") is not None:
            model = get_object(str(row["object"]), want_mesh=True)
            if row.get("boreRadius") is None:
                raise MifOpError("stoppers[%d] names a model - pass its boreRadius (in its own units) so its bore can "
                                 "be matched to the cord. %s" % (i, tail))
        elif row.get("boreRadius") is not None or row.get("boreAxis") is not None:
            raise MifOpError("stoppers[%d]: boreRadius and boreAxis describe a model - pass object too. %s"
                             % (i, tail))
        bore = finite_float(row["boreRadius"], "boreRadius") if row.get("boreRadius") is not None else None
        if bore is not None and not bore > 0:
            raise MifOpError("stoppers[%d].boreRadius must be above 0. %s" % (i, tail))
        axis = str(row.get("boreAxis", "Z")).upper()
        if axis not in _AXIS_VEC:
            raise MifOpError("stoppers[%d].boreAxis must be X, Y or Z. %s" % (i, tail))
        out.append({"at": at, "object": model, "boreRadius": bore, "boreAxis": axis})
    return out


def _along_path(pts, along, d):
    for i in range(len(pts) - 1):
        if along[i + 1] >= d or i == len(pts) - 2:
            seg = along[i + 1] - along[i]
            t = 0.0 if seg <= 0 else min(1.0, max(0.0, (d - along[i]) / seg))
            return pts[i].lerp(pts[i + 1], t), (pts[i + 1] - pts[i]).normalized()
    return pts[-1], (pts[-1] - pts[-2]).normalized()


def _closed_sweep(path, radius, sides):
    """A closed tube round a closed path (no corner widening) - the knot."""
    m = len(path)
    tans = [(path[(i + 1) % m] - path[i - 1]).normalized() for i in range(m)]
    t0 = tans[0]
    ref = Vector((0, 0, 1)) if abs(t0.z) < 0.9 else Vector((1, 0, 0))
    nx = (ref - t0 * ref.dot(t0)).normalized()
    verts, rings = [], []
    for i, t in enumerate(tans):
        if i:
            nx = tans[i - 1].rotation_difference(t) @ nx
            nx = (nx - t * nx.dot(t)).normalized()
        bx = t.cross(nx).normalized()
        rings.append(list(range(len(verts), len(verts) + sides)))
        for k in range(sides):
            ang = 2 * math.pi * k / sides
            verts.append(list(path[i] + (nx * math.cos(ang) + bx * math.sin(ang)) * radius))
    faces = []
    for i in range(m):
        a, b = rings[i], rings[(i + 1) % m]
        for k in range(sides):
            k1 = (k + 1) % sides
            faces.append([a[k], a[k1], b[k1], b[k]])
    if _signed_volume(verts, faces) < 0:
        faces = [list(reversed(f)) for f in faces]
    return verts, faces


def _cord_cap(kind, which, pts, along, radius, sides):
    """A cord end's finish as a closed part: (verts, faces, reply row). out points away from the cord."""
    if which == "START":
        end, out = pts[0], (pts[0] - pts[1]).normalized()
    else:
        end, out = pts[-1], (pts[-1] - pts[-2]).normalized()
    if kind == "AGLET":
        length, r2 = max(0.02, 6 * radius), radius * 1.3
        v, f = _revolve([(0, 0), (r2, 0), (r2, length), (0, length)], end + out * 0.0005, -out, sides)
        return v, f, {"end": which, "kind": kind, "length": round(length, 6), "radius": round(r2, 6)}
    if kind == "KNOT":
        k = 1.8 * radius
        e1 = out.orthogonal().normalized()
        e2 = out.cross(e1).normalized()
        centre = end + out * (2 * k)
        path = []
        for i in range(48):
            t = 2 * math.pi * i / 48
            x, y, z = math.sin(t) + 2 * math.sin(2 * t), math.cos(t) - 2 * math.cos(2 * t), -math.sin(3 * t)
            path.append(centre + (e1 * x + e2 * y + out * z) * k)
        v, f = _closed_sweep(path, radius, sides)
        return v, f, {"end": which, "kind": kind, "size": round(6 * k + 2 * radius, 6)}
    length, outer = 8 * radius, 3 * radius
    v, f = _revolve([(radius, 0), (outer * 0.8, 0), (outer, length * 0.2), (outer, length * 0.8), (outer * 0.8, length),
                     (radius, length)], end - out * length, out, sides, loop=True)
    return v, f, {"end": which, "kind": kind, "length": round(length, 6), "boreRadius": radius,
                  "outerRadius": round(outer, 6)}


# =============================================================================================================
# VEHICLE BODY (2026-10-06, spec "bl_build_vehicle_body: a parked car from a side profile and a plan")
# =============================================================================================================

_CAR_KEYS = ("name", "bottom", "belt", "roof", "halfWidth", "exponents", "roofWidthRatio", "stations", "sectionPoints",
             "arches", "glass", "materials", "location", "rotation", "scale", "collection", "takeOverName", "lids",
             "seams", "bumpers", "pockets", "mirrors", "archLips", "wheels")


def _curve(raw, key, tail):
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        raise MifOpError("%s must be two or more [x, value] points. %s" % (key, tail))
    pts = []
    for p in raw:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            raise MifOpError("each point of %s must be [x, value], got %r. %s" % (key, p, tail))
        pts.append(tuple(finite_floats(list(p), key)))
    if any(b[0] <= a[0] for a, b in zip(pts, pts[1:])):
        raise MifOpError("%s's x must rise from point to point. %s" % (key, tail))
    return pts


def _at(curve, x):
    """The curve's value at x, or None OUTSIDE its own x range - never its end value held past the end (the lesson:
    a roof held at its last height over a falling hood stood a 20 cm slab on a sedan's nose)."""
    if x < curve[0][0] - 1e-9 or x > curve[-1][0] + 1e-9:
        return None
    for (x0, v0), (x1, v1) in zip(curve, curve[1:]):
        if x0 - 1e-9 <= x <= x1 + 1e-9:
            t = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
            return v0 + (v1 - v0) * min(1.0, max(0.0, t))
    return curve[-1][1]


def op_build_vehicle_body(params):
    """A parked car's body shell lofted from side-profile curves and a plan - closed, faced out, with glass slots.

    THE GAP (2026-10-04): three street cars came from a ~1,000-line local script that lofted half sections built from
    bottom, belt and roof profile curves and a plan, then went in through create_mesh. This is that loft, typed.
    Curves are [[x, value], ...] with x rising (metres, x along the car, front at the low x): bottom (the underside's
    height, with the wheel arches cut into it), belt (the waistline's height, where the body is widest), roof
    (optional; the roof line's height) and halfWidth (the plan's half width at the belt). A curve applies ONLY INSIDE
    ITS OWN x RANGE: outside the roof's range a section tops out at the belt - a hood or a boot, not a roof held at its
    last height. bottom, belt and halfWidth must cover the whole length (their common range); stations (default 48)
    sections are taken along it.

    Each half section is sectionPoints (default 19, odd) points: a superellipse quarter of exponent exponents.lower
    (default 4: boxy at bumper level) from the underside's centre out to the belt, then where a roof applies a quarter of
    exponents.upper (default 2.5: rounder) from the belt in to the roof's centre, its half width roofWidthRatio (0.72)
    of the belt's (the glasshouse's tumblehome). Mirrored, skinned, and capped at both ends into one closed solid,
    turned out by signed volume.

    arches [{x, radius, height}] cut a circle (centre x, at height above the ground) up into the bottom line - the
    wheel arches. glass [[x0, x1], ...] puts the glasshouse's faces (above the belt) inside those x ranges on the glass
    slot. materials {body, glass} name existing materials for the two slots. name, location, rotation, scale,
    collection and takeOverName are create_mesh's. Reply adds length, roofRange, stationsWithRoof, glassFaces and
    outwardCheck.

    THE PARTS (2026-10-06), each optional, every one in the body's local space and parented to it:
    lids [{x0, x1, inset 0.06, width 0.006, depth 0.004}] - a hood or boot lid on the top where no roof applies, its
    shut line a groove round it (three cuts, the middle pressed in by depth) between the fender crowns (inset in from
    the plan's half width). seams [{x, zMin, zMax, width, depth, side BOTH | LEFT | RIGHT}] - a pressed vertical seam
    down the side (a door's shut line). bumpers [{end FRONT | REAR, length 0.3, zMin, zMax, gap 0.01, thickness
    0.03}] - a separate shell copied from the body's faces in that band, standing gap off it (gapMeasured).
    pockets [{kind LAMP | GRILLE | PLATE | HANDLE, at FRONT | REAR (HANDLE: LEFT | RIGHT | SIDES), center [y or x, z],
    size [w, h], depth, mirror}] - the rectangle cut into the end cap or the side and pushed in; a LAMP gets a lens
    shell clipped from the body surface there (flush, lensThickness 0.008), an emissive band on the pocket floor and
    (bowls, default true) two reflector bowls with a bulb in each,
    a GRILLE slats, a PLATE a plate and raised characters (text, charHeight, raise) from Blender's built-in font, a
    HANDLE a bar in its dish. mirrors [{x, z, size [along, out, up], reach}] - a housing on an arm each side, its rear
    face on the glass slot. archLips {width, out, embed, segments} - a band round every arch each side. wheels
    {radius, width, rimRadius, inset, segments} - a tyre and rim through build_lathe in every arch, standing on the
    ground. materials adds lens, lamp, trim and tyre for the parts. Reply: parts {lids, seams, bumpers, pockets,
    mirrors, archLips, wheels, objects}.

    Refused before anything exists: a curve whose x does not rise, bottom / belt / halfWidth not covering one common
    length, a belt not above the bottom or a roof not above the belt where both apply, a half width not above 0,
    stations under 4 or over 400, sectionPoints even or under 5, a glass range outside the length; a lid under the
    roof, a seam outside the side, a pocket outside the end's outline or the side, overlapping another or in a
    bumper's band, a mirrored pocket crossing the centre line, a wheel that does not fit its arch, lips or wheels with
    no arches. A part that does not build cleanly removes the body and every part made.
    """
    tail = "NOTHING was created."
    reject_unknown(params, _CAR_KEYS, "build_vehicle_body")
    bottom = _curve(params.get("bottom"), "bottom", tail)
    belt = _curve(params.get("belt"), "belt", tail)
    half_w = _curve(params.get("halfWidth"), "halfWidth", tail)
    roof = _curve(params["roof"], "roof", tail) if params.get("roof") is not None else None
    x0 = max(c[0][0] for c in (bottom, belt, half_w))
    x1 = min(c[-1][0] for c in (bottom, belt, half_w))
    if not x1 - x0 > 1e-6:
        raise MifOpError("bottom, belt and halfWidth share no length (their common x range is %g to %g). %s"
                         % (x0, x1, tail))
    exps = params.get("exponents") or {}
    if not isinstance(exps, dict) or set(exps) - {"lower", "upper"}:
        raise MifOpError("exponents is {lower, upper}, got %r. %s" % (exps, tail))
    e_low = finite_float(exps.get("lower", 4.0), "exponents.lower")
    e_up = finite_float(exps.get("upper", 2.5), "exponents.upper")
    if not (e_low > 0.5 and e_up > 0.5):
        raise MifOpError("exponents must be above 0.5. %s" % tail)
    ratio = take_float(params, "roofWidthRatio", default=0.72)
    if not 0 < ratio <= 1:
        raise MifOpError("roofWidthRatio must be above 0 and at most 1, got %g. %s" % (ratio, tail))
    n_st = take_int(params, "stations", default=48)
    n_sec = take_int(params, "sectionPoints", default=19)
    if not 4 <= n_st <= 400:
        raise MifOpError("stations must be 4-400, got %d. %s" % (n_st, tail))
    if n_sec < 5 or n_sec % 2 == 0:
        raise MifOpError("sectionPoints must be odd and at least 5, got %d. %s" % (n_sec, tail))
    arches = params.get("arches") or []
    if not isinstance(arches, (list, tuple)):
        raise MifOpError("arches is a list of {x, radius, height}. %s" % tail)
    arch_rows = []
    for a in arches:
        if not isinstance(a, dict) or set(a) - {"x", "radius", "height"} or "x" not in a or "radius" not in a:
            raise MifOpError("each arch is {x, radius, height (default: radius)}, got %r. %s" % (a, tail))
        r = finite_float(a["radius"], "radius")
        if not r > 0:
            raise MifOpError("an arch radius must be above 0. %s" % tail)
        arch_rows.append((finite_float(a["x"], "x"), r, finite_float(a.get("height", r), "height")))
    glass = params.get("glass") or []
    if not isinstance(glass, (list, tuple)) or any(not isinstance(g, (list, tuple)) or len(g) != 2 for g in glass):
        raise MifOpError("glass is a list of [x0, x1] ranges. %s" % tail)
    glass = [tuple(finite_floats(list(g), "glass")) for g in glass]
    for g in glass:
        if not (x0 - 1e-9 <= g[0] < g[1] <= x1 + 1e-9):
            raise MifOpError("glass range %s is not inside the body's length %g to %g. %s" % (list(g), x0, x1, tail))
    mats = params.get("materials")
    if mats is not None:
        if not isinstance(mats, dict) or set(mats) - set(_CAR_MATS):
            raise MifOpError("materials is {%s}, got %r. %s" % (", ".join(_CAR_MATS), mats, tail))
        have = set(bpy.data.materials.keys())
        missing = [m for m in mats.values() if m not in have]
        if missing:
            raise MifOpError("no material named %s. %s" % (", ".join(repr(m) for m in missing), tail))

    q = (n_sec - 1) // 2          # points per quarter, belt shared
    sections, roof_on = [], []
    for k in range(n_st):
        x = x0 + (x1 - x0) * k / (n_st - 1)
        zb, zt, w = _at(bottom, x), _at(belt, x), _at(half_w, x)
        for xc, r, h in arch_rows:
            if abs(x - xc) < r:
                zb = max(zb, h + math.sqrt(r * r - (x - xc) ** 2))
        zr = _at(roof, x) if roof is not None else None
        if not w > 0:
            raise MifOpError("halfWidth at x %g is %g - it must be above 0 over the body. %s" % (x, w, tail))
        if not zt > zb + 1e-6:
            raise MifOpError("at x %g the belt (%g) is not above the bottom (%g)%s. %s"
                             % (x, zt, zb, " with the arches cut in" if arch_rows else "", tail))
        if zr is not None and not zr > zt + 1e-6:
            zr = None              # the roof meets the belt: no glasshouse at this station
        roof_on.append(zr is not None)
        half = []
        for i in range(q + 1):     # underside centre -> belt, superellipse of e_low
            t = (math.pi / 2) * i / q
            y = w * math.sin(t) ** (2.0 / e_low)
            z = zt - (zt - zb) * math.cos(t) ** (2.0 / e_low)
            half.append((y, z))
        top_w = w * ratio
        for i in range(1, q + 1):  # belt -> top centre
            t = (math.pi / 2) * i / q
            if zr is None:
                y = w * math.cos(t) ** (2.0 / e_low)
                z = zt
            else:
                yy = top_w + (w - top_w) * (1 - i / q)
                y = yy * math.cos(t) ** (2.0 / e_up)
                z = zt + (zr - zt) * math.sin(t) ** (2.0 / e_up)
            half.append((y, z))
        sections.append((x, half))
    # THE PARTS, checked against the body's own curves before anything is made
    def _zb(x):
        z = _at(bottom, x)
        for xc, r, h in arch_rows:
            if abs(x - xc) < r:
                z = max(z, h + math.sqrt(r * r - (x - xc) ** 2))
        return z

    def _roof_z(x):
        zr = _at(roof, x) if roof is not None else None
        return zr if zr is not None and zr > _at(belt, x) + 1e-6 else None

    def _end_poly(half):
        return [(y, z) for y, z in half] + [(-y, z) for y, z in reversed(half[1:-1])]
    geo = {"x0": x0, "x1": x1, "w": lambda x: _at(half_w, x), "zt": lambda x: _at(belt, x), "zb": _zb,
           "zb0": lambda x: _at(bottom, x), "roof": _roof_z, "arches": arch_rows,
           "arched": lambda x: any(abs(x - xc) < r for xc, r, _h in arch_rows),
           "endPoly": {"FRONT": _end_poly(sections[0][1]), "REAR": _end_poly(sections[-1][1])}}
    part_plan, part_mats = _car_parse_parts(params, geo, tail)
    any_parts = any(part_plan.values())
    # one ring per station: right half (y -> -y) then left, sharing the two centre points
    verts, rings = [], []
    for x, half in sections:
        ring = []
        for y, z in half:                       # bottom centre round the right side to the top centre
            ring.append(len(verts))
            verts.append([x, -y, z])
        for y, z in reversed(half[1:-1]):       # down the left side, back to just above bottom centre
            ring.append(len(verts))
            verts.append([x, y, z])
        rings.append(ring)
    m = len(rings[0])
    faces, slot_of, uvs = [], [], []
    for k in range(n_st - 1):
        a, b = rings[k], rings[k + 1]
        xm = (sections[k][0] + sections[k + 1][0]) / 2
        for j in range(m):
            j1 = (j + 1) % m
            faces.append([a[j], b[j], b[j1], a[j1]])
            upper = (q <= j < 3 * q) and roof_on[k] and roof_on[k + 1]
            is_glass = upper and any(g0 <= xm <= g1 for g0, g1 in glass)
            slot_of.append(1 if is_glass else 0)
            uvs.append([[k / (n_st - 1), j / m], [(k + 1) / (n_st - 1), j / m], [(k + 1) / (n_st - 1), (j + 1) / m],
                        [k / (n_st - 1), (j + 1) / m]])
    # the sides walk the first ring j1 -> j and the last j -> j1, so the caps walk them the other way
    # A UV ON EACH CORNER, not one point: every corner on (0.5, 0.5) gave Unreal 98 near-zero tangents. The cap is flat in
    # x, so its own y and z, boxed onto 0-1, are its projection.
    for ring, rev in ((rings[0], False), (rings[-1], True)):
        f = list(reversed(ring)) if rev else list(ring)
        faces.append(f)
        slot_of.append(0)
        ys, zs = [verts[i][1] for i in f], [verts[i][2] for i in f]
        sy, sz = max(max(ys) - min(ys), 1e-6), max(max(zs) - min(zs), 1e-6)
        uvs.append([[(verts[i][1] - min(ys)) / sy, (verts[i][2] - min(zs)) / sz] for i in f])
    vol = _signed_volume(verts, faces)
    flipped = vol < 0
    if flipped:
        faces = [list(reversed(f)) for f in faces]
        uvs = [list(reversed(u)) for u in uvs]
        vol = -vol
    from .ops_create import op_create_mesh   # noqa: PLC0415
    sub = {"name": params.get("name") or "VehicleBody", "vertices": verts, "faces": faces, "uvs": {"UVMap": uvs},
           "materialIndices": slot_of}
    if mats and all(mats.get(k) for k in ("body", "glass")):
        sub["materials"] = [mats["body"], mats["glass"]]
    for k in ("location", "rotation", "scale", "collection", "takeOverName"):
        if params.get(k) is not None:
            sub[k] = params[k]
    out = op_create_mesh(sub)
    obj = bpy.data.objects[out["name"]]
    # The two end caps are the shell's only n-gons; a section need not be star-shaped round its middle, so they are
    # split by the triangulator rather than fanned.
    bm_caps = bmesh.new()
    try:
        bm_caps.from_mesh(obj.data)
        triangulate_ngons(bm_caps)
        bm_caps.to_mesh(obj.data)
        obj.data.update()
    finally:
        bm_caps.free()
    out.update({"vertexCount": len(obj.data.vertices), "edgeCount": len(obj.data.edges),
                "faceCount": len(obj.data.polygons)})
    if mats and not all(mats.get(k) for k in ("body", "glass")):
        for slot, key in zip(obj.material_slots, ("body", "glass")):
            if mats.get(key):
                slot.material = bpy.data.materials[mats[key]]
        out["materialSlots"] = [s.material.name if s.material else None for s in obj.material_slots]
    if any_parts:
        made = []
        try:
            out["parts"] = _car_build_parts(obj, part_plan, geo, part_mats, params, made)
            out.update({"vertexCount": len(obj.data.vertices), "edgeCount": len(obj.data.edges),
                        "faceCount": len(obj.data.polygons)})
        except Exception as exc:
            for po in made + [obj]:
                data = po.data
                remove_objects([po])
                if data is not None and not data.users:
                    bpy.data.meshes.remove(data)
            if isinstance(exc, MifOpError):
                raise MifOpError("%s The body and its parts were removed - NOTHING was created." % exc)
            raise
    with_roof = [sections[k][0] for k in range(n_st) if roof_on[k]]
    out.update({"op": "build_vehicle_body", "length": round(x1 - x0, 6), "xRange": [round(x0, 6), round(x1, 6)],
                "stations": n_st, "sectionPoints": n_sec, "stationsWithRoof": len(with_roof),
                "roofRange": [round(min(with_roof), 6), round(max(with_roof), 6)] if with_roof else None,
                "glassFaces": sum(slot_of), "slots": ["body", "glass"],
                "outwardCheck": {"signedVolume": round(vol, 9), "closed": True, "turnedOver": flipped}})
    return out


# ---- VEHICLE PARTS (2026-10-06): lids and shut lines, pressed seams, bumper shells, lamp / grille / plate / handle
# pockets with lens shells, emissive bands, slats, plates and raised characters, mirrors, arch lips and lathed wheels.
# Every part is parsed and checked against the body's own curves before anything is made; the body mesh is then
# edited in place (grooves pressed, pockets cut) and the parts are made as objects parented to it, in its local space.

_CAR_PARTS = ("lids", "seams", "bumpers", "pockets", "mirrors", "archLips", "wheels")
_CAR_MATS = ("body", "glass", "lens", "lamp", "trim", "tyre")
_POCKET_KINDS = {"LAMP": ("FRONT", "REAR"), "GRILLE": ("FRONT", "REAR"), "PLATE": ("FRONT", "REAR"),
                 "HANDLE": ("LEFT", "RIGHT", "SIDES")}


def _car_rows(raw, key, allowed, required, tail):
    if raw is None:
        return []
    rows = [raw] if isinstance(raw, dict) else raw
    if not isinstance(rows, (list, tuple)):
        raise MifOpError("%s is a list of {%s}. %s" % (key, ", ".join(allowed), tail))
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise MifOpError("%s[%d] must be {%s}, got %r. %s" % (key, i, ", ".join(allowed), row, tail))
        bad = sorted(set(row) - set(allowed))
        if bad:
            raise MifOpError("%s[%d] has unknown key(s) %s - accepted: %s. %s"
                             % (key, i, ", ".join(bad), ", ".join(allowed), tail))
        miss = [k for k in required if row.get(k) is None]
        if miss:
            raise MifOpError("%s[%d] needs %s. %s" % (key, i, ", ".join(miss), tail))
    return list(rows)


def _car_num(row, key, default, label, tail, lo=None, hi=None):
    val = row.get(key, default)
    if val is None:
        return None
    val = finite_float(val, label)
    if (lo is not None and not val > lo) or (hi is not None and not val < hi):
        raise MifOpError("%s must be above %s%s, got %g. %s"
                         % (label, lo, "" if hi is None else " and below %s" % hi, val, tail))
    return val


def _car_pair(row, key, default, label, tail):
    val = row.get(key, default)
    if not isinstance(val, (list, tuple)) or len(val) != 2:
        raise MifOpError("%s must be [a, b], got %r. %s" % (label, val, tail))
    return tuple(finite_floats(list(val), label))


def _in_poly(pt, poly):
    y, z = pt
    inside = False
    for (y0, z0), (y1, z1) in zip(poly, poly[1:] + poly[:1]):
        if (z0 > z) != (z1 > z) and y < y0 + (z - z0) * (y1 - y0) / (z1 - z0):
            inside = not inside
    return inside


def _car_parse_parts(params, geo, tail):
    """Every part row checked against the body's curves; returns the plan. Writes nothing."""
    x0, x1 = geo["x0"], geo["x1"]
    plan = {k: [] for k in _CAR_PARTS}
    for i, row in enumerate(_car_rows(params.get("lids"), "lids", ("x0", "x1", "inset", "width", "depth"),
                                      ("x0", "x1"), tail)):
        a, b = finite_float(row["x0"], "lids.x0"), finite_float(row["x1"], "lids.x1")
        inset = _car_num(row, "inset", 0.06, "lids[%d].inset" % i, tail, 0)
        width = _car_num(row, "width", 0.006, "lids[%d].width" % i, tail, 0, 0.05)
        depth = _car_num(row, "depth", 0.004, "lids[%d].depth" % i, tail, 0, 0.05)
        if not (x0 + width < a < b < x1 - width):
            raise MifOpError("lids[%d] %g-%g must lie inside the body's length (%g-%g) by its groove width. %s"
                             % (i, a, b, x0, x1, tail))
        xs = [a + (b - a) * k / 20.0 for k in range(21)]
        roofed = [x for x in xs if geo["roof"](x) is not None]
        if roofed:
            raise MifOpError("lids[%d] %g-%g runs under the roof (it applies at x %g) - a lid is a hood or boot top, "
                             "where no roof applies. %s" % (i, a, b, roofed[0], tail))
        yc = min(geo["w"](x) for x in xs) - inset
        if not yc > 0.05:
            raise MifOpError("lids[%d] inset %g leaves no lid between the fender crowns. %s" % (i, inset, tail))
        plan["lids"].append({"x0": a, "x1": b, "yc": yc, "width": width, "depth": depth})
    for i, row in enumerate(_car_rows(params.get("seams"), "seams", ("x", "zMin", "zMax", "width", "depth", "side"),
                                      ("x",), tail)):
        x = finite_float(row["x"], "seams.x")
        if not x0 + 0.05 < x < x1 - 0.05:
            raise MifOpError("seams[%d] x %g is not inside the body's length (%g-%g). %s" % (i, x, x0, x1, tail))
        zb, zt = geo["zb"](x), geo["zt"](x)
        z_lo = _car_num(row, "zMin", zb + 0.04, "seams[%d].zMin" % i, tail)
        z_hi = _car_num(row, "zMax", zt - 0.02, "seams[%d].zMax" % i, tail)
        if not zb < z_lo < z_hi <= zt:
            raise MifOpError("seams[%d] at x %g runs %g-%g; the side there is %g (bottom%s) to %g (belt). %s"
                             % (i, x, z_lo, z_hi, zb, ", arch cut in" if geo["arched"](x) else "", zt, tail))
        side = str(row.get("side", "BOTH")).upper()
        if side not in ("BOTH", "LEFT", "RIGHT"):
            raise MifOpError("seams[%d].side must be BOTH, LEFT or RIGHT, got %r. %s" % (i, row.get("side"), tail))
        plan["seams"].append({"x": x, "zMin": z_lo, "zMax": z_hi, "side": side,
                              "width": _car_num(row, "width", 0.006, "seams[%d].width" % i, tail, 0, 0.05),
                              "depth": _car_num(row, "depth", 0.004, "seams[%d].depth" % i, tail, 0, 0.05)})
    for i, row in enumerate(_car_rows(params.get("bumpers"), "bumpers",
                                      ("end", "length", "zMin", "zMax", "gap", "thickness"),
                                      ("end", "zMin", "zMax"), tail)):
        end = str(row["end"]).upper()
        if end not in ("FRONT", "REAR"):
            raise MifOpError("bumpers[%d].end must be FRONT or REAR, got %r. %s" % (i, row["end"], tail))
        length = _car_num(row, "length", 0.3, "bumpers[%d].length" % i, tail, 0, (x1 - x0) / 2)
        xe = x0 if end == "FRONT" else x1
        z_lo, z_hi = finite_float(row["zMin"], "zMin"), finite_float(row["zMax"], "zMax")
        if not geo["zb"](xe) <= z_lo < z_hi <= geo["zt"](xe):
            raise MifOpError("bumpers[%d] %g-%g is not between the %s end's bottom (%g) and belt (%g). %s"
                             % (i, z_lo, z_hi, end.lower(), geo["zb"](xe), geo["zt"](xe), tail))
        if any(b["end"] == end for b in plan["bumpers"]):
            raise MifOpError("bumpers[%d]: the %s end already has a bumper. %s" % (i, end.lower(), tail))
        plan["bumpers"].append({"end": end, "length": length, "zMin": z_lo, "zMax": z_hi,
                                "gap": _car_num(row, "gap", 0.01, "bumpers[%d].gap" % i, tail, 0, 0.2),
                                "thickness": _car_num(row, "thickness", 0.03, "bumpers[%d].thickness" % i, tail,
                                                      0, 0.3)})
    rects = []
    for i, row in enumerate(_car_rows(params.get("pockets"), "pockets",
                                      ("kind", "at", "center", "size", "depth", "mirror", "lens", "lensThickness",
                                       "band", "bowls", "slats", "text", "charHeight", "raise"),
                                      ("kind", "center"), tail)):
        kind = str(row["kind"]).upper()
        if kind not in _POCKET_KINDS:
            raise MifOpError("pockets[%d].kind must be one of %s, got %r. %s"
                             % (i, ", ".join(sorted(_POCKET_KINDS)), row["kind"], tail))
        at = str(row.get("at", _POCKET_KINDS[kind][0] if kind != "HANDLE" else "SIDES")).upper()
        if at not in _POCKET_KINDS[kind]:
            raise MifOpError("pockets[%d] a %s pocket goes at %s, got %r. %s"
                             % (i, kind, " or ".join(_POCKET_KINDS[kind]), row.get("at"), tail))
        cu, cz = _car_pair(row, "center", None, "pockets[%d].center" % i, tail)
        dflt = {"LAMP": (0.3, 0.12), "GRILLE": (0.6, 0.15), "PLATE": (0.52, 0.12), "HANDLE": (0.2, 0.06)}[kind]
        sw, sh = _car_pair(row, "size", dflt, "pockets[%d].size" % i, tail)
        if not (sw > 0.01 and sh > 0.01):
            raise MifOpError("pockets[%d].size must be above 0.01 each, got %r. %s" % (i, [sw, sh], tail))
        depth = _car_num(row, "depth", {"LAMP": 0.06, "GRILLE": 0.05, "PLATE": 0.012, "HANDLE": 0.015}[kind],
                         "pockets[%d].depth" % i, tail, 0, 0.3)
        p = {"kind": kind, "at": at, "center": (cu, cz), "size": (sw, sh), "depth": depth}
        for key in ("mirror", "lens", "band", "bowls", "slats", "text", "charHeight", "raise", "lensThickness"):
            ok = {"mirror": kind != "HANDLE", "lens": kind == "LAMP", "band": kind == "LAMP", "bowls": kind == "LAMP",
                  "slats": kind == "GRILLE",
                  "text": kind == "PLATE", "charHeight": kind == "PLATE", "raise": kind == "PLATE",
                  "lensThickness": kind == "LAMP"}[key]
            if row.get(key) is not None and not ok:
                raise MifOpError("pockets[%d].%s does not apply to a %s pocket. %s" % (i, key, kind, tail))
        p["mirror"] = bool(row.get("mirror", False))
        p["lens"] = kind == "LAMP" and bool(row.get("lens", True))
        p["band"] = kind == "LAMP" and bool(row.get("band", True))
        p["bowls"] = kind == "LAMP" and bool(row.get("bowls", True))
        p["lensThickness"] = _car_num(row, "lensThickness", 0.008, "pockets[%d].lensThickness" % i, tail, 0, depth)
        p["slats"] = int(row.get("slats", 4)) if kind == "GRILLE" else 0
        if kind == "GRILLE" and not 1 <= p["slats"] <= 20:
            raise MifOpError("pockets[%d].slats must be 1-20. %s" % (i, tail))
        if kind == "PLATE":
            text = row.get("text", "")
            if not isinstance(text, str) or len(text) > 12 or "\n" in text:
                raise MifOpError("pockets[%d].text is one line of up to 12 characters. %s" % (i, tail))
            p["text"] = text.strip()
            p["charHeight"] = _car_num(row, "charHeight", sh * 0.6, "pockets[%d].charHeight" % i, tail, 0, sh)
            p["raise"] = _car_num(row, "raise", 0.002, "pockets[%d].raise" % i, tail, 0, 0.02)
            if not 0.003 + p["raise"] < depth:
                raise MifOpError("pockets[%d] depth %g holds no plate (3 mm) and characters raised %g. %s"
                                 % (i, depth, p["raise"], tail))
        instances = []
        if at in ("FRONT", "REAR"):
            xe = x0 if at == "FRONT" else x1
            poly = geo["endPoly"][at]
            ys = [cu, -cu] if p["mirror"] else [cu]
            if p["mirror"] and not abs(cu) - sw / 2 > 0.005:
                raise MifOpError("pockets[%d] is mirrored and its two copies overlap at the centre line (center %g, "
                                 "width %g). %s" % (i, cu, sw, tail))
            for y in ys:
                corners = [(y + dy * sw / 2, cz + dz * sh / 2) for dy in (-1.1, 1.1) for dz in (-1.1, 1.1)]
                if not all(_in_poly(c, poly) for c in corners):
                    raise MifOpError("pockets[%d] (%s at y %g, z %g, %g x %g) is not inside the %s end's outline. %s"
                                     % (i, kind, y, cz, sw, sh, at.lower(), tail))
                if depth >= (x1 - x0) / 4:
                    raise MifOpError("pockets[%d] depth %g is past a quarter of the body. %s" % (i, depth, tail))
                instances.append({"side": at, "u": y, "plane": xe})
            for b in plan["bumpers"]:
                if b["end"] == at and cz - sh / 2 < b["zMax"] and cz + sh / 2 > b["zMin"]:
                    raise MifOpError("pockets[%d] (%s) lies in the %s bumper's band %g-%g, where the bumper shell would "
                                     "hide it - put it above or below. %s"
                                     % (i, kind, at.lower(), b["zMin"], b["zMax"], tail))
        else:
            sides = {"LEFT": [1], "RIGHT": [-1], "SIDES": [1, -1]}[at]
            xs = [cu - sw / 2 + sw * k / 10.0 for k in range(11)]
            if not (x0 + 0.05 < xs[0] and xs[-1] < x1 - 0.05):
                raise MifOpError("pockets[%d] x %g-%g is not inside the body's length. %s"
                                 % (i, xs[0], xs[-1], tail))
            for x in xs:
                if not geo["zb"](x) + 0.03 < cz - sh / 2 and cz + sh / 2 < geo["zt"](x) - 0.01:
                    raise MifOpError("pockets[%d] z %g-%g at x %g is not on the side between the bottom (%g) and the "
                                     "belt (%g). %s" % (i, cz - sh / 2, cz + sh / 2, x, geo["zb"](x), geo["zt"](x),
                                                        tail))
            for s in plan["seams"]:
                if abs(s["x"] - cu) < sw / 2 + s["width"]:
                    raise MifOpError("pockets[%d] spans the seam at x %g. %s" % (i, s["x"], tail))
            for s in sides:
                instances.append({"side": "LEFT" if s > 0 else "RIGHT", "u": cu, "sign": s})
        for inst in instances:
            box = (inst["side"], inst["u"] - sw / 2, inst["u"] + sw / 2, cz - sh / 2, cz + sh / 2)
            for other in rects:
                if other[0] == box[0] and box[1] < other[2] and other[1] < box[2] and box[3] < other[4] \
                        and other[3] < box[4]:
                    raise MifOpError("pockets[%d] overlaps another pocket on the %s. %s"
                                     % (i, box[0].lower(), tail))
            rects.append(box)
        p["instances"] = instances
        plan["pockets"].append(p)
    for i, row in enumerate(_car_rows(params.get("mirrors"), "mirrors", ("x", "z", "size", "reach"), ("x", "z"), tail)):
        x, z = finite_float(row["x"], "mirrors.x"), finite_float(row["z"], "mirrors.z")
        if not (x0 + 0.1 < x < x1 - 0.1 and geo["zb"](x) < z < (geo["roof"](x) or geo["zt"](x)) + 0.2):
            raise MifOpError("mirrors[%d] at x %g, z %g is not beside the body. %s" % (i, x, z, tail))
        size = row.get("size", [0.1, 0.18, 0.11])
        if not isinstance(size, (list, tuple)) or len(size) != 3:
            raise MifOpError("mirrors[%d].size is [along, out, up]. %s" % (i, tail))
        size = finite_floats(list(size), "mirrors.size")
        if not all(v > 0.01 for v in size):
            raise MifOpError("mirrors[%d].size must be above 0.01 each. %s" % (i, tail))
        plan["mirrors"].append({"x": x, "z": z, "size": size,
                                "reach": _car_num(row, "reach", 0.06, "mirrors[%d].reach" % i, tail, 0, 1)})
    lips = params.get("archLips")
    if lips is not None:
        row = _car_rows(lips, "archLips", ("width", "out", "embed", "segments"), (), tail)
        if len(row) != 1:
            raise MifOpError("archLips is one {width, out, embed, segments} for every arch. %s" % tail)
        row = row[0]
        if not geo["arches"]:
            raise MifOpError("archLips frames the wheel arches, and there are none - pass arches. %s" % tail)
        for xc, r, h in geo["arches"]:
            if not h + r > geo["zb0"](xc) + 0.01:
                raise MifOpError("the arch at x %g does not reach up into the body, so it has no lip. %s" % (xc, tail))
        plan["archLips"].append({"width": _car_num(row, "width", 0.04, "archLips.width", tail, 0, 0.3),
                                 "out": _car_num(row, "out", 0.015, "archLips.out", tail, 0, 0.2),
                                 "embed": _car_num(row, "embed", 0.005, "archLips.embed", tail, 0, 0.1),
                                 "segments": int(row.get("segments", 16))})
        if not 4 <= plan["archLips"][0]["segments"] <= 128:
            raise MifOpError("archLips.segments must be 4-128. %s" % tail)
    wheels = params.get("wheels")
    if wheels is not None:
        row = _car_rows(wheels, "wheels", ("radius", "width", "rimRadius", "inset", "segments"), (), tail)
        if len(row) != 1:
            raise MifOpError("wheels is one {radius, width, rimRadius, inset, segments} for every arch. %s" % tail)
        row = row[0]
        if not geo["arches"]:
            raise MifOpError("wheels go in the wheel arches, and there are none - pass arches. %s" % tail)
        rad = _car_num(row, "radius", None, "wheels.radius", tail, 0)
        width = _car_num(row, "width", 0.2, "wheels.width", tail, 0)
        inset = _car_num(row, "inset", 0.03, "wheels.inset", tail, -0.1, 0.5)
        segs = int(row.get("segments", 32))
        if not 8 <= segs <= 256:
            raise MifOpError("wheels.segments must be 8-256. %s" % tail)
        rows = []
        for xc, r, h in geo["arches"]:
            R = rad if rad is not None else r - 0.025
            if not R > 0.05 or abs(h - R) + R > r + 1e-6:
                raise MifOpError("a wheel of radius %g standing on the ground (centre at %g) does not fit the arch at "
                                 "x %g (radius %g, centre height %g). %s" % (R, R, xc, r, h, tail))
            rim = _car_num(row, "rimRadius", R * 0.62, "wheels.rimRadius", tail, 0, R - 0.03)
            half = geo["w"](xc) - inset - width / 2
            if not half > width / 2:
                raise MifOpError("wheels %g wide at inset %g do not fit the body's half width %g at x %g. %s"
                                 % (width, inset, geo["w"](xc), xc, tail))
            rows.append({"x": xc, "radius": R, "rim": rim, "y": half})
        plan["wheels"] = [{"width": width, "segments": segs, "rows": rows}]
    mats = params.get("materials") or {}
    return plan, mats


def _box_lohi(lo, hi):
    (ax, ay, az), (bx, by, bz) = lo, hi
    v = [[x, y, z] for x in (ax, bx) for y in (ay, by) for z in (az, bz)]
    # index = 4*ix + 2*iy + iz; faces wound outward: -x, +x, -y, +y, -z, +z
    f = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]
    return v, f


def _shell(pverts, pfaces, normals, d_front, d_back):
    """A closed slab from an open patch: the patch moved d_front along its normals (facing them), d_back (facing back),
    walls round its boundary."""
    n = len(pverts)
    verts = [list(Vector(p) + Vector(q) * d_front) for p, q in zip(pverts, normals)]
    verts += [list(Vector(p) + Vector(q) * d_back) for p, q in zip(pverts, normals)]
    faces = [list(f) for f in pfaces] + [[i + n for i in reversed(f)] for f in pfaces]
    directed = {(a, b) for f in pfaces for a, b in zip(f, list(f[1:]) + [f[0]])}
    for a, b in sorted(directed):
        if (b, a) not in directed:
            faces.append([b, a, a + n, b + n])
    return verts, faces


def _revolve(profile, center, axis, n, loop=False):
    """A closed solid of revolution: profile [(r, a)] (r from the axis, a along it), its two ends on the axis - or,
    loop true, a closed profile off the axis (a ring with a bore)."""
    axis = Vector(axis).normalized()
    ref = Vector((0, 0, 1)) if abs(axis.z) < 0.9 else Vector((1, 0, 0))
    e1 = axis.cross(ref).normalized()
    e2 = axis.cross(e1)
    verts, rings = [], []
    for r, a in profile:
        c = Vector(center) + axis * a
        if r < 1e-9:
            rings.append([len(verts)])
            verts.append(list(c))
            continue
        ring = []
        for k in range(n):
            t = 2 * math.pi * k / n
            ring.append(len(verts))
            verts.append(list(c + (e1 * math.cos(t) + e2 * math.sin(t)) * r))
        rings.append(ring)
    faces = []
    for ra, rb in zip(rings, rings[1:] + (rings[:1] if loop else [])):
        for k in range(n):
            k1 = (k + 1) % n
            if len(ra) == 1:
                faces.append([ra[0], rb[k1], rb[k]])
            elif len(rb) == 1:
                faces.append([ra[k], ra[k1], rb[0]])
            else:
                faces.append([ra[k], ra[k1], rb[k1], rb[k]])
    if _signed_volume(verts, faces) < 0:
        faces = [list(reversed(f)) for f in faces]
    return verts, faces


class _Parts:
    """Geometry gathered per part object: verts, faces, a material key per face."""

    def __init__(self):
        self.objs = {}

    def add(self, name, verts, faces, mat="trim", mats_per_face=None):
        v, f, m = self.objs.setdefault(name, ([], [], []))
        base = len(v)
        v.extend(list(p) for p in verts)
        f.extend([i + base for i in face] for face in faces)
        m.extend(mats_per_face or [mat] * len(faces))


def _face_uvs(verts, faces):
    uvs = []
    for f in faces:
        pts = [Vector(verts[i]) for i in f]
        nrm = Vector((0, 0, 0))
        for k in range(len(pts)):
            nrm += pts[k].cross(pts[(k + 1) % len(pts)])
        ax = max(range(3), key=lambda a: abs(nrm[a]))
        u, v = [(1, 2), (0, 2), (0, 1)][ax]
        uvs.append([[p[u], p[v]] for p in pts])
    return uvs


def _car_cut(bm, face_ok, co, no):
    bm.normal_update()
    faces = [f for f in bm.faces if face_ok(f)]
    if not faces:
        return
    geom = faces + list({e for f in faces for e in f.edges}) + list({v for f in faces for v in f.verts})
    bmesh.ops.bisect_plane(bm, geom=geom, dist=1e-7, plane_co=Vector(co), plane_no=Vector(no))


def _car_build_parts(obj, plan, geo, mats, params, made):
    """Edit the body in place and make the part objects. Returns the reply's parts block."""
    from .ops_create import op_create_mesh   # noqa: PLC0415
    x0, x1 = geo["x0"], geo["x1"]
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.normal_update()
    out = {k: [] for k in _CAR_PARTS}
    parts = _Parts()
    base = obj.name

    # BUMPERS first, from the clean body: its faces in the band, offset out by gap and given thickness
    for b in plan["bumpers"]:
        b2 = bm.copy()
        xe = x0 + b["length"] if b["end"] == "FRONT" else x1 - b["length"]
        allg = list(b2.faces) + list(b2.edges) + list(b2.verts)
        bmesh.ops.bisect_plane(b2, geom=allg, dist=1e-7, plane_co=Vector((xe, 0, 0)), plane_no=Vector((1, 0, 0)))
        for z in (b["zMin"], b["zMax"]):
            allg = list(b2.faces) + list(b2.edges) + list(b2.verts)
            bmesh.ops.bisect_plane(b2, geom=allg, dist=1e-7, plane_co=Vector((0, 0, z)), plane_no=Vector((0, 0, 1)))

        def keep(f, b=b, xe=xe):
            c = f.calc_center_median()
            inx = c.x <= xe if b["end"] == "FRONT" else c.x >= xe
            return inx and b["zMin"] <= c.z <= b["zMax"]
        bmesh.ops.delete(b2, geom=[f for f in b2.faces if not keep(f)], context="FACES")
        b2.normal_update()
        idx = {v: k for k, v in enumerate(b2.verts)}
        pv = [list(v.co) for v in b2.verts]
        nv = [list(v.normal) for v in b2.verts]
        pf = [[idx[v] for v in f.verts] for f in b2.faces]
        b2.free()
        if not pf:
            raise MifOpError("the %s bumper's band holds no body face." % b["end"].lower())
        v, f = _shell(pv, pf, nv, b["gap"] + b["thickness"], b["gap"])
        name = "%s_Bumper_%s" % (base, b["end"].title())
        parts.add(name, v, f, "trim")
        bvh = BVHTree.FromBMesh(bm)
        gaps = [bvh.find_nearest(Vector(p))[3] for p in v[len(pv):]]
        out["bumpers"].append({"object": name, "end": b["end"], "faces": len(f),
                               "gapMeasured": [round(min(gaps), 6), round(max(gaps), 6)]})

    # GROOVES: shut lines round each lid on the top, seams down the sides - three cuts, the middle one pressed in
    def top_face(f):
        return f.normal.z > 0.7

    def top_vert(v):
        return any(f.normal.z > 0.7 for f in v.link_faces)
    press = []          # (test(vert) -> bool, depth)
    for lid in plan["lids"]:
        yc, w2 = lid["yc"], lid["width"] / 2
        for xq in (lid["x0"], lid["x1"]):
            for off in (-w2, 0.0, w2):
                _car_cut(bm, top_face, (xq + off, 0, 0), (1, 0, 0))
            press.append((lambda v, xq=xq, yc=yc: abs(v.co.x - xq) < 1e-5 and abs(v.co.y) <= yc + 1e-5
                          and top_vert(v), lid["depth"]))
        for s in (1, -1):
            for off in (-w2, 0.0, w2):
                _car_cut(bm, lambda f, lid=lid: top_face(f) and lid["x0"] - 0.2 < f.calc_center_median().x
                         < lid["x1"] + 0.2, (0, s * yc + off, 0), (0, 1, 0))
            press.append((lambda v, s=s, lid=lid, yc=yc: abs(v.co.y - s * yc) < 1e-5
                          and lid["x0"] - 1e-5 <= v.co.x <= lid["x1"] + 1e-5 and top_vert(v), lid["depth"]))
    for sm in plan["seams"]:
        signs = {"BOTH": (1, -1), "LEFT": (1,), "RIGHT": (-1,)}[sm["side"]]
        w2 = sm["width"] / 2

        def side_face(f, sm=sm, signs=signs):
            c = f.calc_center_median()
            return any(f.normal.y * s > 0.35 for s in signs) and sm["zMin"] - 0.15 < c.z < sm["zMax"] + 0.15
        for off in (-w2, 0.0, w2):
            _car_cut(bm, side_face, (sm["x"] + off, 0, 0), (1, 0, 0))
        press.append((lambda v, sm=sm, signs=signs: abs(v.co.x - sm["x"]) < 1e-5
                      and sm["zMin"] - 1e-5 <= v.co.z <= sm["zMax"] + 1e-5
                      and any(v.co.y * s > 0 and any(f.normal.y * s > 0.35 for f in v.link_faces) for s in signs),
                      sm["depth"]))
    if press:
        bm.normal_update()
        moves = {}
        for k, (test, depth) in enumerate(press):
            hit = [v for v in bm.verts if test(v)]
            if not hit:
                raise MifOpError("a groove pressed no vertex (groove %d)." % k)
            for v in hit:
                moves.setdefault(v, (v.normal.copy(), depth))
            row = (out["lids"] if k < 4 * len(plan["lids"]) else out["seams"])
            row.append(len(hit))
        for v, (nrm, depth) in moves.items():
            v.co -= nrm * depth
        lids_n, seams_n = out["lids"], out["seams"]
        out["lids"] = [{"x0": lid["x0"], "x1": lid["x1"], "halfWidth": round(lid["yc"], 6), "depth": lid["depth"],
                        "pressedVerts": sum(lids_n[4 * i:4 * i + 4])} for i, lid in enumerate(plan["lids"])]
        out["seams"] = [{"x": sm["x"], "zMin": sm["zMin"], "zMax": sm["zMax"], "side": sm["side"],
                         "depth": sm["depth"], "pressedVerts": n} for sm, n in zip(plan["seams"], seams_n)]
    bm.normal_update()

    # POCKETS: four cuts round the rectangle on the end cap or the side, the faces inside pushed in by depth
    for p in plan["pockets"]:
        sw, sh = p["size"]
        cz = p["center"][1]
        row = {"kind": p["kind"], "at": p["at"], "center": list(p["center"]), "size": [sw, sh], "depth": p["depth"],
               "instances": []}
        for inst in p["instances"]:
            u0, u1, z0, z1 = inst["u"] - sw / 2, inst["u"] + sw / 2, cz - sh / 2, cz + sh / 2
            if "plane" in inst:
                xe = inst["plane"]
                inward = Vector((1, 0, 0)) if inst["side"] == "FRONT" else Vector((-1, 0, 0))

                def on(f, xe=xe):
                    return all(abs(v.co.x - xe) < 1e-6 for v in f.verts)

                def inside(c, u0=u0, u1=u1, z0=z0, z1=z1):
                    return u0 < c.y < u1 and z0 < c.z < z1
                for co, no in (((0, u0, 0), (0, 1, 0)), ((0, u1, 0), (0, 1, 0)), ((0, 0, z0), (0, 0, 1)),
                               ((0, 0, z1), (0, 0, 1))):
                    _car_cut(bm, on, co, no)
                region = [f for f in bm.faces if on(f) and inside(f.calc_center_median())]
            else:
                s = inst["sign"]
                inward = Vector((0, -s, 0))

                def on(f, s=s, u0=u0, u1=u1, z0=z0, z1=z1):
                    c = f.calc_center_median()
                    return f.normal.y * s > 0.3 and u0 - 0.3 < c.x < u1 + 0.3 and z0 - 0.3 < c.z < z1 + 0.3

                def inside(c, u0=u0, u1=u1, z0=z0, z1=z1):
                    return u0 < c.x < u1 and z0 < c.z < z1
                for co, no in (((u0, 0, 0), (1, 0, 0)), ((u1, 0, 0), (1, 0, 0)), ((0, 0, z0), (0, 0, 1)),
                               ((0, 0, z1), (0, 0, 1))):
                    _car_cut(bm, on, co, no)
                region = [f for f in bm.faces if on(f) and inside(f.calc_center_median())]
            area = sum(f.calc_area() for f in region)
            if not region or abs(area - sw * sh) > 0.05 * sw * sh + 1e-6:
                raise MifOpError("the %s pocket at %s did not cut cleanly: %d face(s), %.6g of %.6g m2 inside its "
                                 "rectangle." % (p["kind"], inst["side"].lower(), len(region), area, sw * sh))
            bm.normal_update()
            rverts = list({v for f in region for v in f.verts})
            ridx = {v: k for k, v in enumerate(rverts)}
            rco = [list(v.co) for v in rverts]
            rnorm = []
            for v in rverts:
                nn = Vector((0, 0, 0))
                for f in v.link_faces:
                    if f in region:
                        nn += f.normal
                rnorm.append(list(nn.normalized()))
            rfaces = [[ridx[v] for v in f.verts] for f in region]
            ret = bmesh.ops.extrude_face_region(bm, geom=region)
            new_verts = [e for e in ret["geom"] if isinstance(e, bmesh.types.BMVert)]
            bmesh.ops.delete(bm, geom=[f for f in region if f.is_valid], context="FACES")
            for v in new_verts:
                v.co += inward * p["depth"]
            bm.normal_update()
            irow = {"side": inst["side"], "u": round(inst["u"], 6), "faces": len(region), "objects": []}
            out_n = -inward
            depth = p["depth"]
            if p["lens"]:
                v, f = _shell(rco, rfaces, rnorm, 0.0, -p["lensThickness"])
                parts.add("%s_Lenses" % base, v, f, "lens")
                irow["objects"].append("%s_Lenses" % base)
            if "plane" in inst:
                fl = inst["plane"] + inward.x * depth           # the pocket's floor, on x

                def slab(u_a, u_b, z_a, z_b, d_in, d_out, fl=fl, out_n=out_n):
                    xa, xb = sorted((fl + out_n.x * d_in, fl + out_n.x * d_out))
                    return _box_lohi((xa, u_a, z_a), (xb, u_b, z_b))
                if p["band"]:
                    v, f = slab(inst["u"] - sw * 0.4, inst["u"] + sw * 0.4, cz - sh * 0.12, cz + sh * 0.12, 0, 0.004)
                    parts.add("%s_LampBands" % base, v, f, "lamp")
                    irow["objects"].append("%s_LampBands" % base)
                if p["bowls"]:
                    # a reflector bowl on the floor beside the band and a bulb in it, behind the lens
                    rb = 0.42 * min(sw * 0.25, sh)
                    db = min(rb * 0.8, (depth - p["lensThickness"]) * 0.6)
                    for du in (-0.3, 0.3):
                        ctr = (fl, inst["u"] + du * sw, cz)
                        v, f = _revolve([(0, 0), (rb * 0.5, db * 0.3), (rb, db), (rb * 0.94, db),
                                         (rb * 0.47, db * 0.3 + 0.0015), (0, 0.0015)], ctr, out_n, 16)
                        parts.add("%s_LampBowls" % base, v, f, "trim")
                        rbulb = rb * 0.25
                        prof = [(rbulb * math.sin(math.pi * k / 8), rbulb * (1 - math.cos(math.pi * k / 8)))
                                for k in range(9)]
                        v, f = _revolve([(round(r, 9), a + db * 0.25) for r, a in prof], ctr, out_n, 12)
                        parts.add("%s_LampBulbs" % base, v, f, "lamp")
                    irow["objects"] += ["%s_LampBowls" % base, "%s_LampBulbs" % base]
                if p["kind"] == "GRILLE":
                    n = p["slats"]
                    for k in range(n):
                        zc = z0 + sh * (k + 0.5) / n
                        v, f = slab(u0, u1, zc - sh / (4 * n), zc + sh / (4 * n), depth * 0.3, depth - 0.005)
                        parts.add("%s_Grille" % base, v, f, "trim")
                    irow["objects"].append("%s_Grille" % base)
                if p["kind"] == "PLATE":
                    v, f = slab(u0 + sw * 0.02, u1 - sw * 0.02, z0 + sh * 0.04, z1 - sh * 0.04, 0, 0.003)
                    parts.add("%s_Plate" % base, v, f, "trim")
                    irow["objects"].append("%s_Plate" % base)
                    if p["text"]:
                        cv, cf, nchar = _plate_chars(p["text"], p["charHeight"], inst, cz, fl, out_n, p["raise"], sw)
                        parts.add("%s_PlateChars" % base, cv, cf, "trim")
                        irow["objects"].append("%s_PlateChars" % base)
                        irow["characters"] = nchar
                        irow["charsWidth"] = round(max(q[1] for q in cv) - min(q[1] for q in cv), 6)
                        irow["charsHeight"] = round(max(q[2] for q in cv) - min(q[2] for q in cv), 6)
            elif p["kind"] == "HANDLE":
                s = inst["sign"]
                ys = max((c[1] * s for c in rco), default=0.0)
                yf = s * (ys - depth)
                v, f = _box_lohi((inst["u"] - sw * 0.35, min(yf, yf + s * depth * 0.8), cz - sh * 0.18),
                            (inst["u"] + sw * 0.35, max(yf, yf + s * depth * 0.8), cz + sh * 0.18))
                parts.add("%s_Handles" % base, v, f, "trim")
                irow["objects"].append("%s_Handles" % base)
            row["instances"].append(irow)
        out["pockets"].append(row)

    bm.to_mesh(me)
    me.update()
    bvh = BVHTree.FromBMesh(bm)
    span = max(abs(v.co.y) for v in bm.verts) + 1.0
    bm.free()

    def side_y(x, z, s):
        hit = bvh.ray_cast(Vector((x, s * span, z)), Vector((0, -s, 0)))
        return None if hit[0] is None else abs(hit[0].y)

    # MIRRORS: a housing out from the body side at reach, an arm back to it; the housing's rear face on the glass slot
    for m in plan["mirrors"]:
        lx, ly, lz = m["size"]
        for s in (1, -1):
            ys = side_y(m["x"], m["z"], s)
            if ys is None:
                raise MifOpError("the mirror at x %g, z %g finds no body side to stand on." % (m["x"], m["z"]))
            a, b = sorted((s * (ys + m["reach"]), s * (ys + m["reach"] + ly)))
            hv, hf = _box_lohi((m["x"] - lx / 2, a, m["z"] - lz / 2), (m["x"] + lx / 2, b, m["z"] + lz / 2))
            parts.add("%s_Mirrors" % base, hv, hf, None, ["trim", "glass", "trim", "trim", "trim", "trim"])
            a2, b2 = sorted((s * (ys - 0.01), s * (ys + m["reach"] + 0.005)))
            av, af = _box_lohi((m["x"] - 0.02, a2, m["z"] - lz * 0.35), (m["x"] + 0.02, b2, m["z"] - lz * 0.1))
            parts.add("%s_Mirrors" % base, av, af, "trim")
        out["mirrors"].append({"object": "%s_Mirrors" % base, "x": m["x"], "z": m["z"], "sides": 2})

    # ARCH LIPS: a flat band round each arch's edge on each side, standing out from the side by out
    for lip in plan["archLips"]:
        n = lip["segments"]
        for xc, r, h in geo["arches"]:
            zb0 = geo["zb0"](xc)
            ta = math.asin(max(-1.0, min(1.0, (zb0 - h) / r)))
            for s in (1, -1):
                pv, pn = [], []
                angles = [ta + (math.pi - 2 * ta) * k / n for k in range(n + 1)]
                rm = r + lip["width"] / 2
                found = [side_y(xc + rm * math.cos(t), h + rm * math.sin(t), s) for t in angles]
                if not any(y is not None for y in found):
                    raise MifOpError("the arch lip at x %g finds no body side." % xc)
                # an end sample can fall just under the faceted bottom line: it takes its nearest neighbour's side
                for k, t in enumerate(angles):
                    ys = found[k] if found[k] is not None else next(
                        found[j] for j in sorted(range(n + 1), key=lambda j: abs(j - k)) if found[j] is not None)
                    for rr in (r, r + lip["width"]):
                        pv.append([xc + rr * math.cos(t), s * ys, h + rr * math.sin(t)])
                        pn.append([0, s, 0])
                pf = []
                for k in range(n):
                    q = [2 * k, 2 * k + 1, 2 * k + 3, 2 * k + 2]
                    a, b, c = (Vector(pv[i]) for i in q[:3])
                    if (b - a).cross(c - a).y * s < 0:
                        q.reverse()
                    pf.append(q)
                v, f = _shell(pv, pf, pn, lip["out"], -lip["embed"])
                parts.add("%s_ArchLips" % base, v, f, "body")
        out["archLips"].append({"object": "%s_ArchLips" % base, "arches": len(geo["arches"]), "width": lip["width"],
                                "out": lip["out"]})

    # every gathered part becomes one object, parented to the body in its local space
    slot_keys = [k for k in ("body", "trim", "glass", "lens", "lamp") if mats.get(k)]
    objects = []
    for name, (v, f, mk) in parts.objs.items():
        if _signed_volume(v, f) < 0:
            f = [list(reversed(face)) for face in f]
        sub = {"name": name, "vertices": v, "faces": f, "uvs": {"UVMap": _face_uvs(v, f)}}
        used = [k for k in slot_keys if k in set(mk)]
        if used:
            sub["materials"] = [mats[k] for k in used]
            sub["materialIndices"] = [used.index(k) if k in used else 0 for k in mk]
        if params.get("collection") is not None:
            sub["collection"] = params["collection"]
        res = op_create_mesh(sub)
        po = bpy.data.objects[res["name"]]
        made.append(po)
        po.parent = obj
        po.matrix_parent_inverse = Matrix.Identity(4)
        objects.append(po.name)
    # WHEELS through build_lathe: a tyre and rim profile turned round Y, standing on the ground in each arch
    for wh in plan["wheels"]:
        hw = wh["width"] / 2
        for k, row in enumerate(wh["rows"]):
            R, rim = row["radius"], row["rim"]
            prof = [[0, -hw * 0.55], [rim, -hw * 0.55], [rim + 0.015, -hw], [R - 0.03, -hw], [R, -hw + 0.03],
                    [R, hw - 0.03], [R - 0.03, hw], [rim + 0.015, hw], [rim, hw * 0.55], [0, hw * 0.55]]
            for s, tag in ((1, "L"), (-1, "R")):
                sub = {"name": "%s_Wheel%d%s" % (base, k + 1, tag), "profile": prof, "axis": "Y",
                       "segments": wh["segments"], "location": [row["x"], s * row["y"], R]}
                if mats.get("tyre"):
                    sub["material"] = mats["tyre"]
                if params.get("collection") is not None:
                    sub["collection"] = params["collection"]
                res = op_build_lathe(sub)
                po = bpy.data.objects[res["name"]]
                made.append(po)
                po.parent = obj
                po.matrix_parent_inverse = Matrix.Identity(4)
                objects.append(po.name)
                out["wheels"].append({"object": po.name, "x": row["x"], "y": round(s * row["y"], 6), "radius": R,
                                      "rimRadius": rim, "width": wh["width"]})
    out["objects"] = objects
    return {k: v for k, v in out.items() if v}


def _plate_chars(text, height, inst, cz, floor_x, out_n, rise, plate_w):
    """Raised characters from Blender's built-in font, mapped onto the plate face. (verts, faces, glyph count)"""
    cu = bpy.data.curves.new("MifPlateText", "FONT")
    cu.body = text
    cu.extrude = 0.05
    tmp = bpy.data.objects.new("MifPlateText", cu)
    bpy.context.scene.collection.objects.link(tmp)
    try:
        dg = bpy.context.evaluated_depsgraph_get()
        dg.update()
        mesh = bpy.data.meshes.new_from_object(tmp.evaluated_get(dg))
        try:
            # the font's caps and walls come out unwelded - welded here, so each glyph is closed
            bmc = bmesh.new()
            bmc.from_mesh(mesh)
            bmesh.ops.remove_doubles(bmc, verts=list(bmc.verts), dist=1e-5)
            bmc.verts.index_update()
            fv = [tuple(v.co) for v in bmc.verts]
            ff = [[v.index for v in f.verts] for f in bmc.faces]
            bmc.free()
        finally:
            bpy.data.meshes.remove(mesh)
    finally:
        remove_objects([tmp])
        bpy.data.curves.remove(cu)
    if not ff:
        raise MifOpError("the plate text %r made no glyph geometry." % text)
    xs, ys, zs = zip(*fv)
    sc = height / (max(ys) - min(ys))
    if (max(xs) - min(xs)) * sc > plate_w * 0.92:
        sc = plate_w * 0.92 / (max(xs) - min(xs))
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    z_lo, z_hi = min(zs), max(zs)
    right = -1.0 if out_n.x < 0 else 1.0            # the viewer's right: -y in front, +y behind
    face = floor_x + out_n.x * 0.003
    verts = []
    for x, y, z in fv:
        t = (z - z_lo) / (z_hi - z_lo) if z_hi > z_lo else 0.0
        verts.append([face + out_n.x * rise * t, inst["u"] + right * (x - cx) * sc, cz + (y - cy) * sc])
    glyphs = len([c for c in text if not c.isspace()])
    return verts, ff, glyphs


# =============================================================================================================
# TRASH BAG (2026-10-06, spec "bl_build_trash_bag: a tied rubbish bag")
# =============================================================================================================
_BAG_KEYS = ("name", "height", "width", "contact", "seed", "knuckles", "knuckleDepth", "pleats", "twist", "ears",
             "segments", "rings", "color", "material", "location", "rotation", "collection", "takeOverName")
# THE PEAR, as radius over the widest radius against height over the body's height: the contact patch's edge at the
# foot, the belly at 45 %, the shoulder drawn in to the neck
_BAG_PROFILE = ((0.0, None), (0.06, 0.80), (0.18, 0.95), (0.45, 1.0), (0.68, 0.84), (0.85, 0.50), (1.0, 0.0))
_BAG_BODY, _BAG_NECK = 0.84, 0.07            # shares of the height to the knot's top: the body, then the neck
_BAG_NECK_R, _BAG_KNOT_R = 0.045, 0.085       # neck and knot radius as shares of the width


def _smooth_curve(points, x):
    """Catmull-Rom through (x, y) points, x in order; flat beyond the ends."""
    xs = [p[0] for p in points]
    if x <= xs[0]:
        return points[0][1]
    if x >= xs[-1]:
        return points[-1][1]
    i = max(k for k in range(len(xs) - 1) if xs[k] <= x)
    p0 = points[max(i - 1, 0)][1]
    p1, p2 = points[i][1], points[i + 1][1]
    p3 = points[min(i + 2, len(points) - 1)][1]
    t = (x - xs[i]) / (xs[i + 1] - xs[i])
    return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3)


def _ring_surface(rings, segs):
    """Faces and UVs for a closed surface of revolution-like rings: rings[0] and rings[-1] are poles (one vertex),
    every other ring `segs` vertices. Quads between rings, a triangle fan at each pole."""
    faces, uvs = [], []
    n = len(rings)
    for j in range(n - 1):
        a, b = rings[j], rings[j + 1]
        v0, v1 = j / (n - 1), (j + 1) / (n - 1)
        for k in range(segs):
            k1 = (k + 1) % segs
            u0, u1 = k / segs, (k + 1) / segs
            if len(a) == 1:
                faces.append([a[0], b[k1], b[k]])
                uvs.append([[(u0 + u1) / 2, v0], [u1, v1], [u0, v1]])
            elif len(b) == 1:
                faces.append([a[k], a[k1], b[0]])
                uvs.append([[u0, v0], [u1, v0], [(u0 + u1) / 2, v1]])
            else:
                faces.append([a[k], a[k1], b[k1], b[k]])
                uvs.append([[u0, v0], [u1, v0], [u1, v1], [u0, v1]])
    return faces, uvs


def _bag_mesh(height, width, contact, seed, knuckles, depth, pleats, twist, ears, segs, nrings):
    """The bag's vertices, faces and UVs (metres, standing on z 0 round the Z axis), each part turned out by its
    own signed volume. build_trash_bag makes one; build_dumpster's fill places several."""
    rnd_ = random.Random(seed)
    big_r = width / 2.0
    hb = _BAG_BODY * height
    rn, rk = _BAG_NECK_R * width, _BAG_KNOT_R * width
    neck_top = hb + _BAG_NECK * height
    # the knot: a ball of radius rk whose lower cap meets the neck's radius at the neck's top and whose top is height
    phi0 = math.asin(min(1.0, rn / rk))
    rk_fit = (height - neck_top) / (1.0 + math.cos(phi0))
    rk = min(rk, rk_fit)
    phi0 = math.asin(min(1.0, rn / rk)) if rk > rn else math.pi / 2.0
    zk = height - rk
    prof = [(u, (contact if r is None else r)) for u, r in _BAG_PROFILE]
    prof[-1] = (1.0, rn / big_r)
    bumps = [(rnd_.uniform(0, 2 * math.pi), rnd_.uniform(0.15, 0.68), rnd_.uniform(0.6, 1.0)) for _ in range(knuckles)]
    twist_rad = 2 * math.pi * twist
    pleat_phase = rnd_.uniform(0, 2 * math.pi)

    # ---- the body's rings: pole, the flat patch, the pear, the neck, the knot, the top pole
    stations = [("pole", 0.0, 0.0), ("patch", 0.0, 0.5 * contact * big_r), ("patch", 0.0, contact * big_r)]
    nb = max(8, int(nrings * 0.62))
    for j in range(1, nb + 1):
        u = j / nb
        stations.append(("body", u * hb, max(rn, _smooth_curve(prof, u) * big_r)))
    nn = max(3, int(nrings * 0.15))
    for j in range(1, nn + 1):
        stations.append(("neck", hb + (neck_top - hb) * j / nn, rn))
    nk = max(4, nrings - nb - nn - 3)
    for j in range(1, nk):
        phi = phi0 + (math.pi - phi0) * j / nk
        stations.append(("knot", zk - rk * math.cos(phi), rk * math.sin(phi)))
    stations.append(("pole", height, 0.0))

    def wave(z):
        """Pleat strength and twist angle at height z: nothing below 55 % of the body, growing to the neck."""
        s = max(0.0, min(1.0, (z - 0.55 * hb) / (neck_top - 0.55 * hb)))
        s = s * s * (3 - 2 * s)
        turn = twist_rad * max(0.0, min(1.0, (z - 0.55 * hb) / (neck_top - 0.55 * hb))) ** 1.5
        return s, turn

    verts, rings = [], []
    for kind, z, r0 in stations:
        if kind == "pole":
            rings.append([len(verts)])
            verts.append([0.0, 0.0, z])
            continue
        ring = []
        s, turn = wave(z) if kind in ("body", "neck") else (0.0, 0.0)
        for k in range(segs):
            th = 2 * math.pi * k / segs
            r = r0
            if kind == "body" and z > 0.0:
                u = z / hb
                for bt, bu, amp in bumps:
                    dt = math.atan2(math.sin(th - bt), math.cos(th - bt))
                    r += big_r * depth * amp * math.exp(-(dt / 0.42) ** 2 - ((u - bu) / 0.09) ** 2)
            if pleats and s > 0.0:
                fold = 0.5 + 0.5 * math.cos(pleats * th + turn + pleat_phase)
                r *= 1.0 - (0.24 if kind == "body" else 0.32) * s * fold
            ring.append(len(verts))
            verts.append([r * math.cos(th), r * math.sin(th), z])
        rings.append(ring)
    faces, uvs = _ring_surface(rings, segs)
    parts = [("body", 0, len(faces))]

    # ---- the ears: flat closed lobes leaning out from the knot's top, clear of it
    if ears:
        el = 0.2 * height
        lean = math.radians(rnd_.uniform(55.0, 70.0))
        spin = rnd_.uniform(0, math.pi)
        es, er = 12, 9
        for side in (1.0, -1.0):
            d = Vector((side * math.sin(lean) * math.cos(spin), side * math.sin(lean) * math.sin(spin), math.cos(lean)))
            side_ax = Vector((0, 0, 1)).cross(d).normalized()
            flat_ax = d.cross(side_ax).normalized()
            base = Vector((0, 0, height)) + d * (0.004 + 0.02 * rk)
            ear_rings = []
            for j in range(er + 1):
                t = j / er
                c = base + d * (el * t)
                if j in (0, er):
                    ear_rings.append([len(verts)])
                    verts.append(list(c))
                    continue
                w = 0.32 * el * math.sin(math.pi * t) ** 0.8
                th_ = 0.05 * el * math.sin(math.pi * t) ** 0.8
                ring = []
                for k in range(es):
                    a = 2 * math.pi * k / es
                    ring.append(len(verts))
                    verts.append(list(c + side_ax * (w * math.cos(a)) + flat_ax * (th_ * math.sin(a))))
                ear_rings.append(ring)
            f, uv = _ring_surface(ear_rings, es)
            parts.append(("ear", len(faces), len(faces) + len(f)))
            faces += f
            uvs += uv

    # ---- each part turned out by its own signed volume
    checks = []
    for pname, f0, f1 in parts:
        vol = _signed_volume(verts, faces[f0:f1])
        turned = vol < 0
        if turned:
            for i in range(f0, f1):
                faces[i] = list(reversed(faces[i]))
                uvs[i] = list(reversed(uvs[i]))
        checks.append({"part": pname, "signedVolume": round(abs(vol), 9), "turnedOver": turned})
    return {"verts": verts, "faces": faces, "uvs": uvs, "parts": parts, "checks": checks,
            "bodyVerts": sum(len(r) for r in rings), "rn": rn, "rk": rk}


def op_build_trash_bag(params):
    """Build a tied rubbish bag standing on the floor: a pear on a flat contact patch, lumped by its contents, pleats
    gathering and twisting into a twisted neck, a knot and two ears - one mesh, every part a closed solid facing out.

    THE GAP (2026-10-04): a filled bin's bags were built by a local script (a pear profile, content knuckles, pleats
    converging into the neck, a knot and ears) and sent through create_mesh - and the loose bags stayed lumpy
    icospheres. The rules the script followed are built in: the body is one ring surface whose every ring is a closed
    star round the axis at its own height, so no displacement can fold it through itself; the contact patch is flat
    at z 0, so the bag stands on the floor instead of on a point; the ears are separate closed shells set clear of
    the knot. Each part is turned out by its own signed volume (outwardCheck).

    params:
      height (m)        to the top of the knot, default 0.7; 0.1-3
      width (m)         the belly's diameter before the contents lump it, default 0.48; 0.05-2
      contact           the flat contact patch's radius as a share of the belly's, default 0.45; 0.1-0.8
      seed (int)        the same seed is the same bag, default 1
      knuckles (int)    lumps the contents push out of the belly, default 8; 0-32
      knuckleDepth      their height as a share of the radius, default 0.1; 0-0.3
      pleats (int)      folds gathering into the neck, default 9; 0, or 3-32
      twist (turns)     how far the pleats turn on their way up the neck, default 1; 0-5
      ears (bool)       the two tied ears over the knot, default true
      segments (int)    round the bag, default 36 (8-256); rings (int) up it, default 32 (12-200)
      color ([r,g,b])   the made plastic's color, linear 0-1, default a near-black; or material, an existing one
      name, location, rotation, collection, takeOverName   create_mesh's
    Refused before anything exists: a value outside its range, color with material, a material that is not there.
    """
    tail = "NOTHING was created."
    reject_unknown(params, _BAG_KEYS, "build_trash_bag")
    height = take_float(params, "height", default=0.7)
    width = take_float(params, "width", default=0.48)
    contact = take_float(params, "contact", default=0.45)
    seed = take_int(params, "seed", default=1)
    knuckles = take_int(params, "knuckles", default=8)
    depth = take_float(params, "knuckleDepth", default=0.1)
    pleats = take_int(params, "pleats", default=9)
    twist = take_float(params, "twist", default=1.0)
    ears = take_bool(params, "ears", default=True)
    segs = take_int(params, "segments", default=36)
    nrings = take_int(params, "rings", default=32)
    for key, val, lo, hi in (("height", height, 0.1, 3.0), ("width", width, 0.05, 2.0), ("contact", contact, 0.1, 0.8),
                             ("knuckles", knuckles, 0, 32), ("knuckleDepth", depth, 0.0, 0.3), ("twist", twist, 0.0, 5.0),
                             ("segments", segs, 8, 256), ("rings", nrings, 12, 200)):
        if not lo <= val <= hi:
            raise MifOpError("%s must be %g to %g, got %g. %s" % (key, lo, hi, val, tail))
    if pleats != 0 and not 3 <= pleats <= 32:
        raise MifOpError("pleats must be 0 or 3 to 32, got %d. %s" % (pleats, tail))
    if width > 1.5 * height:
        raise MifOpError("a bag %g m wide and %g m tall is wider than one and a half times its height - that is a "
                         "sack lying down, not a tied bag standing. %s" % (width, height, tail))
    mat_name = take(params, "material", kind=str)
    if mat_name is not None and params.get("color") is not None:
        raise MifOpError("'material' reuses an existing material and 'color' colors the one made here; pass one. %s"
                         % tail)
    if mat_name is not None and bpy.data.materials.get(mat_name) is None:
        raise MifOpError("no material named %r. %s" % (mat_name, tail))
    color = _color3(params, "color", (0.018, 0.018, 0.02))

    g = _bag_mesh(height, width, contact, seed, knuckles, depth, pleats, twist, ears, segs, nrings)
    verts, faces, uvs, parts, checks, rn, rk = (g[k] for k in ("verts", "faces", "uvs", "parts", "checks", "rn", "rk"))

    sub = {k: params[k] for k in ("name", "location", "rotation", "collection", "takeOverName") if k in params}
    sub.setdefault("name", "TrashBag")
    out = _build_through_create_mesh(sub, verts, faces, uvs, True, "TrashBag", "build_trash_bag")
    obj = bpy.data.objects.get(out.get("name") or "")
    if obj is not None and obj.type == "MESH":
        mat = bpy.data.materials.get(mat_name) if mat_name else _bag_material("M_%s" % obj.name, color)
        obj.data.materials.append(mat)
        out["materialSlots"] = [m.name for m in obj.data.materials if m]
        out["materialMade"] = mat_name is None
    floor = [Vector(v).xy.length for v in verts if abs(v[2]) < 1e-9]
    body_verts = verts[:g["bodyVerts"]]
    out.update({
        "seed": seed, "parts": [p[0] for p in parts],
        "bag": {"heightM": round(height, 6), "knotTopM": round(max(v[2] for v in body_verts), 6),
                "earsTopM": round(max(v[2] for v in verts), 6) if ears else None,
                "bellyWidthM": round(2.0 * max(Vector(v).xy.length for v in body_verts), 6),
                "contactRadiusM": round(max(floor), 6), "neckRadiusM": round(rn, 6), "knotRadiusM": round(rk, 6),
                "knuckles": knuckles, "pleats": pleats, "twistTurns": twist,
                "measuredBy": "the built vertices: the contact patch is every vertex at z 0"},
        "outwardCheck": {"parts": checks, "note": "each part turned so its signed volume is positive"},
    })
    return out


def _bag_material(name, color):
    """A glossy plastic film: dark, smooth, a little sheen."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = _find_type(mat.node_tree.nodes, "BSDF_PRINCIPLED")
    _sock(bsdf.inputs, "Base Color").default_value = tuple(color) + (1.0,)
    _sock(bsdf.inputs, "Roughness").default_value = 0.32
    return mat


# =============================================================================================================
# DOWNSPOUT, WALL FIXTURE, DUMPSTER (2026-10-06, spec "Prop builders learned from the alley")
# =============================================================================================================
# All three build in metres from RINGS: a part is a chain of rings with one point count, quads between neighbours,
# and either closed into a loop (a hollow tube, a channel round a mouth) or capped at both ends by a ladder of quads
# between the ring's two halves - never an n-gon, never a fan. Each part is turned out by its own signed volume.

def _ladder(ring, flip):
    n = len(ring)
    out = []
    for k in range(n // 2 - 1):
        q = [ring[k], ring[k + 1], ring[n - 2 - k], ring[n - 1 - k]]
        out.append(q[::-1] if flip else q)
    return out


class _PropParts:
    """Parts of one prop mesh: vertices, faces, a material slot and a name per face, each part wound one way and
    turned out by its signed volume."""

    def __init__(self):
        self.verts, self.faces, self.slots, self.names, self.checks = [], [], [], [], []

    def _orient(self, name, f0):
        vol = _signed_volume(self.verts, self.faces[f0:])
        if vol < 0:
            for i in range(f0, len(self.faces)):
                self.faces[i] = self.faces[i][::-1]
        self.checks.append({"part": name, "signedVolume": round(abs(vol), 9), "turnedOver": vol < 0})

    def chain(self, name, slot, rings, closed=False, fan_caps=False):
        """rings: lists of [x, y, z], every ring the same even count. fan_caps closes the two ends with fans from their
        middle, for rings with unevenly spaced points: a ladder folds onto a straight side wherever its last rung lands
        part way along one, and those are faces of no area."""
        f0, idx = len(self.faces), []
        for ring in rings:
            idx.append(list(range(len(self.verts), len(self.verts) + len(ring))))
            self.verts.extend([list(p) for p in ring])
        m = len(idx)
        for j in range(m if closed else m - 1):
            a, b = idx[j], idx[(j + 1) % m]
            n = len(a)
            for k in range(n):
                k1 = (k + 1) % n
                self.faces.append([a[k], a[k1], b[k1], b[k]])
        if not closed and fan_caps:
            for ring, flip in ((idx[0], True), (idx[-1], False)):
                mid = len(self.verts)
                self.verts.append([sum(self.verts[i][k] for i in ring) / len(ring) for k in range(3)])
                for k, i in enumerate(ring):
                    j = ring[(k + 1) % len(ring)]
                    self.faces.append([mid, j, i] if flip else [mid, i, j])
        elif not closed:
            self.faces += _ladder(idx[0], True) + _ladder(idx[-1], False)
        self.slots += [slot] * (len(self.faces) - f0)
        self.names.append(name)
        self._orient(name, f0)

    def raw(self, name, slot, verts, faces):
        f0, base = len(self.faces), len(self.verts)
        self.verts.extend([list(v) for v in verts])
        self.faces.extend([[base + i for i in f] for f in faces])
        self.slots += [slot] * (len(self.faces) - f0)
        self.names.append(name)
        self._orient(name, f0)

    def box(self, name, slot, x0, x1, y0, y1, z0, z1):
        self.chain(name, slot, [[(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)] for z in (z0, z1)])

    def bar(self, name, slot, p0, p1, radius, sides=12, ref=None):
        """A round bar from p0 to p1."""
        p0, p1 = Vector(p0), Vector(p1)
        t = (p1 - p0).normalized()
        ref = Vector(ref) if ref else (Vector((0, 0, 1)) if abs(t.z) < 0.9 else Vector((1, 0, 0)))
        n = (ref - t * ref.dot(t)).normalized()
        b = t.cross(n)
        ring = [n * (radius * math.cos(2 * math.pi * k / sides)) + b * (radius * math.sin(2 * math.pi * k / sides))
                for k in range(sides)]
        self.chain(name, slot, [[p + o for o in ring] for p in (p0, p1)])

    def sheet(self, name, slot, part):
        """A part made by _sheet or _cyl (its "mesh" lists)."""
        verts, faces = part["mesh"]
        self.raw(name, slot, verts, faces)


def _box_uvs_lists(verts, faces, tile=1.0):
    """UV0 per face corner as a box projection in metres, the way build_kit_piece maps its pieces."""
    uvs = []
    for f in faces:
        a, b, c = (Vector(verts[i]) for i in f[:3])
        nrm = (b - a).cross(c - a)
        ax = max(range(3), key=lambda k: abs(nrm[k]))
        u, v = {0: (1, 2), 1: (0, 2), 2: (0, 1)}[ax]
        uvs.append([[verts[i][u] / tile, verts[i][v] / tile] for i in f])
    return uvs


def _path_frames(pts):
    """For an open polyline: per point (tangent, normal, binormal, widen, bend) - the normal carried by parallel
    transport so the section does not twist, widen the 1/cos the corner ring is stretched by across the bend so the
    wall keeps its thickness (a miter), bend the direction it is stretched along."""
    m = len(pts)
    tans = []
    for i in range(m):
        if i == 0:
            t = pts[1] - pts[0]
        elif i == m - 1:
            t = pts[-1] - pts[-2]
        else:
            t = (pts[i + 1] - pts[i]).normalized() + (pts[i] - pts[i - 1]).normalized()
        tans.append(t.normalized())
    ref = Vector((0, 0, 1)) if abs(tans[0].z) < 0.9 else Vector((1, 0, 0))
    nrm = (ref - tans[0] * ref.dot(tans[0])).normalized()
    out = []
    for i, t in enumerate(tans):
        if i:
            nrm = tans[i - 1].rotation_difference(t) @ nrm
            nrm = (nrm - t * nrm.dot(t)).normalized()
        widen, bend = 1.0, None
        if 0 < i < m - 1:
            seg = (pts[i + 1] - pts[i]).normalized()
            widen = 1.0 / max(abs(seg.dot(t)), 0.2)
            bend = seg - t * seg.dot(t)
            bend = bend.normalized() if bend.length > 1e-9 else None
        out.append((t, nrm.copy(), t.cross(nrm).normalized(), widen, bend))
    return out


def _sweep_rings(pts, shape):
    """A ring of the 2D shape [(a, b), ...] (a along the normal, b along the binormal) at every path point."""
    rings = []
    for p, (_t, n, b, widen, bend) in zip(pts, _path_frames(pts)):
        ring = []
        for sa, sb in shape:
            off = n * sa + b * sb
            if bend is not None and widen != 1.0:
                off = off + bend * off.dot(bend) * (widen - 1.0)
            ring.append(p + off)
        rings.append(ring)
    return rings


def _rrect2(hx, hy, r, nx, ny, kc):
    """A rounded rectangle centered on 0, counter-clockwise: (points, outward normals, side labels). nx and ny points
    along the x and y sides, kc per corner; the count 2nx + 2ny + 4kc is even, so a ladder caps it."""
    r = min(r, 0.99 * hx, 0.99 * hy)
    if r < 1e-6:
        r, kc = 0.0, 0
    cx, cy = hx - r, hy - r
    pts, nrms, sides = [], [], []
    plan = (((hx, -cy), (hx, cy), ny, (1, 0), "+x", (cx, cy), 0.0),
            ((cx, hy), (-cx, hy), nx, (0, 1), "+y", (-cx, cy), 90.0),
            ((-hx, cy), (-hx, -cy), ny, (-1, 0), "-x", (-cx, -cy), 180.0),
            ((-cx, -hy), (cx, -hy), nx, (0, -1), "-y", (cx, -cy), 270.0))
    for (x0, y0), (x1, y1), n, nrm, label, (ccx, ccy), a0 in plan:
        for k in range(n):
            t = k / n
            pts.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
            nrms.append(nrm)
            sides.append(label)
        for k in range(kc):
            a = math.radians(a0 + 90.0 * k / kc)
            pts.append((ccx + r * math.cos(a), ccy + r * math.sin(a)))
            nrms.append((math.cos(a), math.sin(a)))
            sides.append("corner")
    return pts, nrms, sides


def _circle2(r, n):
    return [(r * math.cos(2 * math.pi * k / n), r * math.sin(2 * math.pi * k / n)) for k in range(n)]


def _prop_material(name, color, roughness, metallic=0.0, emission=None):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = _find_type(mat.node_tree.nodes, "BSDF_PRINCIPLED")
    _sock(bsdf.inputs, "Base Color").default_value = tuple(color) + (1.0,)
    _sock(bsdf.inputs, "Roughness").default_value = roughness
    _sock(bsdf.inputs, "Metallic").default_value = metallic
    if emission is not None:
        _sock(bsdf.inputs, "Emission Color", "Emission").default_value = tuple(emission[0]) + (1.0,)
        _sock(bsdf.inputs, "Emission Strength").default_value = emission[1]
    return mat


def _finish_prop(params, parts, op, default_name, slot_mats):
    """Hand the parts to create_mesh, then make the slots' materials and set each face's slot. slot_mats is a list of
    (label, maker) in slot order; maker(name) returns the material. Returns create_mesh's reply with the prop's."""
    uvs = _box_uvs_lists(parts.verts, parts.faces)
    sub = {k: params[k] for k in ("name", "location", "rotation", "collection", "takeOverName") if k in params}
    out = _build_through_create_mesh(sub, parts.verts, parts.faces, uvs, False, default_name, op)
    obj = bpy.data.objects.get(out.get("name") or "")
    used = sorted(set(parts.slots))
    remap = {s: i for i, s in enumerate(used)}
    if obj is not None and obj.type == "MESH":
        mesh = obj.data
        for s in used:
            label, maker = slot_mats[s]
            mesh.materials.append(maker("M_%s_%s" % (obj.name, label)))
        mesh.polygons.foreach_set("material_index", [remap[s] for s in parts.slots])
        # FLAT FACES SHADE FLAT, ROUND ONES SMOOTH: smooth everywhere, sharp edges past 35 degrees
        mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
        _resharpen(mesh, 35.0)
        mesh.update()
        out["materialSlots"] = [{"slot": remap[s], "part": slot_mats[s][0], "material": mesh.materials[remap[s]].name}
                                for s in used]
    counts = {}
    for n in parts.names:
        counts[n] = counts.get(n, 0) + 1
    lo = [min(v[i] for v in parts.verts) for i in range(3)]
    hi = [max(v[i] for v in parts.verts) for i in range(3)]
    out.update({"parts": counts, "boundsLocalM": {"min": [round(v, 6) for v in lo], "max": [round(v, 6) for v in hi]},
                "outwardCheck": {"parts": len(parts.checks),
                                 "turnedOver": sum(1 for c in parts.checks if c["turnedOver"]),
                                 "note": "each part wound one way by construction, then turned so its signed volume "
                                         "is positive"}})
    return out


def _color_or(params, key, default, tail):
    try:
        return _color3(params, key, default)
    except MifOpError as exc:
        raise MifOpError(str(exc).replace("NOTHING was created.", tail))


# ------------------------------------------------------------------------------------------------- build_downspout
_DOWNSPOUT_KEYS = ("name", "height", "offset", "section", "width", "depth", "standoff", "straps", "strapSpacing",
                   "shoe", "color", "location", "rotation", "collection", "takeOverName")
_SPOUT_WALL = 0.0015      # the sheet's thickness: a hollow pipe, so its ends show their opening


def op_build_downspout(params):
    """Build a downspout down a wall: a hollow rectangular or round pipe from the gutter's outlet over the eave's
    offset (a gooseneck of two 45-degree elbows), down the wall on straps screwed to it, out at the foot by a kick-out
    shoe. The wall is the plane y 0 and the pipe stands in +y; z 0 is the ground; place it with location and rotation.

    THE GAP (2026-10-04): downspouts were built in scratch scripts with the same rules as every prop - closed solids,
    butted parts. Here the pipe is one closed hollow solid (its outer and inner walls swept along the path with
    mitred elbows, joined by quad rims at both ends), each strap a formed sheet capped with quads, each part turned out
    by its signed volume.

    params:
      height (m)         the gutter outlet's height, default 3; 0.8-30
      offset (m)         the outlet's distance from the wall (the eave's overhang), default 0.3; under 0.15 it runs
                         straight down with no gooseneck
      section            rect (default, 2 x 3 in) or round (3 in)
      width, depth (m)   a rect pipe's width along the wall (default 0.076) and depth out from it (0.051); a round
                         pipe's diameter is width (default 0.076); depth refused on round
      standoff (m)       the gap between the wall and the pipe's back, default 0.02
      straps (bool)      straps every strapSpacing (m, default 1.5) on the straight run, default true
      shoe (bool)        a kick-out shoe at the foot, default true
      color ([r,g,b])    default a painted brown; name, location, rotation, collection, takeOverName
    """
    tail = "NOTHING was created."
    reject_unknown(params, _DOWNSPOUT_KEYS, "build_downspout")
    height = take_float(params, "height", default=3.0)
    offset = take_float(params, "offset", default=0.3)
    section = str(take(params, "section", default="rect", kind=str)).lower()
    if section not in ("rect", "round"):
        raise MifOpError("section must be rect or round, got %r. %s" % (section, tail))
    if section == "round" and params.get("depth") is not None:
        raise MifOpError("depth sizes a rect pipe; a round one is width across. %s" % tail)
    width = take_float(params, "width", default=0.076)
    depth = take_float(params, "depth", default=0.051) if section == "rect" else width
    standoff = take_float(params, "standoff", default=0.02)
    straps = take_bool(params, "straps", default=True)
    spacing = take_float(params, "strapSpacing", default=1.5)
    shoe = take_bool(params, "shoe", default=True)
    for key, val, lo, hi in (("height", height, 0.8, 30.0), ("offset", offset, 0.0, 2.0), ("width", width, 0.03, 0.3),
                             ("depth", depth, 0.03, 0.3), ("standoff", standoff, 0.005, 0.2),
                             ("strapSpacing", spacing, 0.3, 5.0)):
        if not lo <= val <= hi:
            raise MifOpError("%s must be %g to %g m, got %g. %s" % (key, lo, hi, val, tail))
    if not straps and params.get("strapSpacing") is not None:
        raise MifOpError("strapSpacing places straps and straps is false. %s" % tail)
    color = _color_or(params, "color", (0.20, 0.13, 0.08), tail)
    yc = standoff + depth / 2.0
    pts = [Vector((0.0, max(offset, yc), height))]
    goose = offset - yc >= 0.15
    run_top = height
    if goose:
        pts.append(Vector((0.0, offset, height - 0.08)))
        run_top = height - 0.08 - (offset - yc)
        pts.append(Vector((0.0, yc, run_top)))
    shoe_len = 0.25
    foot = 0.06 + 0.7071 * shoe_len if shoe else 0.03
    if run_top - foot < 0.3:
        raise MifOpError("a %g m downspout has no straight run left between its gooseneck (down to %g m) and its "
                         "foot (%g m); raise height or lower offset. %s" % (height, round(run_top, 3), foot, tail))
    pts.append(Vector((0.0, yc, foot)))
    if shoe:
        pts.append(Vector((0.0, yc + 0.7071 * shoe_len, foot - 0.7071 * shoe_len)))
    if section == "rect":
        outer = _rrect2(width / 2.0, depth / 2.0, 0.006, 4, 3, 3)[0]
        inner = _rrect2(width / 2.0 - _SPOUT_WALL, depth / 2.0 - _SPOUT_WALL, 0.006 - _SPOUT_WALL, 4, 3, 3)[0]
    else:
        outer, inner = _circle2(width / 2.0, 24), _circle2(width / 2.0 - _SPOUT_WALL, 24)
    parts = _PropParts()
    out_rings, in_rings = _sweep_rings(pts, outer), _sweep_rings(pts, inner)
    parts.chain("pipe", 0, out_rings + in_rings[::-1], closed=True)
    zs = []
    if straps:
        lo_z, hi_z = foot + 0.15, run_top - 0.15
        n = max(2, int(math.ceil((hi_z - lo_z) / spacing)) + 1)
        zs = [lo_z + (hi_z - lo_z) * k / (n - 1) for k in range(n)]
        hw = width / 2.0 + 0.0007
        st = 0.0015
        for z in zs:
            if section == "rect":
                front = yc + depth / 2.0 + 0.0007 + st / 2.0
                path = [(-hw - 0.025, st / 2.0), (-hw - st / 2.0, st / 2.0), (-hw - st / 2.0, front),
                        (hw + st / 2.0, front), (hw + st / 2.0, st / 2.0), (hw + 0.025, st / 2.0)]
            else:
                rs = hw + st / 2.0
                path = ([(-rs - 0.025, st / 2.0), (-rs, st / 2.0)] + _arc(0.0, yc, rs, range(180, -1, -20))
                        + [(rs, st / 2.0), (rs + 0.025, st / 2.0)])
            parts.sheet("strap", 1, _sheet(path, st, z - 0.0125, z + 0.0125, _along_z, "strap"))
            for sx in (-1.0, 1.0):
                parts.sheet("screw", 1, _cyl(sx * (hw + 0.014), z, 0.004, st, st + 0.003, 6, _along_y, "screw"))
    out = _finish_prop(params, parts, "build_downspout", "Downspout",
                       [("pipe", lambda n: _prop_material(n, color, 0.45, 0.3)),
                        ("straps", lambda n: _prop_material(n, (0.35, 0.35, 0.36), 0.4, 0.8))])
    along = sum((b - a).length for a, b in zip(pts, pts[1:]))
    out["downspout"] = {"section": section, "outletM": rnd(list(pts[0])), "footM": rnd(list(pts[-1])),
                        "gooseneck": goose, "shoe": shoe, "pathLengthM": round(along, 6), "straps": len(zs),
                        "strapHeightsM": [round(z, 4) for z in zs], "pipeBackYM": round(yc - depth / 2.0, 6),
                        "wallThicknessM": _SPOUT_WALL}
    return out


# ---------------------------------------------------------------------------------------------- build_wall_fixture
_FIXTURE_KEYS = ("name", "kind", "scale", "armLength", "conduitLength", "conduit", "color", "lampColor",
                 "lampStrength", "location", "rotation", "collection", "takeOverName")
_FIXTURE_KINDS = ("wallpack", "gooseneck", "box")


def op_build_wall_fixture(params):
    """Build a fixture mounted on a wall: a wall pack light, a gooseneck barn light, or a junction box with its
    conduit run. The wall is the plane y 0, the fixture stands in +y, its mounting point is the origin; place it with
    location and rotation.

    THE GAP (2026-10-04): wall lights and boxes were built in scratch scripts. Every part here is a closed
    solid - the shade a revolved thin sheet closed on itself, the arm a capped tube, the housing a capped prism, the
    straps formed sheets capped with quads - and each is turned out by its signed volume. A lamp's lens or bulb is its
    own emissive slot.

    params:
      kind               wallpack (default): a slanted housing with a lens; gooseneck: a backplate, a curved arm,
                         a dish shade and a bulb; box: a junction box with a cover and a conduit run with straps
      scale              every size times this, default 1; 0.25-4
      armLength (m)      a gooseneck's reach from the wall, default 0.45; 0.2-1.5
      conduitLength (m)  a box's conduit run, default 1; 0 for none, up to 5
      conduit            down (default) or up
      color ([r,g,b])    the body; lampColor ([r,g,b]) and lampStrength (default 5) the lens's or bulb's light
      name, location, rotation, collection, takeOverName
    """
    tail = "NOTHING was created."
    reject_unknown(params, _FIXTURE_KEYS, "build_wall_fixture")
    kind = str(take(params, "kind", default="wallpack", kind=str)).lower()
    if kind not in _FIXTURE_KINDS:
        raise MifOpError("kind must be one of %s, got %r. %s" % (", ".join(_FIXTURE_KINDS), kind, tail))
    for key, kinds in (("armLength", ("gooseneck",)), ("conduitLength", ("box",)), ("conduit", ("box",)),
                       ("lampColor", ("wallpack", "gooseneck")), ("lampStrength", ("wallpack", "gooseneck"))):
        if params.get(key) is not None and kind not in kinds:
            raise MifOpError("'%s' shapes a %s; kind is %s. %s" % (key, " or ".join(kinds), kind, tail))
    scale = take_float(params, "scale", default=1.0)
    arm = take_float(params, "armLength", default=0.45)
    clen = take_float(params, "conduitLength", default=1.0)
    cdir = str(take(params, "conduit", default="down", kind=str)).lower()
    strength = take_float(params, "lampStrength", default=5.0)
    for key, val, lo, hi in (("scale", scale, 0.25, 4.0), ("armLength", arm, 0.2, 1.5),
                             ("conduitLength", clen, 0.0, 5.0), ("lampStrength", strength, 0.0, 1000.0)):
        if not lo <= val <= hi:
            raise MifOpError("%s must be %g to %g, got %g. %s" % (key, lo, hi, val, tail))
    if cdir not in ("down", "up"):
        raise MifOpError("conduit must be down or up, got %r. %s" % (cdir, tail))
    color = _color_or(params, "color", {"box": (0.55, 0.56, 0.55)}.get(kind, (0.12, 0.12, 0.12)), tail)
    lamp = _color_or(params, "lampColor", (1.0, 0.78, 0.5), tail)
    parts = _PropParts()
    info = {"kind": kind}
    if kind == "wallpack":
        prof = [(0.0, -0.13), (0.17, -0.13), (0.17, 0.02), (0.15, 0.08), (0.11, 0.13), (0.0, 0.13)]
        parts.chain("housing", 0, [[(x, y, z) for y, z in prof] for x in (-0.15, 0.15)])
        parts.box("lens", 1, -0.12, 0.12, 0.17, 0.176, -0.11, 0.0)
        for x in (-0.09, 0.0, 0.09):
            parts.box("fin", 0, x - 0.004, x + 0.004, 0.02, 0.10, 0.13, 0.15)
        info["lens"] = "the front face, slot lamp"
    elif kind == "gooseneck":
        parts.sheet("backplate", 0, _cyl(0.0, 0.0, 0.065, 0.0, 0.015, 16, _along_y, "backplate"))
        rb = min(0.12, arm * 0.4)
        path = [Vector((0.0, 0.015, 0.0)), Vector((0.0, arm - rb, 0.0))]
        for k in range(1, 7):
            a = math.radians(90.0 * k / 6)
            path.append(Vector((0.0, arm - rb + rb * math.sin(a), -rb + rb * math.cos(a))))
        parts.chain("arm", 0, _sweep_rings(path, _circle2(0.011, 12)))
        end = path[-1]
        zs = end.z - 0.04
        parts.sheet("socket", 0, _cyl(0.0, end.y, 0.02, zs, end.z, 16, _along_z, "socket"))
        # THE SHADE: a thin dish revolved round the drop - its section a closed loop, so the revolve closes on itself
        outer_run = [(0.021, 0.0), (0.06, -0.02), (0.12, -0.07), (0.17, -0.12)]
        loop = outer_run + [(0.17, -0.124)] + [(r - 0.001, z - 0.0035) for r, z in reversed(outer_run[1:])] \
            + [(0.021, -0.004)]
        segs = 32
        rings = [[(r * math.cos(2 * math.pi * k / segs), end.y + r * math.sin(2 * math.pi * k / segs), zs + z)
                  for k in range(segs)] for r, z in loop]
        parts.chain("shade", 0, rings, closed=True)
        bulb = [(0.012, -0.004), (0.024, -0.016), (0.034, -0.04), (0.032, -0.06), (0.02, -0.075), (0.008, -0.08)]
        parts.chain("bulb", 1, [[(r * math.cos(2 * math.pi * k / 16), end.y + r * math.sin(2 * math.pi * k / 16),
                                  zs + z) for k in range(16)] for r, z in bulb])
        info.update({"reachM": round(end.y * scale, 6), "shadeRimM": round((zs - 0.124) * scale, 6)})
    else:
        pts, _n, _s = _rrect2(0.05, 0.05, 0.006, 3, 3, 3)
        parts.chain("box", 0, [[(x, y, z) for x, z in pts] for y in (0.0, 0.045)])
        pts2, _n, _s = _rrect2(0.055, 0.055, 0.008, 3, 3, 3)
        parts.chain("cover", 0, [[(x, y, z) for x, z in pts2] for y in (0.045, 0.05)])
        for z in (-0.04, 0.04):
            parts.sheet("screw", 0, _cyl(0.0, z, 0.004, 0.05, 0.053, 6, _along_y, "screw"))
        sgn = -1.0 if cdir == "down" else 1.0
        straps_at = []
        if clen > 0.0:
            yc = 0.0225
            z0 = sgn * 0.05
            parts.sheet("connector", 0, _cyl(0.0, yc, 0.013, min(z0, z0 + sgn * 0.025), max(z0, z0 + sgn * 0.025),
                                            12, _along_z, "connector"))
            z1, z2 = z0 + sgn * 0.025, z0 + sgn * (0.025 + clen)
            parts.sheet("conduit", 0, _cyl(0.0, yc, 0.0095, min(z1, z2), max(z1, z2), 12, _along_z, "conduit"))
            n = max(1, int(clen / 0.9))
            rs = 0.0095 + 0.0007 + 0.00075
            for k in range(n):
                z = z1 + sgn * clen * (k + 0.5) / n
                path = [(-rs, 0.00075)] + _arc(0.0, yc, rs, range(180, -1, -20)) + [(rs, 0.00075),
                                                                                      (rs + 0.025, 0.00075)]
                parts.sheet("conduit strap", 0, _sheet(path, 0.0015, z - 0.01, z + 0.01, _along_z, "strap"))
                straps_at.append(round(z * scale, 4))
        info.update({"conduit": cdir if clen > 0 else None, "conduitLengthM": clen * scale,
                     "conduitStrapsAtM": straps_at})
    if scale != 1.0:
        parts.verts = [[c * scale for c in v] for v in parts.verts]
    out = _finish_prop(params, parts, "build_wall_fixture", "WallFixture",
                       [("body", lambda n: _prop_material(n, color, 0.5, 0.6)),
                        ("lamp", lambda n: _prop_material(n, lamp, 0.2, 0.0, (lamp, strength)))])
    info["scale"] = scale
    info["mountedOn"] = "the wall plane y 0 (the fixture's back is at y 0, it stands in +y)"
    out["fixture"] = info
    return out


# --------------------------------------------------------------------------------------------------- build_dumpster
_DUMPSTER_KEYS = ("name", "width", "depth", "height", "frontSlope", "cornerRadius", "ribs", "dents", "seed",
                  "lidAngles", "casters", "fill", "carton", "color", "lidColor", "location", "rotation", "collection",
                  "takeOverName")
_DUMP_T, _DUMP_FLOOR, _DUMP_RIB = 0.003, 0.004, 0.018     # the steel sheet, the floor plate, a pressed rib's depth
_DUMP_STEP = 0.02                                          # plan points every 2 cm, so a rib has points to press
# WHERE THE PLAN'S POINTS ARE KEPT (2026-11-01). Every 2 cm all round gave 31,904 triangles at 1.83 m, four times gameReady's
# number for the size, nearly all on flat sheet. A rib keeps every point, a dent every fourth, flat sheet every twelfth;
# levels up the wall every 14 cm, and the inner sheet, seen only through an open lid, every third one. Up the sloped front
# they stay 8 cm apart: the plan shortens there and a rib slides along it, and wider apart it would smear across the sheet.
_DUMP_EVERY_DENT, _DUMP_EVERY_FLAT = 4, 12
_DUMP_LEVEL, _DUMP_LEVEL_SLOPE = 0.14, 0.08


def _dump_span(spec, z):
    """(y0, y1): the body's plan at height z runs from the back (y 0) to the front, set back on the sloped lower
    front."""
    zb, zs = spec["zb"], spec["slopeTop"]
    s = spec["frontSlope"] * (zs - z) / (zs - zb) if z < zs else 0.0
    return 0.0, spec["depth"] - s


def _dump_press(spec, side, along, z):
    """How far (m) the sheet is pressed in at a plan point: the vertical ribs on the sides and the back, and the
    seeded dents. Both fade out 12-20 cm from the bottom and the rim, so the rim and the floor stay true."""
    zb, zt = spec["zb"], spec["zt"]
    env = min(1.0, max(0.0, (z - zb - 0.12) / 0.08), max(0.0, (zt - 0.12 - z) / 0.08))
    if env <= 0.0:
        return 0.0
    d = 0.0
    if side in ("+x", "-x", "-y") and spec["ribs"]:
        span = spec["depth"] if side != "-y" else spec["width"]
        off = 0.0 if side != "-y" else -spec["width"] / 2.0
        for k in range(spec["ribs"]):
            u = (along - (off + span * (k + 1) / (spec["ribs"] + 1))) / 0.05
            if abs(u) < 1.0:
                d += _DUMP_RIB * 0.5 * (1.0 + math.cos(math.pi * u))
    for dside, dal, dz, rad, dep in spec["dentList"]:
        if dside == side:
            q = ((along - dal) ** 2 + (z - dz) ** 2) / (rad * rad)
            if q < 4.0:
                d += dep * math.exp(-q)
    return d * env


def _at_corner(sides, i):
    """A point of a corner's arc, or the point of a straight side that meets one: without it the corner is cut short."""
    return "corner" in (sides[i], sides[i - 1], sides[(i + 1) % len(sides)])


def _plan_even(pts, keep):
    """The kept plan points with no two on one spot (a corner's arc begins where the side ends, and both are in the plan:
    two such points make zero-area faces up the whole wall), in an even count, since a chain of rings needs one."""
    out = []
    for i in keep:
        if not out or math.hypot(pts[i][0] - pts[out[-1]][0], pts[i][1] - pts[out[-1]][1]) > 1e-6:
            out.append(i)
    if len(out) > 1 and math.hypot(pts[out[0]][0] - pts[out[-1]][0], pts[out[0]][1] - pts[out[-1]][1]) <= 1e-6:
        out.pop()
    if len(out) % 2:
        have = set(out)
        far = max((i for i in range(len(pts)) if i not in have),
                  key=lambda i: min(math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]) for j in out))
        out = sorted(out + [far])
    return out


def _dump_keep(spec):
    """Which of the plan's points the body keeps, as indices: an even count, one list for every ring so they stay a
    chain. The side walls' points slide forward up the slope, so a rib or dent there covers the span it slides over: at
    the height it fades in, its place along the shorter plan is a greater share of it."""
    width, depth = spec["width"], spec["depth"]
    pts, _n, sides = _rrect2(width / 2.0, depth / 2.0, spec["cornerRadius"], spec["nx"], spec["ny"], 4)
    y_low = _dump_span(spec, spec["zb"] + 0.12)[1]
    slide = depth / max(y_low, 1e-6)

    def zones(side):
        fine, soft = [], []
        along_scale = slide if side in ("+x", "-x") else 1.0
        if side in ("+x", "-x", "-y") and spec["ribs"]:
            span = depth if side != "-y" else width
            off = 0.0 if side != "-y" else -width / 2.0
            for k in range(spec["ribs"]):
                c = off + span * (k + 1) / (spec["ribs"] + 1)
                fine.append((c - 0.07, c * along_scale + 0.07 * along_scale))
        for dside, dal, _dz, rad, _dep in spec["dentList"]:
            if dside == side:
                soft.append((dal - 2.0 * rad, dal * along_scale + 2.0 * rad * along_scale))
        return fine, soft
    by_side = {s: zones(s) for s in ("+x", "-x", "-y", "+y")}
    keep = []
    for i, ((px, py), side) in enumerate(zip(pts, sides)):
        if _at_corner(sides, i):
            keep.append(i)
            continue
        along = depth / 2.0 + py if side in ("+x", "-x") else px
        fine, soft = by_side[side]
        if any(a <= along <= b for a, b in fine):
            keep.append(i)
        elif i % _DUMP_EVERY_DENT == 0 and any(a <= along <= b for a, b in soft):
            keep.append(i)
        elif i % _DUMP_EVERY_FLAT == 0:
            keep.append(i)
    return _plan_even(pts, keep)


def _dump_ring(spec, z, inset):
    """One plan ring of the body at height z, inset by `inset` (the inner wall), pressed by ribs and dents."""
    y0, y1 = _dump_span(spec, z)
    hx, hy = spec["width"] / 2.0 - inset, (y1 - y0) / 2.0 - inset
    pts, nrms, sides = _rrect2(hx, hy, max(0.003, spec["cornerRadius"] - inset), spec["nx"], spec["ny"], 4)
    cy = (y0 + y1) / 2.0
    ring = []
    keep = spec["keep"]
    pts, nrms, sides = [pts[i] for i in keep], [nrms[i] for i in keep], [sides[i] for i in keep]
    for (px, py), (nx, ny), side in zip(pts, nrms, sides):
        along = (cy + py) if side in ("+x", "-x") else px
        d = _dump_press(spec, side, along, z) if side != "corner" else 0.0
        ring.append((px - nx * d, cy + py - ny * d, z))
    return ring


def _dump_lid(spec, x0, x1, skirt_left, skirt_right, angle):
    """One moulded lid as a closed height-field slab: a domed top with two raised ribs, a flat underside, and a skirt
    down its outer side and its front (not at the hinge or the seam, where it would meet the rim channel or the other
    lid). Built closed at the hinge, then turned up by `angle` degrees round the hinge rod."""
    zt, depth = spec["zt"], spec["depth"]
    yh, zh = spec["hinge"]
    under = zt + 0.008
    y0, y1 = -0.025, depth + 0.045
    xs = _with_lines(_with_grid((x0, x1), 0.04), [x0 + 0.006, x1 - 0.006])
    ys = _with_lines(_with_grid((y0, y1), 0.04), [y1 - 0.006])
    nx, ny = len(xs), len(ys)
    w = x1 - x0

    def top(x, y):
        u = 2.0 * (x - x0) / w - 1.0
        v = 2.0 * (y - y0) / (y1 - y0) - 1.0
        h = under + 0.03 + 0.03 * (1.0 - u * u) * (1.0 - v * v)
        for c in (-0.5, 0.5):
            q = (u - c) / 0.12
            if abs(q) < 1.0 and abs(v) < 0.85:
                h += 0.012 * 0.5 * (1.0 + math.cos(math.pi * q)) * min(1.0, (0.85 - abs(v)) / 0.1)
        return h

    def bottom(i, j):
        skirt = (j == ny - 1) or (i == 0 and skirt_left) or (i == nx - 1 and skirt_right)
        return under - 0.04 if skirt else under

    verts = []
    for side in (0, 1):
        for i in range(nx):
            for j in range(ny):
                verts.append([xs[i], ys[j], top(xs[i], ys[j]) if side == 0 else bottom(i, j)])

    def t(i, j):
        return i * ny + j

    def b(i, j):
        return nx * ny + i * ny + j

    faces = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            faces.append([t(i, j), t(i + 1, j), t(i + 1, j + 1), t(i, j + 1)])
            faces.append([b(i, j), b(i, j + 1), b(i + 1, j + 1), b(i + 1, j)])
    loop = [(i, 0) for i in range(nx - 1)] + [(nx - 1, j) for j in range(ny - 1)] + \
        [(i, ny - 1) for i in range(nx - 1, 0, -1)] + [(0, j) for j in range(ny - 1, 0, -1)]
    for k, (i, j) in enumerate(loop):
        qi, qj = loop[(k + 1) % len(loop)]
        faces.append([t(qi, qj), t(i, j), b(i, j), b(qi, qj)])
    grip = [[x, y, z] for x in ((x0 + x1) / 2.0 - 0.12, (x0 + x1) / 2.0 + 0.12) for y in (y1, y1 + 0.025)
            for z in (under - 0.03, under)]
    knuckles = []
    for c in (x0 + 0.2 * w, x1 - 0.2 * w):
        rings = []
        for r, xa in ((0.02, c - 0.04), (0.02, c + 0.04), (0.0135, c + 0.04), (0.0135, c - 0.04)):
            rings.append([[xa, yh + r * math.cos(2 * math.pi * k / 16), zh + r * math.sin(2 * math.pi * k / 16)]
                          for k in range(16)])
        knuckles.append(rings)
    a = math.radians(angle)
    ca, sa = math.cos(a), math.sin(a)

    def turn(p):
        dy, dz = p[1] - yh, p[2] - zh
        return [p[0], yh + dy * ca - dz * sa, zh + dy * sa + dz * ca]

    return ([turn(p) for p in verts], faces, [turn(p) for p in grip],
            [[[turn(p) for p in ring] for ring in rings] for rings in knuckles])


def _dump_inside(spec, p, margin=0.005):
    """Is a point inside the bin's inner shell - between its walls (pressed in by ribs and dents at most) and above
    its floor? Above the rim there is no wall to cross."""
    x, y, z = p
    if z >= spec["zt"]:
        return True
    if z < spec["zb"] + _DUMP_FLOOR - 1e-9:
        return False
    y0, y1 = _dump_span(spec, z)
    press = _DUMP_RIB + spec["dentMax"] + _DUMP_T + margin
    return abs(x) <= spec["width"] / 2.0 - press and y0 + press <= y <= y1 - press


def _dump_fill(spec, rnd_):
    """Bags (and a carton) in the bin up to fill x its inner height, in layers on a grid, each jittered and turned.
    EVERY ONE IS CHECKED VERTEX BY VERTEX AGAINST THE WALLS: one that crosses is shrunk, and dropped if it still
    crosses - a heap built without the check pushed three bags through a bin's front and side walls."""
    floor = spec["zb"] + _DUMP_FLOOR
    top = spec["zb"] + spec["fill"] * (spec["zt"] - spec["zb"])
    inner_w = spec["width"] - 2.0 * (_DUMP_RIB + spec["dentMax"] + _DUMP_T)
    bag_w = min(0.42, inner_w * 0.9)
    bag_h = bag_w * 1.2
    placed, report = [], {"checked": 0, "shrunk": 0, "dropped": 0, "crossingsFound": 0}
    layer = 0
    while True:
        z = floor + layer * bag_h * 0.92
        if z + bag_h * 0.5 > top or z >= spec["zt"]:
            break
        y0, y1 = _dump_span(spec, z)
        # A PITCH WITH ROOM FOR THE KNUCKLES AND THE JITTER, so the check has little to drop
        pitch = bag_w * 1.12
        nxb = max(1, int((inner_w - 0.06) / pitch))
        nyb = max(1, int((y1 - y0 - 2.0 * (_DUMP_RIB + spec["dentMax"] + _DUMP_T) - 0.06) / pitch))
        # AN ODD LAYER SITS IN THE GAPS of the one below: one bag fewer each way, offset half a pitch
        if layer % 2:
            nxb, nyb = max(1, nxb - 1), max(1, nyb - 1)
        for i in range(nxb):
            for j in range(nyb):
                cx = (i - (nxb - 1) / 2.0) * pitch
                cy = (y0 + y1) / 2.0 + (j - (nyb - 1) / 2.0) * pitch
                cx += rnd_.uniform(-0.03, 0.03)
                cy += rnd_.uniform(-0.03, 0.03)
                g = _bag_mesh(bag_h, bag_w, 0.45, rnd_.randint(1, 10 ** 6), 6, 0.1, 9, 1.0, True, 24, 20)
                spin, tilt = rnd_.uniform(0, 2 * math.pi), math.radians(rnd_.uniform(-8.0, 8.0))
                rot = Matrix.Rotation(spin, 3, "Z") @ Matrix.Rotation(tilt, 3, "X")
                for attempt in range(4):
                    scl = 0.9 ** attempt
                    pts = [rot @ (Vector(v) * scl) for v in g["verts"]]
                    lift = z - min(p.z for p in pts)        # a tilted bag rests on its lowest point, not below it
                    pts = [[cx + p.x, cy + p.y, lift + p.z] for p in pts]
                    report["checked"] += 1
                    if all(_dump_inside(spec, p) for p in pts):
                        break
                    report["crossingsFound"] += 1
                else:
                    report["dropped"] += 1
                    continue
                if attempt:
                    report["shrunk"] += 1
                placed.append(("bag", pts, g["faces"]))
        layer += 1
    if spec["carton"] and placed:
        name, pts, _f = placed[-1]
        c = Vector([sum(p[k] for p in pts) / len(pts) for k in range(3)])
        zc = min(p[2] for p in pts)
        rot = Matrix.Rotation(rnd_.uniform(0, math.pi), 3, "Z")
        for attempt in range(4):
            s = 0.9 ** attempt
            hx, hy, hz = 0.2 * s, 0.15 * s, 0.25 * s
            corners = [list(Vector((c.x, c.y, zc)) + rot @ Vector((sx * hx, sy * hy, sz * hz)))
                       for sz in (0.0, 1.0) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            report["checked"] += 1
            if all(_dump_inside(spec, p) for p in corners):
                placed[-1] = ("carton", corners, None)
                break
            report["crossingsFound"] += 1
    return placed, report


def op_build_dumpster(params):
    """Build a front-load refuse container: a steel body lofted from rounded-rectangle sections with a sloped lower
    front, ribs pressed into the sheet and a soft dent field, a rolled channel round the mouth, fork pockets, skids or
    swivel casters, a hinge rod and two moulded lids opened by angle - and optionally filled with bags and a carton,
    each checked against the walls vertex by vertex. The back is at y 0, the front toward +y, the floor at z 0, the
    origin at the middle of the back's foot.

    THE GAP (2026-10-04): a much-seen dumpster came from a 330-line local geometry script sent through
    create_mesh; its first heap poked three bags through the front and side walls until a vertex-in-wall check
    found them. Here the body is one closed solid - the outer sheet lofted up, the rim, the inner sheet lofted down to
    the floor, capped with quads - so the inside is seen through an open lid; ribs and dents press both sheets alike.
    Every part is closed and turned out by its signed volume.

    params:
      width, depth, height (m)   default 1.83 x 1.4 x 1.2 (to the rim); 0.8-4, 0.6-3, 0.6-2.5
      frontSlope (m)    how far the lower front is set back at the foot, default 0.35; 0 to half the depth
      cornerRadius (m)  the brake-formed corners, default 0.04; 0.01-0.2
      ribs (int)        vertical ribs pressed into each side and the back, default 2; 0-6
      dents             the dent field's strength, default 0.3; 0-1. seed (int) default 1
      lidAngles         [left, right] degrees open, default [0, 0]; 0-100
      casters (bool)    four swivel casters instead of skids, default false
      fill              0 (default) to 1.2 x the inner height filled with bags (over 1 heaps above the rim);
                        carton (bool, default true when filled) puts a carton on top
      color, lidColor ([r,g,b]); name, location, rotation, collection, takeOverName
    """
    tail = "NOTHING was created."
    reject_unknown(params, _DUMPSTER_KEYS, "build_dumpster")
    width = take_float(params, "width", default=1.83)
    depth = take_float(params, "depth", default=1.4)
    height = take_float(params, "height", default=1.2)
    slope = take_float(params, "frontSlope", default=0.35)
    corner = take_float(params, "cornerRadius", default=0.04)
    ribs = take_int(params, "ribs", default=2)
    dents = take_float(params, "dents", default=0.3)
    seed = take_int(params, "seed", default=1)
    casters = take_bool(params, "casters", default=False)
    fill = take_float(params, "fill", default=0.0)
    for key, val, lo, hi in (("width", width, 0.8, 4.0), ("depth", depth, 0.6, 3.0), ("height", height, 0.6, 2.5),
                             ("frontSlope", slope, 0.0, depth / 2.0), ("cornerRadius", corner, 0.01, 0.2),
                             ("ribs", ribs, 0, 6), ("dents", dents, 0.0, 1.0), ("fill", fill, 0.0, 1.2)):
        if not lo <= val <= hi:
            raise MifOpError("%s must be %g to %g, got %g. %s" % (key, lo, hi, val, tail))
    angles = params.get("lidAngles", [0.0, 0.0])
    if not isinstance(angles, (list, tuple)) or len(angles) != 2:
        raise MifOpError("lidAngles is [left, right] in degrees, got %r. %s" % (angles, tail))
    angles = finite_floats(list(angles), "lidAngles")
    if any(not 0.0 <= a <= 100.0 for a in angles):
        raise MifOpError("each lid opens 0 to 100 degrees, got %r. %s" % (angles, tail))
    if params.get("carton") is not None and fill <= 0.0:
        raise MifOpError("carton goes on the fill and fill is 0. %s" % tail)
    carton = take_bool(params, "carton", default=fill > 0.0)
    color = _color_or(params, "color", (0.05, 0.20, 0.10), tail)
    lid_color = _color_or(params, "lidColor", (0.02, 0.025, 0.02), tail)
    rnd_ = random.Random(seed)
    zb = 0.16 if casters else 0.06
    zt = height
    if zt - zb < 0.4:
        raise MifOpError("a %g m bin on its %s has a body under 0.4 m. %s" % (height, "casters" if casters else "skids",
                                                                            tail))
    spec = {"width": width, "depth": depth, "zb": zb, "zt": zt, "slopeTop": zb + 0.45 * (zt - zb),
            "frontSlope": slope, "cornerRadius": corner, "ribs": ribs, "fill": fill, "carton": carton,
            "nx": int(math.ceil(width / _DUMP_STEP)), "ny": int(math.ceil(depth / _DUMP_STEP)),
            "hinge": (-0.045, zt + 0.016)}
    dent_list = []
    for _ in range(6 if dents > 0 else 0):
        side = rnd_.choice(("+x", "-x", "-y", "+y"))
        span = depth if side in ("+x", "-x") else width
        lo = 0.0 if side in ("+x", "-x") else -width / 2.0
        dent_list.append((side, lo + rnd_.uniform(0.15, 0.85) * span, rnd_.uniform(zb + 0.25, zt - 0.25),
                          rnd_.uniform(0.1, 0.2), dents * rnd_.uniform(0.008, 0.02)))
    spec["dentList"] = dent_list
    spec["dentMax"] = max([d[4] for d in dent_list] or [0.0])
    spec["keep"] = _dump_keep(spec)
    fill_parts, fill_report = ([], None)
    if fill > 0.0:
        fill_parts, fill_report = _dump_fill(spec, rnd_)
        if not fill_parts:
            raise MifOpError("fill %g placed no bag: every one crossed a wall even shrunk (checked %d). %s"
                             % (fill, fill_report["checked"], tail))

    parts = _PropParts()
    # ---- the body: outer sheet up, rim, inner sheet down to the floor - one closed solid
    n1 = max(3, int(math.ceil((spec["slopeTop"] - zb) / _DUMP_LEVEL_SLOPE)) + 1)
    n2 = max(3, int(math.ceil((zt - spec["slopeTop"]) / _DUMP_LEVEL)) + 1)
    levels = [zb + (spec["slopeTop"] - zb) * k / (n1 - 1) for k in range(n1)] + \
        [spec["slopeTop"] + (zt - spec["slopeTop"]) * k / (n2 - 1) for k in range(1, n2)]
    outer = [_dump_ring(spec, z, 0.0) for z in levels]
    inner = [_dump_ring(spec, z, _DUMP_T) for z in list(reversed(levels[1:]))[::3]] + \
        [_dump_ring(spec, zb + _DUMP_FLOOR, _DUMP_T)]
    parts.chain("body", 0, outer + inner, fan_caps=True)
    # ---- the rolled channel round the mouth: a D section swept round the rim, closed on itself
    y0, y1 = _dump_span(spec, zt)
    pts, nrms, rim_sides = _rrect2(width / 2.0, (y1 - y0) / 2.0, corner, spec["nx"], spec["ny"], 4)
    cy = (y0 + y1) / 2.0
    # The rim runs straight between its corners, so only the corners need points.
    rim_keep = _plan_even(pts, [i for i in range(len(pts)) if _at_corner(rim_sides, i) or i % (2 * _DUMP_EVERY_FLAT) == 0])
    dsec = [(0.002 + 0.03 * math.cos(math.radians(-90 + 180 * i / 7)),
             -0.02 + 0.025 * math.sin(math.radians(-90 + 180 * i / 7))) for i in range(8)]
    parts.chain("rim channel", 0, [[(pts[i][0] + nrms[i][0] * o, cy + pts[i][1] + nrms[i][1] * o, zt + v) for o, v in dsec]
                                   for i in rim_keep], closed=True)
    # ---- fork pockets: open rectangular tubes along the sides
    zp = zb + 0.6 * (zt - zb)
    po, _n, _s = _rrect2(0.115, 0.04, 0.006, 4, 2, 3)
    pi_, _n, _s = _rrect2(0.112, 0.037, 0.003, 4, 2, 3)
    for sx in (-1.0, 1.0):
        xc = sx * (width / 2.0 + 0.004 + 0.115)
        path = [Vector((xc, 0.06, zp)), Vector((xc, depth - 0.06, zp))]
        rings_o = _sweep_rings(path, [(b, a) for a, b in po])
        rings_i = _sweep_rings(path, [(b, a) for a, b in pi_])
        parts.chain("fork pocket", 2, rings_o + rings_i[::-1], closed=True)
    # ---- skids or casters
    foot_y1 = depth - slope
    if casters:
        for sx in (-1.0, 1.0):
            for cyy in (0.12, foot_y1 - 0.12):
                cx = sx * (width / 2.0 - 0.12)
                parts.box("caster", 2, cx - 0.05, cx + 0.05, cyy - 0.05, cyy + 0.05, zb - 0.008, zb)
                parts.bar("caster", 2, (cx, cyy, zb - 0.008), (cx, cyy, 0.13), 0.012, 8)
                parts.box("caster", 2, cx - 0.038, cx + 0.038, cyy - 0.03, cyy + 0.03, 0.122, 0.13)
                for fx in (-1.0, 1.0):
                    parts.box("caster", 2, cx + fx * 0.032 - 0.003, cx + fx * 0.032 + 0.003, cyy - 0.03, cyy + 0.03,
                              0.03, 0.122)
                    parts.bar("caster", 2, (cx + fx * 0.035, cyy, 0.055), (cx + fx * 0.041, cyy, 0.055), 0.007, 6)
                parts.bar("wheel", 3, (cx - 0.0225, cyy, 0.055), (cx + 0.0225, cyy, 0.055), 0.055, 16)
    else:
        for sx in (-1.0, 1.0):
            cx = sx * (width / 2.0 - 0.25)
            parts.box("skid", 2, cx - 0.04, cx + 0.04, 0.05, foot_y1 - 0.05, 0.0, zb)
    # ---- the hinge rod on brackets behind the rim, and the lids
    yh, zh = spec["hinge"]
    parts.bar("hinge rod", 2, (-width / 2.0 + 0.02, yh, zh), (width / 2.0 - 0.02, yh, zh), 0.012, 12)
    for xb in (-width / 2.0 + 0.15, 0.0, width / 2.0 - 0.15):
        parts.box("hinge bracket", 2, xb - 0.004, xb + 0.004, -0.06, -0.034, zt - 0.09, zt + 0.002)
        parts.box("hinge bracket", 2, xb - 0.02, xb + 0.02, -0.034, -0.0005, zt - 0.09, zt - 0.06)
    for (x0, x1, sl, sr), ang in zip(((-width / 2.0 - 0.04, -0.004, True, False),
                                      (0.004, width / 2.0 + 0.04, False, True)), angles):
        lv, lf, grip, knuckles = _dump_lid(spec, x0, x1, sl, sr, ang)
        parts.raw("lid", 1, lv, lf)
        parts.chain("lid grip", 1, [[grip[i] for i in (0, 4, 6, 2)], [grip[i] for i in (1, 5, 7, 3)]])
        for rings in knuckles:
            parts.chain("lid knuckle", 1, rings, closed=True)
    # ---- the fill
    for kind, pts_, faces in fill_parts:
        if kind == "bag":
            parts.raw("bag", 4, pts_, faces)
        else:
            parts.chain("carton", 5, [pts_[:4], pts_[4:]])
    out = _finish_prop(params, parts, "build_dumpster", "Dumpster",
                       [("body", lambda n: _prop_material(n, color, 0.55, 0.4)),
                        ("lid", lambda n: _prop_material(n, lid_color, 0.45)),
                        ("steel", lambda n: _prop_material(n, (0.3, 0.3, 0.3), 0.5, 0.8)),
                        ("wheel", lambda n: _prop_material(n, (0.05, 0.05, 0.05), 0.8)),
                        ("bags", lambda n: _prop_material(n, (0.018, 0.018, 0.02), 0.32)),
                        ("carton", lambda n: _prop_material(n, (0.45, 0.33, 0.2), 0.85))])
    # THE CAPACITY: the inner plan's area integrated from the floor to the rim
    cap, prev = 0.0, None
    for z in [zb + _DUMP_FLOOR] + levels[1:]:
        y0, y1 = _dump_span(spec, z)
        hx, hy, r = width / 2.0 - _DUMP_T, (y1 - y0) / 2.0 - _DUMP_T, max(0.003, corner - _DUMP_T)
        area = 4.0 * hx * hy - (4.0 - math.pi) * r * r
        if prev is not None:
            cap += 0.5 * (area + prev[1]) * (z - prev[0])
        prev = (z, area)
    out["dumpster"] = {"widthM": width, "depthM": depth, "rimHeightM": zt, "bodyFootM": zb,
                       "frontSlopeM": slope, "ribsPerSide": ribs, "dents": len(dent_list), "lidAnglesDeg": angles,
                       "casters": casters, "capacityM3": round(cap, 4), "capacityYd3": round(cap * 1.307951, 3),
                       "fill": fill}
    if fill_report is not None:
        fill_report.update({"bags": sum(1 for k, _p, _f in fill_parts if k == "bag"),
                            "carton": any(k == "carton" for k, _p, _f in fill_parts),
                            "rule": "every vertex of a bag or carton below the rim lies inside the inner walls "
                                    "(less the deepest rib and dent) and above the floor; one that crosses is "
                                    "shrunk to 0.9, 0.81, 0.73, then dropped"})
        out["fillCheck"] = fill_report
    return out


# ---------------------------------------------------------------------------------------------------- build_prop
# THE GENERAL DRESSING PROPS. The builders above each make one machine-made thing; an environment is also full of carpentry,
# and a box with a wood color is what it got. Here every prop is made the way it is made in a shop: separate boards, each a
# closed solid with its own eased edges, bow and worn corners and its grain running along it, butted or laid on one another.
# Sizes are in centimeters like the kit's and the rock's. A slot for the wood and a slot for the iron where there is any.
_PROP_KEYS = ("name", "kind", "seed", "width", "depth", "height", "wear", "detail", "color", "color2", "metalColor", "lid",
              "collision", "location", "rotation", "collection", "takeOverName")
_PROP_KINDS = ("crate", "pallet", "barrel", "table", "bench", "chair", "shelf")
# width, depth, height in cm, each a common size of the thing
_PROP_SIZES = {"crate": (70.0, 50.0, 50.0), "pallet": (120.0, 80.0, 14.4), "barrel": (60.0, 60.0, 88.0),
               "table": (150.0, 80.0, 75.0), "bench": (140.0, 34.0, 45.0), "chair": (44.0, 46.0, 90.0),
               "shelf": (100.0, 36.0, 180.0)}
# the least width, depth and height each can be built at, in cm
_PROP_LIMITS = {"crate": (20.0, 20.0, 25.0), "pallet": (40.0, 40.0, 8.0), "barrel": (25.0, 25.0, 35.0),
                "table": (40.0, 40.0, 40.0), "bench": (40.0, 20.0, 25.0), "chair": (30.0, 30.0, 60.0),
                "shelf": (40.0, 20.0, 60.0)}
_NOTES_PROP = {"size": "pass width, depth and height in centimeters.", "scale": "pass width, depth and height in centimeters.",
               "material": "the prop makes a wood material and, where there is iron, a metal one; color, color2 and "
                           "metalColor tune them.", "pivot": "the pivot is always the bottom center."}


class _Carpentry(_PropParts):
    """_PropParts whose faces carry UVs: each board's grain runs along its U, in meters, so a tiling wood texture follows
    the board whichever way it lies, and each board starts somewhere else on it so no two match. One tone per part."""

    def __init__(self, rng, wear, low=False):
        _PropParts.__init__(self)
        self.rng, self.wear, self.low, self.uvs, self.tones = rng, wear, low, [], []
        self.step = 0.22        # a station along a board this far apart; a kind can ask for fewer

    def _local_uvs(self, f0, origin, s_dir, a_dir, b_dir):
        origin, s_dir, a_dir, b_dir = Vector(origin), Vector(s_dir), Vector(a_dir), Vector(b_dir)
        du, dv = self.rng.uniform(0.0, 7.0), self.rng.uniform(0.0, 7.0)
        tone = self.rng.random()
        for f in self.faces[f0:]:
            pts = [Vector(self.verts[i]) - origin for i in f]
            nrm = (pts[1] - pts[0]).cross(pts[2] - pts[0])
            ns, na, nb = abs(nrm.dot(s_dir)), abs(nrm.dot(a_dir)), abs(nrm.dot(b_dir))
            if ns >= na and ns >= nb:
                row = [[du + q.dot(a_dir), dv + q.dot(b_dir)] for q in pts]    # end grain
            elif na >= nb:
                row = [[du + q.dot(s_dir), dv + q.dot(b_dir)] for q in pts]
            else:
                row = [[du + q.dot(s_dir), dv + q.dot(a_dir)] for q in pts]
            self.uvs.append(row)
            self.tones.append(tone)

    def board(self, name, p0, p1, across, w, t, step=None, plain=False, stiff=False, sag=None, gap=None, cut0=None, cut1=None):
        """A board from p0 to p1 (meters, the middle of its ends), w wide along `across`, t thick: a rounded section with a
        chamfered ring at each end, bowed a little and worn at the corners by the prop's wear.

        A BOW MUST NOT OPEN A JOINT. It was a free gauss of 0.35% of the length, so a shelf's rail stood 6 mm clear of the bowed
        upright it butted and a plank rose off the beam under it. stiff is a framing member, straight to a
        millimeter; sag is the side a plank's supports are on, which it may bow toward and never away from; gap is the clear
        space to the plank beside it, which its bow and twist across stay under half of so neighbors never run through each other.
        cut0 and cut1 are the normal of the plane each end is sawn on, for a raked leg to stand flat or a brace to fit its corner."""
        rng, wear = self.rng, self.wear
        p0, p1 = Vector(p0), Vector(p1)
        length = (p1 - p0).length
        s_dir = (p1 - p0).normalized()
        a_dir = Vector(across) - s_dir * Vector(across).dot(s_dir)
        a_dir.normalize()
        b_dir = s_dir.cross(a_dir)
        ease = min(0.12 * min(w, t), 0.004)
        chamfer = min(0.3 * min(w, t), 0.004, 0.2 * length)
        low = self.low or plain
        mids = 1 if low else max(1, int(length / (step or self.step)))
        stations = [0.0, chamfer] + [chamfer + (length - 2.0 * chamfer) * k / mids for k in range(1, mids)] + \
            [length - chamfer, length]

        def within(val, lim):
            return max(-lim, min(lim, val))

        bow_a = within(rng.gauss(0.0, 0.0035 * length), 0.007 * length)
        bow_b = within(rng.gauss(0.0, 0.0025 * length), 0.005 * length)
        twist = within(rng.gauss(0.0, 0.04), 0.08)
        if stiff:
            bow_a, bow_b, twist = within(bow_a, 0.001), within(bow_b, 0.001), 0.25 * twist
        if gap is not None:
            bow_a, twist = within(bow_a, 0.3 * gap), within(twist, 0.04)
        if sag is not None:
            sag = Vector(sag)
            if abs(sag.dot(a_dir)) > 0.3:
                bow_a = math.copysign(within(bow_a, 0.2 * w), sag.dot(a_dir))
            if abs(sag.dot(b_dir)) > 0.3:
                bow_b = math.copysign(within(bow_b, 0.2 * t), sag.dot(b_dir))
        cut0, cut1 = Vector(cut0) if cut0 else None, Vector(cut1) if cut1 else None
        pts0, _n, sides = _rrect2(0.5 * w, 0.5 * t, ease, 1, 1, 1 if low else 2)
        rings = []
        for i, s in enumerate(stations):
            u = s / length
            shrink = chamfer if i in (0, len(stations) - 1) else 0.0
            bend = math.sin(math.pi * u)
            centre = p0 + s_dir * s + a_dir * (bow_a * bend) + b_dir * (bow_b * bend)
            ang = twist * (u - 0.5)
            ca, sa = math.cos(ang), math.sin(ang)
            ring = []
            for (x, y), side in zip(pts0, sides):
                x2 = x * (1.0 - 2.0 * shrink / w)
                y2 = y * (1.0 - 2.0 * shrink / t)
                # A WORN CORNER is rubbed in, by a different amount every station; a flat face stays flat. By 2.6 mm at most, however
                # thick the board: at 16% of a 7.5 cm leg's width it took 12 mm off the face a rail was let into, and opened the joint
                if side == "corner" and wear > 0.0:
                    rub = wear * rng.uniform(0.0, 0.16) * min(w, t, 0.016)
                    x2 -= math.copysign(rub, x)
                    y2 -= math.copysign(rub * rng.uniform(0.3, 1.0), y)
                off = a_dir * (x2 * ca - y2 * sa) + b_dir * (x2 * sa + y2 * ca)
                # A SAWN END: the whole board sheared along itself, so the end lies on the plane and no ring folds over the next
                slide = 0.0
                if cut0 is not None:
                    slide -= (1.0 - u) * off.dot(cut0) / s_dir.dot(cut0)
                if cut1 is not None:
                    slide -= u * off.dot(cut1) / s_dir.dot(cut1)
                ring.append(centre + off + s_dir * slide)
            rings.append(ring)
        f0 = len(self.faces)
        self.chain(name, 0, rings)
        self._local_uvs(f0, p0, s_dir, a_dir, b_dir)

    def post(self, name, x, y, z0, z1, w, d=None, lean=(0.0, 0.0)):
        self.board(name, (x, y, z0), (x + lean[0], y + lean[1], z1), (1.0, 0.0, 0.0), w, d or w, stiff=True)

    def plain(self, name, slot, rings, closed=False):
        """A part whose UVs are a box projection in meters: iron, and the end-on pieces of a barrel's head."""
        f0 = len(self.faces)
        self.chain(name, slot, rings, closed)
        faces = self.faces[f0:]
        self.uvs.extend(_box_uvs_lists(self.verts, faces))
        tone = self.rng.random()
        self.tones.extend([tone] * len(faces))


# EVERY END IS IN A JOINT. A butted end stops at its neighbor's face, and its 4 mm chamfer then reads as a groove all round the joint,
# so a butted end runs this far on into the neighbor: past its chamfer and a framing member's millimeter of bow. A lapped board is let this far into what carries it, which
# keeps them touching where the faces would otherwise only be coplanar until a board's twist or a worn corner lifts one: 1 mm lost one
# shelf board in 126 builds at full wear.
_PROP_SEAT = 0.008
_PROP_BEAR = 0.003


def _lay_brace(c, name, x0, x1, z0, z1, y, inward, w, t, step=None):
    """A diagonal laid on a frame's face, sawn to fit, every corner of it inside x0..x1 and z0..z1. Cut square it stuck a corner out
    past the frame whichever way it ran. A steep one is sawn level and a shallow one plumb, as a carpenter would, on the frame."""
    dx, dz = x1 - x0, z1 - z0
    run_x, run_z = dx, dz
    level = dz >= dx
    for _ in range(4):
        ang = math.atan2(run_z, run_x)
        if level:
            run_x = max(0.2 * dx, dx - w / max(math.sin(ang), 0.05))
        else:
            run_z = max(0.2 * dz, dz - w / max(math.cos(ang), 0.05))
    mx, mz = 0.5 * (x0 + x1), 0.5 * (z0 + z1)
    a, b = Vector((mx - 0.5 * run_x, y, mz - 0.5 * run_z)), Vector((mx + 0.5 * run_x, y, mz + 0.5 * run_z))
    if c.rng.random() < 0.5:
        a, b = Vector((mx + 0.5 * run_x, y, mz - 0.5 * run_z)), Vector((mx - 0.5 * run_x, y, mz + 0.5 * run_z))
    cut = (0.0, 0.0, 1.0) if level else (1.0, 0.0, 0.0)
    c.board(name, a, b, Vector((0.0, 1.0, 0.0)).cross(b - a).normalized(), w, t, step=step, stiff=True,
            sag=(0.0, inward, 0.0), cut0=cut, cut1=cut)


def _prop_crate(c, W, D, H, lid):
    """Slats on all four sides over corner posts, a diagonal brace on the long sides, a floor, and a lid lying on the top slats with
    its battens inside. A station every 35 cm rather than 22, and plain posts, battens and, under a lid, floor, all seen little or not at all: 4,144 triangles fail gameReady at 70 cm, and a
    70 cm board bows as well on two stations as on three."""
    rng = c.rng
    c.step = 0.35
    t = 0.016
    post = max(0.03, 0.06 * min(W, D, 1.0))
    hx, hy = 0.5 * W, 0.5 * D
    # The walls run floor to lid. They stopped 3.6 cm under it, on battens that stood on the posts, and the light came in all round
    top = H - (t if lid else 0.0)
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            # inside the slats, seen through their gaps at most
            c.board("post", (sx * (hx - t - 0.5 * post + _PROP_BEAR), sy * (hy - t - 0.5 * post + _PROP_BEAR), 0.0),
                    (sx * (hx - t - 0.5 * post + _PROP_BEAR), sy * (hy - t - 0.5 * post + _PROP_BEAR), top + (_PROP_SEAT if lid else 0.0)),
                    (1.0, 0.0, 0.0), post, post, plain=True, stiff=True)
    n = max(2, int(round(top / 0.115)))
    between = 0.012
    slat = (top - (n - 1) * between) / n
    for k in range(n):
        z = (k + 0.5) * slat + k * between
        edge = k in (0, n - 1)      # the first stands on the ground and the last carries the lid: neither is a hair narrow
        up = 1.0 if edge else 0.0
        for sy in (-1.0, 1.0):
            y = sy * (hy - 0.5 * t)
            c.board("slat", (-hx, y, z), (hx, y, z), (0.0, 0.0, 1.0), slat * (1.0 if edge else rng.uniform(0.96, 1.0)), t,
                    sag=(0.0, -sy, up), gap=between)
        for sx in (-1.0, 1.0):
            x = sx * (hx - 0.5 * t)
            c.board("slat", (x, -hy + t - _PROP_SEAT, z), (x, hy - t + _PROP_SEAT, z), (0.0, 0.0, 1.0),
                    slat * (1.0 if edge else rng.uniform(0.96, 1.0)), t, sag=(-sx, 0.0, up), gap=between)
    # the brace laid on the outside of the slats, corner to corner
    for sy in (-1.0, 1.0):
        _lay_brace(c, "brace", -hx + 0.004, hx - 0.004, 0.004, top - 0.004, sy * (hy + 0.5 * t - 0.002), -sy, 0.05, t, step=0.5)
    boards = max(2, int(D / 0.11))
    for k in range(boards):
        y = -hy + t + (k + 0.5) * (D - 2.0 * t) / boards
        c.board("floor", (-hx + t - _PROP_SEAT, y, 0.5 * t), (hx - t + _PROP_SEAT, y, 0.5 * t), (0.0, 1.0, 0.0),
                (D - 2.0 * t) / boards - 0.004, t, plain=lid, stiff=True, sag=(0.0, 0.0, 1.0))
    if lid:
        boards = max(2, int(D / 0.11))
        for k in range(boards):
            y = -hy + (k + 0.5) * D / boards
            c.board("lid", (-hx, y, H - 0.5 * t), (hx, y, H - 0.5 * t), (0.0, 1.0, 0.0), D / boards - 0.005, t, sag=(0.0, 0.0, -1.0),
                    gap=0.005)
        # under the lid and inside the walls, beside the posts, where they hold it square
        for sx in (-1.0, 1.0):
            x = sx * (hx - t - post - 0.027)
            c.board("batten", (x, -hy + t - _PROP_SEAT, top - 0.011 + _PROP_BEAR), (x, hy - t + _PROP_SEAT, top - 0.011 + _PROP_BEAR), (1.0, 0.0, 0.0),
                    0.05, 0.022, plain=True, stiff=True)


def _prop_pallet(c, W, D, H):
    """A deck of boards on three stringer boards over nine blocks, and three bottom boards, the outer ones flush with the edge."""
    t = 0.022
    block = max(0.0, H - 3.0 * t)
    if block < 0.02:
        t = H / 4.0
        block = H - 3.0 * t
    hx, hy = 0.5 * W, 0.5 * D
    bw = min(0.14, 0.18 * D)
    wide_b = 0.72 * bw
    rows = (-hy + 0.5 * wide_b, 0.0, hy - 0.5 * wide_b)
    for y in rows:
        c.board("bottom", (-hx, y, 0.5 * t), (hx, y, 0.5 * t), (0.0, 1.0, 0.0), wide_b, t, sag=(0.0, 0.0, 1.0))
    bl = min(0.14, 0.14 * W)
    for x in (-hx + 0.5 * bl, 0.0, hx - 0.5 * bl):
        for y in rows:
            # a hair narrower than the stringer, whose worn sides would otherwise leave the block's shoulder standing proud of them
            c.board("block", (x, y, t - _PROP_SEAT), (x, y, t + block + _PROP_BEAR + _PROP_SEAT), (1.0, 0.0, 0.0), bl - 0.006, wide_b, stiff=True)
        c.board("stringer", (x, -hy, t + block + 0.5 * t + _PROP_BEAR), (x, hy, t + block + 0.5 * t + _PROP_BEAR), (1.0, 0.0, 0.0), bl, t,
                stiff=True, sag=(0.0, 0.0, 1.0))
    n = max(3, int(D / 0.125))
    wide = D / (2.0 * n - 1.0) * 1.35
    pitch = (D - wide) / (n - 1)
    for k in range(n):
        y = -hy + 0.5 * wide + k * pitch
        z = t + block + t + 0.5 * t
        c.board("deck", (-hx, y, z), (hx, y, z), (0.0, 1.0, 0.0), wide * c.rng.uniform(0.94, 1.0), t, sag=(0.0, 0.0, -1.0),
                gap=pitch - wide)


def _prop_barrel(c, W, H, slot_iron=1):
    """Staves in a bulge, closed heads set in below the rim, and iron hoops driven on."""
    rng = c.rng
    R_mid = 0.5 * W
    R_end = R_mid * 0.8
    wall = max(0.016, 0.05 * R_mid)

    def radius(z):
        u = 2.0 * z / H - 1.0
        return R_end + (R_mid - R_end) * (1.0 - u * u)

    # Staves 9.4 cm wide and 6 levels up them at the default size, where 8.5 and 10 failed gameReady at 4,200 triangles: the
    # bulge's chord over a level is 1.7 mm against a 3 mm hoop, and a stave is the same width to the eye.
    n = max(12, int(2.0 * math.pi * R_mid / 0.094))
    gap = math.radians(0.25)
    levels = [H * k / 6 for k in range(7)]
    for k in range(n):
        a0 = 2.0 * math.pi * k / n + gap
        a1 = 2.0 * math.pi * (k + 1) / n - gap
        am = 0.5 * (a0 + a1)
        proud = rng.uniform(-0.0008, 0.0012) * (0.5 + c.wear)
        rings = []
        for z in levels:
            r = radius(z) + proud
            ri = r - wall
            rings.append([(r * math.cos(a), r * math.sin(a), z) for a in (a0, am, a1)]
                         + [(ri * math.cos(a), ri * math.sin(a), z) for a in (a1, am, a0)])
        f0 = len(c.faces)
        c.chain("stave", 0, rings)
        # the stave's grain runs up it
        c._local_uvs(f0, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (-math.sin(am), math.cos(am), 0.0),
                     (math.cos(am), math.sin(am), 0.0))
    # THE HEADS: boards side by side, cut to the circle, set in the croze
    inset = 0.03
    thick = 0.022
    boards = 5
    for z0 in (inset, H - inset - thick):
        rh = radius(z0 + 0.5 * thick) - wall + 0.002
        for b in range(boards):
            x0, x1 = -rh + 2.0 * rh * b / boards, -rh + 2.0 * rh * (b + 1) / boards - 0.002
            xs = [x0 + (x1 - x0) * i / 2.0 for i in range(3)]
            rings = []
            for x in xs:
                y = math.sqrt(max(rh * rh - x * x, 1e-5))
                rings.append([(x, -y, z0), (x, y, z0), (x, y, z0 + thick), (x, -y, z0 + thick)])
            f0 = len(c.faces)
            c.chain("head", 0, rings)
            c._local_uvs(f0, (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    # THE HOOPS: flat iron bands round the staves, wider at the bilge
    segs = 20 if c.low else 24
    # A hoop sits on the proudest stave. At 3 mm a stave wear had pushed 1.8 mm proud came up through the iron beside each hoop's flat
    iron = 0.003 + 0.0012 * c.wear
    for z, band in ((0.04 * H, 0.03), (0.2 * H, 0.04), (0.8 * H, 0.04), (0.96 * H, 0.03)):
        rings = []
        for k in range(segs):
            a = 2.0 * math.pi * k / segs
            ca, sa = math.cos(a), math.sin(a)
            r0, r1 = radius(z - 0.5 * band), radius(z + 0.5 * band)
            rings.append([((r0 + 0.0002) * ca, (r0 + 0.0002) * sa, z - 0.5 * band),
                          ((r0 + iron) * ca, (r0 + iron) * sa, z - 0.5 * band),
                          ((r1 + iron) * ca, (r1 + iron) * sa, z + 0.5 * band),
                          ((r1 + 0.0002) * ca, (r1 + 0.0002) * sa, z + 0.5 * band)])
        c.plain("hoop", slot_iron, rings, closed=True)


def _prop_table(c, W, D, H):
    """A plank top on four legs, with an apron let in between them."""
    t = 0.035
    leg = max(0.045, 0.075 * min(1.0, H / 0.75))
    hx, hy = 0.5 * W, 0.5 * D
    n = max(3, int(D / 0.15))
    for k in range(n):
        y = -hy + (k + 0.5) * D / n
        c.board("top", (-hx, y, H - 0.5 * t), (hx, y, H - 0.5 * t), (0.0, 1.0, 0.0),
                D / n - 0.003, t, step=0.3, sag=(0.0, 0.0, -1.0), gap=0.003)
    inset = 0.06
    lx, ly = hx - inset - 0.5 * leg, hy - inset - 0.5 * leg
    under = H - t
    apron = min(0.09, 0.2 * under)
    mid = 0.5 * leg - _PROP_SEAT
    zmid = under - 0.5 * apron + _PROP_BEAR
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            c.post("leg", sx * lx, sy * ly, 0.0, under + _PROP_SEAT, leg)
        c.board("apron", (sx * lx, -ly + mid, zmid), (sx * lx, ly - mid, zmid), (0.0, 0.0, 1.0), apron, 0.022, stiff=True, sag=(0.0, 0.0, 1.0))
    for sy in (-1.0, 1.0):
        c.board("apron", (-lx + mid, sy * ly, zmid), (lx - mid, sy * ly, zmid), (0.0, 0.0, 1.0), apron, 0.022, stiff=True, sag=(0.0, 0.0, 1.0))


def _prop_bench(c, W, D, H):
    """Seat planks on two splayed leg frames, tied by a stretcher let into their cross rails."""
    t = 0.032
    hx, hy = 0.5 * W, 0.5 * D
    n = max(2, int(D / 0.11))
    for k in range(n):
        y = -hy + (k + 0.5) * D / n
        c.board("seat", (-hx, y, H - 0.5 * t), (hx, y, H - 0.5 * t), (0.0, 1.0, 0.0), D / n - 0.004, t, step=0.3,
                sag=(0.0, 0.0, -1.0), gap=0.004)
    under = H - t - 0.03
    leg = 0.05
    level = (0.0, 0.0, 1.0)
    for sx in (-1.0, 1.0):
        x = sx * (hx - 0.14)
        # the rail stops 5 mm past the raked leg's outer face: it ran on 4 cm past it to nothing, only the seat over it
        reach = hy - 0.5 * leg - 0.05 + 0.015 + 0.005
        c.board("rail", (x, -reach, under + 0.015 + _PROP_BEAR), (x, reach, under + 0.015 + _PROP_BEAR), (1.0, 0.0, 0.0), leg, 0.03,
                stiff=True, sag=(0.0, 0.0, 1.0))
        for sy in (-1.0, 1.0):
            foot = sy * (hy - 0.5 * leg - 0.005)
            head = sy * (hy - 0.5 * leg - 0.05)
            # sawn level at both ends: square to its rake it stood on one edge and met the rail on the other
            c.board("leg", (x, foot, 0.0), (x, head, under + _PROP_BEAR + _PROP_SEAT), (1.0, 0.0, 0.0), leg, 0.03, stiff=True, cut0=level, cut1=level)
    # to the middle of the cross rail, and no taller than it: it stopped 14 mm short of both, touching nothing
    c.board("stretcher", (-hx + 0.14, 0.0, 0.42 * under), (hx - 0.14, 0.0, 0.42 * under), (0.0, 0.0, 1.0), 0.045, 0.022, stiff=True)
    for sx in (-1.0, 1.0):
        x = sx * (hx - 0.14)
        c.board("cross", (x, -hy + 0.05, 0.42 * under), (x, hy - 0.05, 0.42 * under), (0.0, 0.0, 1.0), 0.05, 0.022, stiff=True)


def _prop_chair(c, W, D, H):
    """A slatted seat, back legs running up into the back's rails and slats, and stretchers between the legs."""
    seat_z = min(0.46, 0.52 * H)
    t = 0.022
    leg = 0.035
    hx, hy = 0.5 * W, 0.5 * D
    lx, ly = hx - 0.5 * leg, hy - 0.5 * leg
    frame = seat_z - t
    lean = 0.05
    mid = 0.5 * leg - _PROP_SEAT
    level = (0.0, 0.0, 1.0)
    # back legs: straight to the seat, then raked back, each one board
    for sx in (-1.0, 1.0):
        c.post("front leg", sx * lx, ly, 0.0, frame + _PROP_SEAT, leg)
        c.board("back leg", (sx * lx, -ly + 0.015, 0.0), (sx * lx, -ly, frame + _PROP_SEAT), (1.0, 0.0, 0.0), leg, leg, stiff=True, cut0=level)
        # let down into the leg, the leg being the wider, so the seam between them is a shoulder rather than two chamfers
        c.board("back post", (sx * lx, -ly + lean * 0.008 / max(H - frame, 1e-3), frame - 0.008), (sx * lx, -ly - lean, H), (1.0, 0.0, 0.0),
                leg, 0.03, stiff=True)
        c.board("side rail", (sx * lx, -ly + mid, frame - 0.03 + _PROP_BEAR), (sx * lx, ly - mid, frame - 0.03 + _PROP_BEAR),
                (0.0, 0.0, 1.0), 0.06, 0.02, stiff=True, sag=(0.0, 0.0, 1.0))
        c.board("side stretcher", (sx * lx, -ly + mid, 0.17), (sx * lx, ly - mid, 0.17), (0.0, 0.0, 1.0), 0.025, 0.018, stiff=True)
    for y, w in ((ly, 0.06), (-ly, 0.06)):
        c.board("rail", (-lx + mid, y, frame - 0.03 + _PROP_BEAR), (lx - mid, y, frame - 0.03 + _PROP_BEAR), (0.0, 0.0, 1.0), w, 0.02, stiff=True,
                sag=(0.0, 0.0, 1.0))
    # between the two side stretchers, at their height, let into them: higher, it was a rod floating free of everything
    c.board("stretcher", (-lx, 0.0, 0.17), (lx, 0.0, 0.17), (0.0, 1.0, 0.0), 0.025, 0.014, stiff=True)
    n = max(3, int(D / 0.085))
    for k in range(n):
        y = -hy + (k + 0.5) * D / n
        c.board("seat", (-hx, y, seat_z - 0.5 * t), (hx, y, seat_z - 0.5 * t), (0.0, 1.0, 0.0), D / n - 0.005, t, sag=(0.0, 0.0, -1.0),
                gap=0.005)
    # the back: a top rail, a lower rail, slats between
    z_top = H - 0.04
    z_low = seat_z + 0.5 * (H - seat_z) * 0.55
    along = (0.0, -lean / max(H - frame, 1e-3), 1.0)

    def back_y(z):
        return -ly - lean * (z - frame) / (H - frame)

    c.board("top rail", (-lx + mid, back_y(z_top), z_top), (lx - mid, back_y(z_top), z_top), along, 0.08, 0.02, stiff=True)
    c.board("low rail", (-lx + mid, back_y(z_low), z_low), (lx - mid, back_y(z_low), z_low), along, 0.05, 0.02, stiff=True)
    m = 3
    for k in range(m):
        x = (-lx + 0.5 * leg) + (k + 0.5) * (2.0 * lx - leg) / m
        z0, z1 = z_low + 0.025 - _PROP_SEAT, z_top - 0.04 + _PROP_SEAT
        c.board("back slat", (x, back_y(z0), z0), (x, back_y(z1), z1), (1.0, 0.0, 0.0), 0.05, 0.012, stiff=True)


def _prop_shelf(c, W, D, H):
    """Two ladder frames of uprights and cross rails, shelves of boards lying on the rails, and a diagonal let on across the back."""
    up = 0.045
    hx, hy = 0.5 * W, 0.5 * D
    levels = max(3, int(H / 0.45))
    for sx in (-1.0, 1.0):
        x = sx * (hx - 0.5 * up)
        for sy in (-1.0, 1.0):
            c.post("upright", x, sy * (hy - 0.5 * up), 0.0, H, up)
    t = 0.022
    rail_t = 0.02
    for k in range(levels):
        # the top shelf lies level with the uprights' tops
        z = 0.1 + (H - 0.1 - t) * k / (levels - 1)
        for sx in (-1.0, 1.0):
            x = sx * (hx - 0.5 * up)
            c.board("rail", (x, -hy + up - _PROP_SEAT, z - 0.03 + _PROP_BEAR), (x, hy - up + _PROP_SEAT, z - 0.03 + _PROP_BEAR), (0.0, 0.0, 1.0),
                    0.06, rail_t, stiff=True, sag=(0.0, 0.0, 1.0))
        n = max(2, int(D / 0.11))
        # A board ends over its rail. It stopped at the uprights' inner face, 12.5 mm short of the rail and resting on nothing
        reach = hx - 0.5 * up + 0.5 * rail_t - 0.002
        for j in range(n):
            y = -hy + up + (j + 0.5) * (D - 2.0 * up) / n
            c.board("shelf", (-reach, y, z + 0.5 * t), (reach, y, z + 0.5 * t), (0.0, 1.0, 0.0),
                    (D - 2.0 * up) / n - 0.004, t, sag=(0.0, 0.0, -1.0), gap=0.004)
    # a diagonal across the back, so it stands: sawn to lie on the uprights' back faces at both ends. It spans bottom shelf to top shelf,
    # whose boards hide where it leaves the upright: from 5 cm up it tapered out to a point in the open, beside the foot
    t_b = 0.012
    _lay_brace(c, "brace", -hx + 0.002, hx - 0.002, 0.1, H - 0.002, -hy - 0.5 * t_b + 0.002, 1.0, 0.04, t_b)


def _wood_material(name, color, color2, wear=0.0):
    """Grain drawn out along the UVs' U, in the board's own meters, between two tones, each board shifted by vertex color R and,
    at any wear, grayed toward old timber by how much sun vertex color G says it took. Wear 0 has no fading node at all."""
    from .ops_nature import _math, _mix_color, _noise   # noqa: PLC0415 - ops_nature's node helpers
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = _find_type(nt.nodes, "BSDF_PRINCIPLED")
    uv = nt.nodes.new("ShaderNodeUVMap")
    uv.uv_map = "UVMap"
    mapping = nt.nodes.new("ShaderNodeMapping")
    nt.links.new(uv.outputs[0], _sock(mapping.inputs, "Vector"))
    _sock(mapping.inputs, "Scale").default_value = (1.0, 22.0, 1.0)
    grain_uv = _sock(mapping.outputs, "Vector")
    broad = _noise(nt, grain_uv, 3.0, 5.0, 0.6)
    fine = _noise(nt, grain_uv, 22.0, 4.0, 0.55)
    grain = _math(nt, "ADD", _math(nt, "MULTIPLY", broad, 0.7), _math(nt, "MULTIPLY", fine, 0.3))
    grain = _math(nt, "SUBTRACT", _math(nt, "MULTIPLY", grain, 3.2), 1.1, clamp=True)
    col = _mix_color(nt, grain, tuple(color), tuple(color2))
    attr = nt.nodes.new("ShaderNodeVertexColor")
    attr.layer_name = "Col"
    # Wear widens the boards' spread: boards from one prop age apart, and at 0.45 they read as one plank
    spread = 0.45 + 0.75 * wear
    tone = _math(nt, "ADD", _math(nt, "MULTIPLY", _sock(attr.outputs, "Color"), spread), 1.005 - 0.5 * spread)
    col = _mix_color(nt, 1.0, col, _mix_color(nt, tone, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), blend="MULTIPLY")
    if wear > 0.0:
        split = nt.nodes.new("ShaderNodeSeparateColor")
        nt.links.new(_sock(attr.outputs, "Color"), split.inputs[0])
        sun = _math(nt, "MULTIPLY", _math(nt, "ADD", _math(nt, "MULTIPLY", split.outputs[1], 0.8), 0.2), min(1.0, wear), clamp=True)
        old = nt.nodes.new("ShaderNodeHueSaturation")
        _sock(old.inputs, "Saturation").default_value = 0.42
        _sock(old.inputs, "Value").default_value = 1.45
        nt.links.new(col, _sock(old.inputs, "Color"))
        col = _mix_color(nt, sun, col, _sock(old.outputs, "Color"))
    nt.links.new(col, _sock(bsdf.inputs, "Base Color"))
    _sock(bsdf.inputs, "Roughness").default_value = 0.72
    bump = nt.nodes.new("ShaderNodeBump")
    _sock(bump.inputs, "Strength").default_value = 0.4
    _sock(bump.inputs, "Distance").default_value = 0.002
    nt.links.new(fine, _sock(bump.inputs, "Height"))
    nt.links.new(_sock(bump.outputs, "Normal"), _sock(bsdf.inputs, "Normal"))
    return mat


def _weather_prop(mat, wear, seed, iron=False):
    """What `wear` means on the material: set_material_properties' own weathering, its amounts rising with wear. Edges rub back, corners
    and the joints between boards collect grime. The chips are object-space noise, so a seed moves them and the geometry stays the same."""
    from .ops_material import op_set_material_properties   # noqa: PLC0415
    op_set_material_properties({"material": mat.name, "weathering": {
        "edgeWear": {"amount": 0.35 + 0.5 * wear, "width": 0.004 + 0.008 * wear, "bump": 0.3},
        "grime": {"amount": 0.3 + 0.55 * wear, "distance": 0.045, "color": [0.022, 0.016, 0.011], "roughness": 0.92},
        "colorVariation": 0.04 + 0.07 * wear, "roughnessVariation": 0.1 + 0.15 * wear,
        "scale": 3.0 if iron else 9.0, "seed": float(seed % 1000)}})


def op_build_prop(params):
    """Build one wooden dressing prop as carpentry: a crate, a pallet, a barrel, a table, a bench, a chair or a shelf unit - ONE
    mesh of separate boards, each a closed solid with eased and chamfered edges, a slight bow and corners worn by `wear`, its
    grain along the board (UV0 in meters, U along the grain, each board starting elsewhere on the texture), with vertex color
    a tone per board in R and how faded it is in G. A barrel's staves, heads and iron hoops have their own parts, in a second slot. The pivot is at the bottom
    center, Z up; the object is in meters at the size asked. The same seed and parameters give the same prop.
    Wear shows in the material: the wood's boards fade and part in tone, and set_material_properties' weathering rubs the edges back and
    darkens corners and joints. That is a Cycles node chain, so an engine gets it through bake_texture_set with uvLayer LightmapUV and
    bakedMaterial; the baked material samples that layer, UV channel 1.

    params:
      kind (str)         crate (default), pallet, barrel, table, bench, chair, shelf
      width, depth, height (cm)   each kind's common size by default: crate 70x50x50, pallet 120x80x14.4, barrel 60x60x88 (width
                         is the bilge; depth is ignored and refused), table 150x80x75, bench 140x34x45, chair 44x46x90,
                         shelf 100x36x180; up to 400, and no smaller than the kind can be built
      seed (int)         0 or more, default 1: which bows, wear and tones
      wear (0-1)         how worn the prop is: corners, faded and parted boards, rubbed edges, grime, default 0.35; 0 is fresh-planed
                         with no weathering in the material; the triangles are the same at any wear
      detail (str)       full (default: two-segment eased edges and a station every 22 cm, which carry the wear and the bow) or low (one
                         segment, no station between the ends: about a third of the triangles, for a mesh without Nanite)
      lid (bool)         a crate's battened lid, default true
      color, color2 ([r,g,b] linear 0-1)   the wood's two tones
      metalColor ([r,g,b])   a barrel's hoops
      collision (bool)   a UCX_<name>_00 convex hull round the whole prop, default true
      name               default SM_<Kind>_01a, the first number free
      location, rotation, collection, takeOverName
    """
    tail = "NOTHING was created."
    reject_unknown(params, _PROP_KEYS, "build_prop", _NOTES_PROP)
    started = time.perf_counter()
    kind = take(params, "kind", default="crate", kind=str)
    if kind not in _PROP_KINDS:
        raise MifOpError("kind must be one of %s, got '%s'. %s" % (", ".join(_PROP_KINDS), kind, tail))
    if kind == "barrel" and params.get("depth") is not None:
        raise MifOpError("a barrel is round: width is its bilge and it takes no depth. %s" % tail)
    if params.get("lid") is not None and kind != "crate":
        raise MifOpError("lid closes a crate; kind is '%s'. %s" % (kind, tail))
    if params.get("metalColor") is not None and kind != "barrel":
        raise MifOpError("metalColor is a barrel's hoops; a %s has no iron. %s" % (kind, tail))
    dw, dd, dh = _PROP_SIZES[kind]
    lo_w, lo_d, lo_h = _PROP_LIMITS[kind]
    width = take_float(params, "width", default=dw)
    depth = take_float(params, "depth", default=width if kind == "barrel" else dd)
    height = take_float(params, "height", default=dh)
    for key, val, lo in (("width", width, lo_w), ("depth", depth, lo_d), ("height", height, lo_h)):
        if not lo <= val <= 400.0:
            raise MifOpError("%s is %g cm; a %s here is %g to 400 cm. %s" % (key, val, kind, lo, tail))
    seed = take_int(params, "seed", default=1)
    if seed < 0:
        raise MifOpError("seed must be 0 or more, got %d. %s" % (seed, tail))
    wear = take_float(params, "wear", default=0.35)
    if not 0.0 <= wear <= 1.0:
        raise MifOpError("wear is %g; it is 0 (fresh-planed) to 1. %s" % (wear, tail))
    lid = take_bool(params, "lid", default=True)
    detail = take(params, "detail", default="full", kind=str)
    if detail not in ("full", "low"):
        raise MifOpError("detail is full or low, got '%s'. %s" % (detail, tail))
    color = _color_or(params, "color", (0.20, 0.115, 0.055), tail)
    color2 = _color_or(params, "color2", (0.085, 0.045, 0.02), tail)
    metal = _color_or(params, "metalColor", (0.05, 0.045, 0.04), tail)
    want_hull = take_bool(params, "collision", default=True)
    label = kind.capitalize()
    default_name = "SM_%s_01a" % label
    k = 1
    taken = set(bpy.data.objects.keys())                    # once: a lookup per try scans the scene's objects
    while default_name in taken:
        k += 1
        default_name = "SM_%s_%02da" % (label, k)
    W, D, H = width / 100.0, depth / 100.0, height / 100.0
    c = _Carpentry(random.Random(seed), wear, detail == "low")
    if kind == "crate":
        _prop_crate(c, W, D, H, lid)
    elif kind == "pallet":
        _prop_pallet(c, W, D, H)
    elif kind == "barrel":
        _prop_barrel(c, W, H)
    elif kind == "table":
        _prop_table(c, W, D, H)
    elif kind == "bench":
        _prop_bench(c, W, D, H)
    elif kind == "chair":
        _prop_chair(c, W, D, H)
    else:
        _prop_shelf(c, W, D, H)
    # BOTTOM CENTER, at the size asked: a board's bow and the rack of a chair move the bounds a little off the nominal
    lo = [min(v[i] for v in c.verts) for i in range(3)]
    hi = [max(v[i] for v in c.verts) for i in range(3)]
    shift = (-0.5 * (lo[0] + hi[0]), -0.5 * (lo[1] + hi[1]), -lo[2])
    c.verts = [[v[0] + shift[0], v[1] + shift[1], v[2] + shift[2]] for v in c.verts]
    sub = {key: params[key] for key in ("name", "location", "rotation", "collection", "takeOverName") if key in params}
    out = _build_through_create_mesh(sub, c.verts, c.faces, c.uvs, False, default_name, "build_prop")
    obj = bpy.data.objects.get(out.get("name") or "")
    if obj is None or obj.type != "MESH":
        return out
    mesh = obj.data
    mesh.materials.append(_wood_material("M_%s_Wood" % obj.name, color, color2, wear))
    if 1 in c.slots:
        mesh.materials.append(_prop_material("M_%s_Iron" % obj.name, metal, 0.5, 0.85))
    if wear > 0.0:
        for mat in mesh.materials:
            _weather_prop(mat, wear, seed, "Iron" in mat.name)
    mesh.polygons.foreach_set("material_index", c.slots)
    mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
    _resharpen(mesh, 35.0)
    attr = mesh.color_attributes.new(name="Col", type="BYTE_COLOR", domain="CORNER")
    tones = []
    for tone, poly in zip(c.tones, mesh.polygons):
        # G, how much sun the board took, comes off its tone: a new draw on the seeded stream would move every board after it
        tones.extend((tone, (tone * 7.31) % 1.0, tone, 1.0) * poly.loop_total)
    attr.data.foreach_set("color_srgb", tones)
    mesh.update()
    op_uv_unwrap({"object": obj.name, "method": "SMART", "uvLayer": _LIGHTMAP_LAYER, "islandMargin": 0.02})
    mesh.uv_layers.active = mesh.uv_layers["UVMap"]
    hull = None
    if want_hull:
        from .ops_mesh import op_create_collision_hull   # noqa: PLC0415
        got = op_create_collision_hull({"object": obj.name, "maxVertices": 32})
        hull = got.get("object")
        made = bpy.data.objects.get(hull or "")
        if made is not None:
            made.display_type = "WIRE"
            made.hide_render = True
    counts = {}
    for name in c.names:
        counts[name] = counts.get(name, 0) + 1
    bounds_lo = [min(v[i] for v in c.verts) for i in range(3)]
    bounds_hi = [max(v[i] for v in c.verts) for i in range(3)]
    out.update({"kind": kind, "seed": seed, "wear": wear, "detail": detail, "parts": counts,
                "sizeCm": [round((b - a) * 100.0, 2) for a, b in zip(bounds_lo, bounds_hi)],
                "triangles": sum(len(f) - 2 for f in c.faces),
                "materialSlots": [m.name for m in mesh.materials], "uvLayers": [u.name for u in mesh.uv_layers],
                "colorAttribute": {"name": "Col", "domain": "CORNER", "R": "a tone per board", "G": "how faded the board is",
                                   "note": "the wood material reads both, and fading scales with wear"},
                "collisionHull": hull, "pivot": "the bottom center, at the object's origin, Z up",
                "elapsedSeconds": round(time.perf_counter() - started, 2)})
    return out


OPS = {
    "build_prop": op_build_prop,
    "build_vehicle_body": op_build_vehicle_body,
    "build_downspout": op_build_downspout,
    "build_wall_fixture": op_build_wall_fixture,
    "build_dumpster": op_build_dumpster,
    "build_lathe": op_build_lathe,
    "build_trash_bag": op_build_trash_bag,
    "build_tube": op_build_tube,
    "build_kit_piece": op_build_kit_piece,
    "set_vertex_color_by_rule": op_set_vertex_color_by_rule,
    "build_lod_chain": op_build_lod_chain,
}
