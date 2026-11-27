"""MifBlender ops: line rendering, a sun by place and time, and a playblast - set_freestyle, set_sun_position,
render_viewport (1.3 gap audit, 2026-10-19).

THE GAP. Freestyle - Blender's renderer-side line tool for technical drawings and toon outlines - had no route at all;
the bundled Sun Position add-on left Blender at 4.2, so a sun placed by latitude, longitude, date and time had no route
on any current build; and an animation review had only render_animation, minutes per frame where a blocking pass needs
seconds.
"""
from __future__ import annotations

import datetime
import math
import os
import re
import time

import bpy
import mathutils

from .ops_common import (MifOpError, check_output_path, finite_float, finite_floats, get_object, reject_unknown, remove_objects, rnd,
                         require_editable, take, take_bool, take_float, take_int)


# --------------------------------------------------------------------------------------------- set_freestyle

_FS_KEYS = ("enabled", "viewLayer", "lineThickness", "thicknessMode", "creaseAngle", "lineSets", "removeLineSets",
            "edgeMarks")
_EDGE_TYPES = {"SILHOUETTE": "select_silhouette", "BORDER": "select_border", "CREASE": "select_crease",
               "RIDGE_VALLEY": "select_ridge_valley", "SUGGESTIVE_CONTOUR": "select_suggestive_contour",
               "MATERIAL_BOUNDARY": "select_material_boundary", "CONTOUR": "select_contour",
               "EXTERNAL_CONTOUR": "select_external_contour", "EDGE_MARK": "select_edge_mark"}
_LS_KEYS = ("name", "edgeTypes", "collection", "color", "thickness", "alpha", "visibility", "excludeCollection",
            "edgeTypeNegation", "edgeTypeCombination", "faceMarks")
_FACEMARK_KEYS = ("negation", "condition")


def _lineset_row(ls):
    st = ls.linestyle
    return {"name": ls.name, "show": bool(ls.show_render),
            "edgeTypes": sorted(k for k, a in _EDGE_TYPES.items() if getattr(ls, a, False)),
            "collection": ls.collection.name if ls.select_by_collection and ls.collection else None,
            "collectionNegation": ls.collection_negation if ls.select_by_collection else None,
            "edgeTypeNegation": ls.edge_type_negation, "edgeTypeCombination": ls.edge_type_combination,
            "faceMarks": ({"negation": ls.face_mark_negation, "condition": ls.face_mark_condition}
                          if ls.select_by_face_marks else None),
            "visibility": ls.visibility,
            "color": [round(c, 6) for c in st.color] if st else None,
            "thickness": round(st.thickness, 6) if st else None, "alpha": round(st.alpha, 6) if st else None}


def _mark_edges_plan(spec, tail):
    """Check one edgeMarks entry {object, edges [indices] | sharp: true | seams: true, clear} - writes nothing.

    SPLIT FROM THE WRITE 2026-10-28, so set_freestyle checks every mark before its first write: the check ran after the
    scene switches and the line sets had been written, so a bad object or edge index refused with "NOTHING was
    changed" over a changed file.
    """
    if not isinstance(spec, dict) or not isinstance(spec.get("object"), str):
        raise MifOpError("edgeMarks is {object, edges [indices] | sharp: true | seams: true, clear}. %s" % tail)
    bad = sorted(set(spec) - {"object", "edges", "sharp", "seams", "clear"})
    if bad:
        raise MifOpError("edgeMarks has unknown key(s) %s. %s" % (bad, tail))
    for k in ("sharp", "seams", "clear"):
        if spec.get(k) is not None and not isinstance(spec[k], bool):
            raise MifOpError("edgeMarks.%s is true or false, got %r. %s" % (k, spec[k], tail))
    obj = get_object(spec["object"], want_mesh=True)
    require_editable(obj, "changed")
    me = obj.data
    idx = set()
    if spec.get("edges") is not None:
        e = spec["edges"]
        if not isinstance(e, list) or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(me.edges)
                                          for i in e):
            raise MifOpError("edgeMarks.edges are edge indices of '%s'. %s" % (obj.name, tail))
        idx |= set(e)
    if spec.get("sharp"):
        idx |= {e.index for e in me.edges if e.use_edge_sharp}
    if spec.get("seams"):
        idx |= {e.index for e in me.edges if e.use_seam}
    return obj, idx, bool(spec.get("clear", False))


def _mark_edges(plan):
    """Write one checked edgeMarks entry (_mark_edges_plan) and read the marks back."""
    obj, idx, clear = plan
    me = obj.data
    if "use_freestyle_mark" in bpy.types.MeshEdge.bl_rna.properties:
        for e in me.edges:
            if e.index in idx:
                e.use_freestyle_mark = not clear
        marked = sum(1 for e in me.edges if e.use_freestyle_mark)
    else:
        at = me.attributes.get("freestyle_edge") or me.attributes.new("freestyle_edge", "BOOLEAN", "EDGE")
        for i in idx:
            at.data[i].value = not clear
        marked = sum(1 for d in me.attributes["freestyle_edge"].data if d.value)
    return {"object": obj.name, "edgesChanged": len(idx), "edgesMarked": marked}


def _fs_enum(owner_type, prop, value, where, tail):
    """An enum value checked against this build's RNA items before any write."""
    items = [e.identifier for e in owner_type.bl_rna.properties[prop].enum_items]
    v = value.upper() if isinstance(value, str) else None
    if v not in items:
        raise MifOpError("%s is one of %s, got %r. %s" % (where, ", ".join(items), value, tail))
    return v


def op_set_freestyle(params):
    """Freestyle line rendering: the scene and view-layer switches, line thickness, line sets by edge type and
    collection with their color, thickness and alpha, and Freestyle edge marks on meshes - read back.

    params:
      enabled (bool)           scene.render.use_freestyle (and the view layer's use_freestyle)
      viewLayer (str)          default the active view layer
      lineThickness (float)    scene line thickness; thicknessMode ABSOLUTE (pixels, default) | RELATIVE (to 480 px)
      creaseAngle (float)      degrees - an edge sharper than this is a CREASE
      lineSets (list)          [{name, edgeTypes [SILHOUETTE | BORDER | CREASE | RIDGE_VALLEY | SUGGESTIVE_CONTOUR |
                               MATERIAL_BOUNDARY | CONTOUR | EXTERNAL_CONTOUR | EDGE_MARK], collection, color [r,g,b],
                               thickness, alpha, visibility VISIBLE | HIDDEN | RANGE, excludeCollection (ink all but
                               it; not with collection), edgeTypeNegation INCLUSIVE | EXCLUSIVE, edgeTypeCombination OR |
                               AND, faceMarks true | {negation, condition ONE | BOTH}}] - a line set of that name is
                               edited, a new one made
      removeLineSets (list)    names
      edgeMarks (dict | list)  {object, edges [indices] | sharp: true | seams: true, clear}
    Freestyle draws in Cycles and EEVEE renders (not Workbench). Measured present on 3.6 - 5.2 (ViewLayer
    freestyle_settings, RenderSettings.use_freestyle).
    """
    reject_unknown(params, _FS_KEYS, "set_freestyle")
    tail = "NOTHING was changed."
    sc = bpy.context.scene
    vl_name = take(params, "viewLayer", default=None, kind=str)
    vl = sc.view_layers.get(vl_name) if vl_name else bpy.context.view_layer
    if vl is None:
        raise MifOpError("scene '%s' has no view layer '%s'. %s" % (sc.name, vl_name, tail))
    fs = vl.freestyle_settings
    sets = params.get("lineSets")
    if sets is not None:
        if not isinstance(sets, list):
            raise MifOpError("lineSets is a list of {name, edgeTypes, ...}. %s" % tail)
        for i, s in enumerate(sets):
            if not isinstance(s, dict) or not isinstance(s.get("name"), str):
                raise MifOpError("lineSets[%d] needs a name. %s" % (i, tail))
            bad = sorted(set(s) - set(_LS_KEYS))
            if bad:
                raise MifOpError("lineSets[%d] has unknown key(s) %s - accepted: %s. %s" % (i, bad, ", ".join(_LS_KEYS),
                                                                                         tail))
            et = s.get("edgeTypes")
            if et is not None and (not isinstance(et, list) or any(str(x).upper() not in _EDGE_TYPES for x in et)):
                raise MifOpError("lineSets[%d].edgeTypes is a list of %s. %s" % (i, ", ".join(_EDGE_TYPES), tail))
            if s.get("collection") is not None and s.get("excludeCollection") is not None:
                raise MifOpError("lineSets[%d] takes collection (ink only it) OR excludeCollection (ink all but it) - a "
                                 "line set selects by one collection. %s" % (i, tail))
            for ck in ("collection", "excludeCollection"):
                if s.get(ck) is not None and bpy.data.collections.get(s[ck]) is None:
                    raise MifOpError("no collection named '%s'. %s" % (s[ck], tail))
    rm = params.get("removeLineSets")
    if rm is not None:
        if not isinstance(rm, list) or any(not isinstance(n, str) for n in rm):
            raise MifOpError("removeLineSets is a list of line set names. %s" % tail)
        missing = [n for n in rm if fs.linesets.get(n) is None]
        if missing:
            raise MifOpError("removeLineSets names line sets the view layer does not have: %s. %s" % (missing, tail))
    # EVERY VALUE IS CHECKED BEFORE THE FIRST WRITE (2026-10-28). thicknessMode went straight into an RNA enum and the
    # line sets' visibility, thickness and alpha through float() / setattr after use_freestyle and linesets.new had
    # run, so a bad value raised a raw TypeError out of a half-written call - found by the version matrix's
    # corrupted-value pass on thicknessMode.
    on = take_bool(params, "enabled") if params.get("enabled") is not None else None
    thick = take_float(params, "lineThickness") if params.get("lineThickness") is not None else None
    tmode = (_fs_enum(bpy.types.RenderSettings, "line_thickness_mode", params["thicknessMode"], "thicknessMode", tail)
             if params.get("thicknessMode") is not None else None)
    crease = take_float(params, "creaseAngle") if params.get("creaseAngle") is not None else None
    marks = params.get("edgeMarks")
    plans = []
    for i, s in enumerate(sets or []):
        where = "lineSets[%d]" % i
        row = {}
        if s.get("visibility") is not None:
            row["visibility"] = _fs_enum(bpy.types.FreestyleLineSet, "visibility", s["visibility"],
                                         where + ".visibility", tail)
        if s.get("color") is not None:
            if not isinstance(s["color"], list) or len(s["color"]) < 3:
                raise MifOpError("%s.color is [r, g, b], got %r. %s" % (where, s["color"], tail))
            row["color"] = finite_floats(list(s["color"])[:3], where + ".color")
        for k in ("thickness", "alpha"):
            if s.get(k) is not None:
                row[k] = finite_float(s[k], "%s.%s" % (where, k))
        # THE OTHER SELECTIONS A LINE SET OFFERS (FOUND 2026-11-10: 19 detail meshes could not be left un-inked except by
        # building a collection of the 67 that should be). Each value is checked against this build's RNA.
        for key, prop in (("edgeTypeNegation", "edge_type_negation"), ("edgeTypeCombination", "edge_type_combination")):
            if s.get(key) is not None:
                row[key] = _fs_enum(bpy.types.FreestyleLineSet, prop, s[key], "%s.%s" % (where, key), tail)
        fm = s.get("faceMarks")
        if fm is not None:
            if fm is True:
                fm = {}
            if not isinstance(fm, dict) or set(fm) - set(_FACEMARK_KEYS):
                raise MifOpError("%s.faceMarks is true or {negation INCLUSIVE | EXCLUSIVE, condition ONE | BOTH}, got "
                                 "%r. %s" % (where, s["faceMarks"], tail))
            row["faceMarks"] = {
                "negation": _fs_enum(bpy.types.FreestyleLineSet, "face_mark_negation", fm.get("negation", "INCLUSIVE"),
                                     where + ".faceMarks.negation", tail),
                "condition": _fs_enum(bpy.types.FreestyleLineSet, "face_mark_condition", fm.get("condition", "ONE"),
                                      where + ".faceMarks.condition", tail)}
        plans.append(row)
    mark_plans = [_mark_edges_plan(m, tail) for m in (marks if isinstance(marks, list) else [marks])]         if marks is not None else None
    if on is not None:
        sc.render.use_freestyle = on
        vl.use_freestyle = on
    if thick is not None:
        sc.render.line_thickness = thick
    if tmode is not None:
        sc.render.line_thickness_mode = tmode
    if crease is not None:
        fs.crease_angle = math.radians(crease)
    for n in rm or []:
        fs.linesets.remove(fs.linesets[n])
    for s, row in zip(sets or [], plans):
        ls = fs.linesets.get(s["name"]) or fs.linesets.new(s["name"])
        if s.get("edgeTypes") is not None:
            ls.select_by_edge_types = True
            want = {str(x).upper() for x in s["edgeTypes"]}
            for k, a in _EDGE_TYPES.items():
                setattr(ls, a, k in want)
        if s.get("collection") is not None:
            ls.select_by_collection = True
            ls.collection = bpy.data.collections[s["collection"]]
            ls.collection_negation = "INCLUSIVE"
        if s.get("excludeCollection") is not None:
            ls.select_by_collection = True
            ls.collection = bpy.data.collections[s["excludeCollection"]]
            ls.collection_negation = "EXCLUSIVE"
        if "edgeTypeNegation" in row:
            ls.edge_type_negation = row["edgeTypeNegation"]
        if "edgeTypeCombination" in row:
            ls.edge_type_combination = row["edgeTypeCombination"]
        if "faceMarks" in row:
            ls.select_by_face_marks = True
            ls.face_mark_negation = row["faceMarks"]["negation"]
            ls.face_mark_condition = row["faceMarks"]["condition"]
        if "visibility" in row:
            ls.select_by_visibility = True
            ls.visibility = row["visibility"]
        st = ls.linestyle
        if st is None:
            st = ls.linestyle = bpy.data.linestyles.new(s["name"])
        if "color" in row:
            st.color = row["color"]
        if "thickness" in row:
            st.thickness = row["thickness"]
        if "alpha" in row:
            st.alpha = row["alpha"]
    marked = [_mark_edges(pl) for pl in mark_plans] if mark_plans is not None else None
    return {"scene": sc.name, "viewLayer": vl.name, "enabled": bool(sc.render.use_freestyle),
            "viewLayerEnabled": bool(vl.use_freestyle), "lineThickness": round(sc.render.line_thickness, 6),
            "thicknessMode": sc.render.line_thickness_mode, "creaseAngle": round(math.degrees(fs.crease_angle), 4),
            "lineSets": [_lineset_row(ls) for ls in fs.linesets], "edgeMarks": marked,
            "engine": sc.render.engine,
            "engineNote": ("Freestyle draws in Cycles and EEVEE renders; this scene renders with Workbench, which draws no "
                           "lines" if sc.render.engine == "BLENDER_WORKBENCH" else None)}


# --------------------------------------------------------------------------------------------- set_sun_position

_SUN_KEYS = ("object", "latitude", "longitude", "utcOffset", "date", "time", "dst", "northOffset", "sky", "distance",
             "keyFrame", "strength")


def _julian_day(dt_utc):
    y, m = dt_utc.year, dt_utc.month
    d = dt_utc.day + (dt_utc.hour + dt_utc.minute / 60.0 + dt_utc.second / 3600.0) / 24.0
    if m <= 2:
        y, m = y - 1, m + 12
    a = y // 100
    b = 2 - a + a // 4
    return int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + b - 1524.5


def solar_position(lat, lon, dt_utc):
    """(azimuth from north, clockwise; elevation) in degrees - the NOAA solar position (Meeus), no refraction."""
    jc = (_julian_day(dt_utc) - 2451545.0) / 36525.0
    l0 = (280.46646 + jc * (36000.76983 + jc * 0.0003032)) % 360.0
    m = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    e = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)
    mr = math.radians(m)
    c = (math.sin(mr) * (1.914602 - jc * (0.004817 + 0.000014 * jc)) + math.sin(2 * mr) * (0.019993 - 0.000101 * jc)
         + math.sin(3 * mr) * 0.000289)
    true_long = l0 + c
    omega = 125.04 - 1934.136 * jc
    app_long = true_long - 0.00569 - 0.00478 * math.sin(math.radians(omega))
    eps0 = 23.0 + (26.0 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60.0) / 60.0
    eps = eps0 + 0.00256 * math.cos(math.radians(omega))
    decl = math.asin(math.sin(math.radians(eps)) * math.sin(math.radians(app_long)))
    yv = math.tan(math.radians(eps / 2.0)) ** 2
    l0r = math.radians(l0)
    eqt = 4.0 * math.degrees(yv * math.sin(2 * l0r) - 2 * e * math.sin(mr) + 4 * e * yv * math.sin(mr) * math.cos(2 * l0r)
                             - 0.5 * yv * yv * math.sin(4 * l0r) - 1.25 * e * e * math.sin(2 * mr))
    minutes = dt_utc.hour * 60.0 + dt_utc.minute + dt_utc.second / 60.0
    tst = (minutes + eqt + 4.0 * lon) % 1440.0
    ha = tst / 4.0 - 180.0
    latr = math.radians(lat)
    cz = math.sin(latr) * math.sin(decl) + math.cos(latr) * math.cos(decl) * math.cos(math.radians(ha))
    zen = math.acos(max(-1.0, min(1.0, cz)))
    el = 90.0 - math.degrees(zen)
    den = math.cos(latr) * math.sin(zen)
    if abs(den) > 1e-9:
        x = (math.sin(latr) * math.cos(zen) - math.sin(decl)) / den
        az = math.degrees(math.acos(max(-1.0, min(1.0, x))))
        az = (az + 180.0) % 360.0 if ha > 0 else (540.0 - az) % 360.0
    else:
        az = 180.0 if lat > 0 else 0.0
    return az, el


def _rise_set(lat, lon, day, offset_h):
    """Local sunrise and sunset (HH:MM) by stepping the day minute by minute through elevation -0.833 deg."""
    out = {"sunrise": None, "sunset": None}
    start = datetime.datetime(day.year, day.month, day.day) - datetime.timedelta(hours=offset_h)
    prev = None
    for minute in range(0, 24 * 60 + 1, 2):
        t = start + datetime.timedelta(minutes=minute)
        el = solar_position(lat, lon, t)[1] + 0.833
        if prev is not None:
            if prev < 0 <= el and out["sunrise"] is None:
                out["sunrise"] = "%02d:%02d" % divmod(minute, 60)
            elif prev >= 0 > el:
                out["sunset"] = "%02d:%02d" % divmod(minute, 60)
        prev = el
    return out


def op_set_sun_position(params):
    """Place and aim a Sun light (and a sky texture's sun) from a real place and time - the Sun Position add-on's job,
    native, since the add-on left Blender at 4.2.

    params:
      object (str)          a SUN light object (made, named "Sun", when there is none of that name)
      latitude, longitude   degrees (north and east positive); required
      date (str)            YYYY-MM-DD (default today); time (str) HH:MM local (default 12:00)
      utcOffset (float)     hours (e.g. -5 for New York in winter); dst (bool) adds one hour
      northOffset (float)   degrees the scene's north is turned from +Y, counterclockwise seen from above
      sky (str | bool)      a world Sky Texture node's name (true: the first one) - its sun follows: sun_direction
                            (PREETHAM / HOSEK_WILKIE) and sun_elevation / sun_rotation (NISHITA, SINGLE_ / MULTIPLE_
                            SCATTERING)
      distance (float)      the light is also MOVED to that distance toward the sun (default: left where it is)
      strength (float)      the sun's strength
      keyFrame (int)        key the light's rotation (and the sky's sun) at that frame - a time-lapse in calls
    Reply: azimuth (from north, clockwise) and elevation in degrees, the direction toward the sun in scene axes, sunrise
    and sunset (local time), and a note when the sun is below the horizon. The solar position is NOAA's (Meeus) without
    refraction - within a tenth of a degree of the NOAA calculator.
    """
    reject_unknown(params, _SUN_KEYS, "set_sun_position")
    tail = "NOTHING was changed."
    from mathutils import Vector                           # noqa: PLC0415
    lat = take_float(params, "latitude", required=True)
    lon = take_float(params, "longitude", required=True)
    if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 180.0:
        raise MifOpError("latitude is -90 to 90 and longitude -180 to 180, got %g, %g. %s" % (lat, lon, tail))
    offset = take_float(params, "utcOffset", default=0.0)
    if take_bool(params, "dst", default=False):
        offset += 1.0
    try:
        day = (datetime.date.fromisoformat(params["date"]) if params.get("date") else datetime.date.today())
        tm = re.fullmatch(r"\s*(\d{1,2}):(\d{1,2})(?::\d{1,2}(?:\.\d+)?)?\s*", str(params.get("time") or "12:00"))
        if tm is None:
            raise ValueError("time")
        hh, mm = int(tm.group(1)), int(tm.group(2))
        local = datetime.datetime(day.year, day.month, day.day, hh, mm)
    except (ValueError, TypeError):
        raise MifOpError("date is YYYY-MM-DD and time HH:MM, got %r and %r. %s" % (params.get("date"), params.get("time"),
                                                                                tail))
    north = take_float(params, "northOffset", default=0.0)
    utc = local - datetime.timedelta(hours=offset)
    az, el = solar_position(lat, lon, utc)
    azr = math.radians(az - north)
    elr = math.radians(el)
    toward = Vector((math.sin(azr) * math.cos(elr), math.cos(azr) * math.cos(elr), math.sin(elr)))
    name = take(params, "object", default="Sun", kind=str)
    # EVERYTHING THAT CAN REFUSE, BEFORE A SUN IS MADE: the sky node, distance, strength and keyFrame were resolved
    # after lights.new and objects.new, so a refusal left a new Sun in the scene under "NOTHING was changed".
    sky_node = None
    sky = params.get("sky")
    if sky:
        world = bpy.context.scene.world
        nodes = world.node_tree.nodes if world and world.use_nodes else []
        skies = [n for n in nodes if n.type == "TEX_SKY"]
        sky_node = (skies[0] if sky is True and skies else next((n for n in skies if n.name == sky), None))
        if sky_node is None:
            raise MifOpError("the world has no Sky Texture node %s (it has: %s). %s"
                             % ("" if sky is True else repr(sky), [n.name for n in skies] or "none", tail))
    distance = take_float(params, "distance") if params.get("distance") is not None else None
    strength = take_float(params, "strength") if params.get("strength") is not None else None
    key_frame = take_int(params, "keyFrame") if params.get("keyFrame") is not None else None
    obj = bpy.data.objects.get(name)
    made = False
    if obj is None:
        light = bpy.data.lights.new(name, "SUN")
        obj = bpy.data.objects.new(name, light)
        bpy.context.scene.collection.objects.link(obj)
        made = True
    elif obj.type != "LIGHT" or obj.data.type != "SUN":
        raise MifOpError("'%s' is not a SUN light. %s" % (name, tail))
    obj.rotation_mode = "XYZ"
    # a sun shines down its local -Z, so its +Z is turned toward the sun
    obj.rotation_euler = toward.to_track_quat("Z", "Y").to_euler()
    if distance is not None:
        obj.location = toward * distance
    if strength is not None:
        obj.data.energy = strength
    if sky_node is not None:
        if sky_node.sky_type in ("PREETHAM", "HOSEK_WILKIE"):
            sky_node.sun_direction = toward
        else:
            sky_node.sun_elevation = elr
            sky_node.sun_rotation = azr % (2 * math.pi)
    if key_frame is not None:
        f = key_frame
        obj.keyframe_insert("rotation_euler", frame=f)
        if distance is not None:
            obj.keyframe_insert("location", frame=f)
        if sky_node is not None:
            nt = bpy.context.scene.world.node_tree
            for prop in (("sun_direction",) if sky_node.sky_type in ("PREETHAM", "HOSEK_WILKIE")
                         else ("sun_elevation", "sun_rotation")):
                nt.keyframe_insert('nodes["%s"].%s' % (sky_node.name, prop), frame=f)
    bpy.context.view_layer.update()
    aim = obj.matrix_world.to_3x3() @ Vector((0, 0, 1))
    out = {"object": obj.name, "made": made, "latitude": lat, "longitude": lon, "local": local.isoformat(timespec="minutes"),
           "utcOffset": offset, "utc": utc.isoformat(timespec="minutes"), "azimuth": round(az, 4),
           "elevation": round(el, 4), "northOffset": north, "towardSun": [round(v, 6) for v in toward],
           "lightPointsAwayFrom": [round(v, 6) for v in aim], "sky": sky_node.name if sky_node else None,
           "keyFrame": key_frame}
    out.update(_rise_set(lat, lon, day, offset))
    if el < 0:
        out["note"] = "the sun is %.2f degrees below the horizon at that time - it lights the scene from below" % -el
    return out


# --------------------------------------------------------------------------------------------- render_viewport

_RV_KEYS = ("filePath", "folder", "frameStart", "frameEnd", "camera", "shading", "colorType", "resolutionX",
            "resolutionY", "percentage", "overlays", "silhouette")


def op_render_viewport(params):
    """A playblast - a viewport-style render of a still or a frame range in seconds per frame.

    In a session with a window and a 3D View, Blender's Viewport Render (render.opengl) - what the viewport shows, with
    its overlays when overlays is true. Headless (blender -b), where render.opengl has no viewport to draw, the same look
    through a Workbench render: solid shading with the material (or object / single / texture) colors, the studio light.
    The reply says which route ran.

    params:
      filePath (str)        a still (PNG); or
      folder (str) + frameStart / frameEnd   a frame range, one PNG per frame (frame_####.png)
      camera (str)          default the scene camera
      shading (str)         SOLID (default) | WIREFRAME (Workbench) - the UI route uses the viewport's own shading
      colorType (str)       MATERIAL (default) | OBJECT | SINGLE | TEXTURE | VERTEX | RANDOM
      resolutionX, resolutionY, percentage   default the scene's
      overlays (bool)       UI route: draw overlays (bones, outlines) - default false. Headless has no overlays to draw:
                            the reply says so in overlaysNote rather than leaving true looking honored
      silhouette            true | {color, background, collections {name: [r,g,b]}} - every object one flat colour on a
                            flat background (black on white by default), anti-aliasing off: two values in the file. It
                            takes the Workbench route in any session; shading and colorType do not apply
    The scene's engine, camera, resolution and output are put back after.
    """
    reject_unknown(params, _RV_KEYS, "render_viewport")
    tail = "NOTHING was rendered."
    sc = bpy.context.scene
    still = take(params, "filePath", default=None, kind=str)
    folder = take(params, "folder", default=None, kind=str)
    if (still is None) == (folder is None):
        raise MifOpError("give filePath (a still) or folder with frameStart / frameEnd (a range). %s" % tail)
    cam = sc.camera
    if params.get("camera") is not None:
        cam = get_object(take(params, "camera", kind=str))
        if cam.type != "CAMERA":
            raise MifOpError("'%s' is not a camera. %s" % (cam.name, tail))
    if cam is None:
        raise MifOpError("there is no scene camera - pass camera. %s" % tail)
    if still is not None:
        for k in ("frameStart", "frameEnd"):
            if params.get(k) is not None:
                raise MifOpError("%s goes with folder (a range). %s" % (k, tail))
        path = os.path.abspath(bpy.path.abspath(still))
        check_output_path(still, path, "rendered")
        frames = [sc.frame_current]
        paths = [path]
    else:
        d = os.path.abspath(bpy.path.abspath(folder))
        os.makedirs(d, exist_ok=True)
        start = take_int(params, "frameStart", default=sc.frame_start)
        end = take_int(params, "frameEnd", default=sc.frame_end)
        if end < start or end - start > 10000:
            raise MifOpError("frameEnd must be at or after frameStart, within 10000 frames. %s" % tail)
        frames = list(range(start, end + 1))
        paths = [os.path.join(d, "frame_%04d.png" % f) for f in frames]
    sil = silhouette_plan(params.get("silhouette"), tail)
    if sil is not None and (params.get("shading") is not None or params.get("colorType") is not None
                            or params.get("overlays")):
        raise MifOpError("a silhouette is one flat colour per object, so shading, colorType and overlays would change "
                         "nothing - pass silhouette alone. %s" % tail)
    shading = str(take(params, "shading", default="SOLID", kind=str)).upper()
    want_overlays = take_bool(params, "overlays", default=False)
    ctype = str(take(params, "colorType", default="MATERIAL", kind=str)).upper()
    r = sc.render
    saved = {"engine": r.engine, "camera": sc.camera, "x": r.resolution_x, "y": r.resolution_y,
             "pct": r.resolution_percentage, "fp": r.filepath, "fmt": r.image_settings.file_format,
             "frame": sc.frame_current}
    # THE WINDOW IS LOOKED UP, NOT READ OFF THE CONTEXT (1.4, FOUND 2026-11-11). Ops run from the server's drain timer,
    # where bpy.context.window and screen are None, so a live GUI session always took the Workbench route: no overlays,
    # no image empties. The window manager's windows are searched for a 3D View, the active window first.
    win, view3d = None, None
    if not bpy.app.background:
        wm = bpy.context.window_manager
        wins = [w for w in ([bpy.context.window] + list(wm.windows if wm is not None else [])) if w is not None]
        for w in wins:
            area = next((a for a in (w.screen.areas if w.screen is not None else []) if a.type == "VIEW_3D"), None)
            if area is not None:
                win, view3d = w, area
                break
    route = "viewport" if view3d is not None and sil is None else "workbench"
    t0 = time.time()
    written = []
    sil_undo, sil_reply = [], None
    try:
        sc.camera = cam
        if params.get("resolutionX") is not None:
            r.resolution_x = take_int(params, "resolutionX")
        if params.get("resolutionY") is not None:
            r.resolution_y = take_int(params, "resolutionY")
        if params.get("percentage") is not None:
            r.resolution_percentage = take_int(params, "percentage")
        r.image_settings.file_format = "PNG"
        if route == "workbench":
            r.engine = "BLENDER_WORKBENCH"
            disp = sc.display.shading
            saved["shading"] = (disp.light, disp.color_type, disp.type if hasattr(disp, "type") else None)
            disp.light = "STUDIO"
            try:
                disp.color_type = ctype
            except TypeError as exc:
                raise MifOpError("colorType %r: %s. %s" % (ctype, exc, tail))
            if shading == "WIREFRAME" and hasattr(disp, "type"):
                disp.type = "WIREFRAME"
        if sil is not None:
            sil_reply = silhouette_apply(sc, sil, sil_undo)
        for f, p in zip(frames, paths):
            sc.frame_set(f)
            r.filepath = p
            if route == "viewport":
                space = view3d.spaces.active
                keep = space.overlay.show_overlays
                space.overlay.show_overlays = want_overlays
                region = next(rg for rg in view3d.regions if rg.type == "WINDOW")
                try:
                    with bpy.context.temp_override(window=win, screen=win.screen, area=view3d, region=region):
                        bpy.ops.render.opengl(write_still=True, view_context=False)
                finally:
                    space.overlay.show_overlays = keep
            else:
                bpy.ops.render.render(write_still=True)
            if os.path.isfile(p) and os.path.getmtime(p) >= t0 - 1:
                written.append(p)
    finally:
        sil_failed = review_restore(sil_undo)
        if route == "workbench" and "shading" in saved:
            disp = sc.display.shading
            disp.light, disp.color_type = saved["shading"][0], saved["shading"][1]
            if saved["shading"][2] is not None:
                disp.type = saved["shading"][2]
        r.engine = saved["engine"]
        sc.camera = saved["camera"]
        r.resolution_x, r.resolution_y, r.resolution_percentage = saved["x"], saved["y"], saved["pct"]
        r.filepath = saved["fp"]
        r.image_settings.file_format = saved["fmt"]
        sc.frame_set(saved["frame"])
    secs = time.time() - t0
    out = {"route": route, "routeNote": ("Viewport Render in this session's 3D View" if route == "viewport" else
                                         "headless: a Workbench render with solid shading stands in for the viewport"
                                         if bpy.app.background else
                                         "no window of this session shows a 3D View: a Workbench render with solid "
                                         "shading stands in for the viewport. Open a 3D View for the viewport route"),
           "frames": len(frames), "framesWritten": len(written), "files": written[:20], "camera": cam.name,
           "seconds": round(secs, 3), "secondsPerFrame": round(secs / max(len(frames), 1), 4)}
    if route != "viewport" and params.get("overlays"):
        out["overlaysNote"] = ("overlays were not drawn: headless has no viewport, and the Workbench render that stands in "
                               "for it has no overlays. A session with a 3D View draws them.")
    if len(written) != len(frames):
        out["warning"] = "%d of %d frames reached disk" % (len(written), len(frames))
    if sil_reply is not None:
        out["silhouette"] = dict(sil_reply, sceneRestored=sil_failed == 0)
    return out


# --------------------------------------------------------------------------------------------- review renders
# render_still's review modes, silhouette, Bevel/AO preview, single-light and colour-cast check (1.4, FOUND 2026-11-18
# by every build of the UEFN hand-off): the standard review is clay, wire and final per view, a black-on-white silhouette
# for form, and a final judged under a stage that does not tint it. Every one of them changes the scene for one render
# and puts it back: review_apply returns the undo steps, review_restore runs them newest first.

STILL_MODES = ("final", "clay", "wire", "clay+wire")
REVIEW_KEYS = ("mode", "clayColor", "wireColor", "wirePx", "silhouette", "preview", "soloLight", "castCheck")
_TMP = "MifReview_"
_WIRE_PROP = "mif_review_wire"
_SIL_PROP = "mif_review_sil"
_SIL_KEYS = ("color", "background", "collections")


def _rgb(raw, default, key, tail):
    if raw is None:
        return tuple(default)
    if not isinstance(raw, (list, tuple)) or len(raw) not in (3, 4):
        raise MifOpError("%s is [r, g, b] in 0-1, got %r. %s" % (key, raw, tail))
    vals = finite_floats(list(raw)[:3], key)
    if any(v < 0.0 or v > 1.0 for v in vals):
        raise MifOpError("%s channels are 0-1, got %r. %s" % (key, raw, tail))
    return tuple(vals)


def _render_visible_meshes(sc, types=("MESH",)):
    vl = bpy.context.view_layer
    in_layer = {o.name for o in vl.objects}
    return [o for o in sc.objects if o.type in types and not o.hide_render and o.name in in_layer
            and o.visible_get(view_layer=vl)]


def silhouette_plan(raw, tail):
    """silhouette true | {color, background, collections {name: [r,g,b]}} -> a plan, or None for false / absent."""
    if raw is None or raw is False:
        return None
    if raw is True:
        raw = {}
    if not isinstance(raw, dict):
        raise MifOpError("silhouette is true or {color, background, collections}, got %r. %s" % (raw, tail))
    bad = sorted(set(raw) - set(_SIL_KEYS))
    if bad:
        raise MifOpError("silhouette takes %s, not %s. %s" % (", ".join(_SIL_KEYS), ", ".join(bad), tail))
    cols = raw.get("collections") or {}
    if not isinstance(cols, dict):
        raise MifOpError("silhouette.collections is {collection name: [r, g, b]}, got %r. %s" % (cols, tail))
    by_coll = {}
    for name, col in cols.items():
        coll = bpy.data.collections.get(str(name))
        if coll is None:
            raise MifOpError("silhouette.collections names '%s', which is not a collection. %s" % (name, tail))
        by_coll[coll.name] = _rgb(col, (0, 0, 0), "silhouette.collections.%s" % name, tail)
    return {"color": _rgb(raw.get("color"), (0.0, 0.0, 0.0), "silhouette.color", tail),
            "background": _rgb(raw.get("background"), (1.0, 1.0, 1.0), "silhouette.background", tail),
            "collections": by_coll}


def review_plan(params, sc, tail="NOTHING was rendered."):
    """render_still's review keys, every check and no write -> a plan dict (empty when the call asks for none)."""
    mode = str(take(params, "mode", default="final", kind=str)).strip().lower().replace(" ", "")
    mode = {"clay_wire": "clay+wire", "claywire": "clay+wire", "clay-wire": "clay+wire"}.get(mode, mode)
    if mode not in STILL_MODES:
        raise MifOpError("mode is one of %s, got %r. %s" % (", ".join(STILL_MODES), params.get("mode"), tail))
    for key in ("clayColor", "wireColor", "wirePx"):
        if params.get(key) is not None and mode == "final":
            raise MifOpError("%s shapes the clay and wire modes and mode is final, so it would change nothing. %s"
                             % (key, tail))
    if params.get("clayColor") is not None and mode == "wire":
        raise MifOpError("clayColor colours the clay, and mode wire draws no clay - use clay+wire. %s" % tail)
    if params.get("wireColor") is not None and mode == "clay":
        raise MifOpError("wireColor colours the wire, and mode clay draws none - use clay+wire. %s" % tail)
    wire_px = take_float(params, "wirePx", default=1.5)
    if not 0.25 <= wire_px <= 20.0:
        raise MifOpError("wirePx is the wire's width in pixels, 0.25-20, got %g. %s" % (wire_px, tail))
    if mode != "final" and "WORKBENCH" in sc.render.engine:
        raise MifOpError("mode %s overrides the materials, and Workbench draws the viewport colours instead. Render with "
                         "Cycles or EEVEE, or use render_viewport (shading SOLID / WIREFRAME) for a Workbench look. %s"
                         % (mode, tail))
    sil = silhouette_plan(params.get("silhouette"), tail)
    if sil is not None and mode != "final":
        raise MifOpError("silhouette draws every object one flat colour, so mode %s would not show - pass one. %s"
                         % (mode, tail))
    preview = take_bool(params, "preview", default=False)
    solo = None
    if params.get("soloLight") is not None:
        solo = bpy.data.objects.get(str(take(params, "soloLight", kind=str)))
        if solo is None or solo.type != "LIGHT":
            lights = sorted(o.name for o in sc.objects if o.type == "LIGHT")
            raise MifOpError("soloLight names %r, which is not a light in this scene (lights: %s). %s"
                             % (params.get("soloLight"), ", ".join(lights[:20]) or "none", tail))
        if solo.hide_render:
            raise MifOpError("soloLight '%s' is hidden from the render, so it alone would render black. %s"
                             % (solo.name, tail))
        if sil is not None:
            raise MifOpError("a silhouette is unlit, so soloLight would change nothing. %s" % tail)
    cast = params.get("castCheck")
    if cast is not None and not isinstance(cast, (bool, dict)):
        raise MifOpError("castCheck is true or {threshold}, got %r. %s" % (cast, tail))
    cast_thr = None
    if cast:
        cast_thr = 0.05
        if isinstance(cast, dict):
            bad = sorted(set(cast) - {"threshold"})
            if bad:
                raise MifOpError("castCheck takes threshold, not %s. %s" % (", ".join(bad), tail))
            cast_thr = finite_float(cast.get("threshold", 0.05), "castCheck.threshold")
        if "WORKBENCH" in sc.render.engine or sil is not None:
            raise MifOpError("castCheck measures the scene's world and lights on a grey sphere, and %s draws no lighting. "
                             "%s" % ("a silhouette" if sil is not None else "Workbench", tail))
    return {"mode": mode, "clay": _rgb(params.get("clayColor"), (0.8, 0.8, 0.8), "clayColor", tail),
            "wire": _rgb(params.get("wireColor"), (0.02, 0.02, 0.02), "wireColor", tail), "wirePx": wire_px,
            "silhouette": sil, "preview": preview, "solo": solo, "castThreshold": cast_thr}


def _set(undo, holder, attr, value):
    """Write holder.attr, remembering the old value in undo."""
    old = getattr(holder, attr)
    if not isinstance(old, (str, int, float, bool)) and old is not None and not isinstance(old, bpy.types.ID):
        old = tuple(old)
    undo.append(lambda h=holder, a=attr, v=old: setattr(h, a, v))
    setattr(holder, attr, value)


def review_restore(undo):
    """Run the undo steps newest first; a step that fails does not stop the others. -> the number that failed."""
    failed = 0
    while undo:
        step = undo.pop()
        try:
            step()
        except Exception:                                  # noqa: BLE001 - put back what can be put back
            failed += 1
    return failed


def _new_material(undo, name):
    mat = bpy.data.materials.new(_TMP + name)
    mat.use_nodes = True
    undo.append(lambda m=mat: bpy.data.materials.remove(m))
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    return mat, nt, out


def _emission(nt, rgb):
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (rgb[0], rgb[1], rgb[2], 1.0)
    em.inputs["Strength"].default_value = 1.0
    return em


def _pixel_world(sc, cam, point):
    """The world size of one pixel at `point`, seen from the scene camera."""
    r = sc.render
    px = max(r.resolution_x, r.resolution_y) * r.resolution_percentage / 100.0
    if cam.data.type == "ORTHO":
        return cam.data.ortho_scale / max(px, 1.0)
    cm = cam.matrix_world
    fwd = (cm.to_quaternion() @ mathutils.Vector((0.0, 0.0, -1.0))).normalized()
    d = max((point - cm.to_translation()).dot(fwd), cam.data.clip_start)
    return 2.0 * d * math.tan(cam.data.angle / 2.0) / max(px, 1.0)


def _wire_copies(sc, undo, wire_px):
    """A copy of every render-visible mesh sharing its data, its modifiers plus a WIREFRAME (even offset OFF - the spikes
    a hand-made overlay threw off sliver faces), seen by camera rays only, tagged for the override's Attribute node."""
    cam = sc.camera
    coll = bpy.data.collections.new(_TMP + "Wire")
    sc.collection.children.link(coll)
    undo.append(lambda c=coll: bpy.data.collections.remove(c))
    made = []
    for obj in _render_visible_meshes(sc):
        cp = obj.copy()
        coll.objects.link(cp)
        undo.append(lambda o=cp: remove_objects([o]))
        for m in [m for m in cp.modifiers if m.type == "PARTICLE_SYSTEM"]:
            cp.modifiers.remove(m)
        corners = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
        centre = sum(corners, mathutils.Vector()) / 8.0
        scale = max(sum(abs(v) for v in obj.matrix_world.to_scale()) / 3.0, 1e-9)
        mod = cp.modifiers.new(_TMP + "Wireframe", "WIREFRAME")
        mod.thickness = _pixel_world(sc, cam, centre) * wire_px / scale
        mod.use_even_offset = False
        mod.use_relative_offset = False
        mod.use_replace = True
        mod.use_boundary = True
        mod.offset = 0.0
        cp[_WIRE_PROP] = 1.0
        for attr in ("visible_diffuse", "visible_glossy", "visible_transmission", "visible_volume_scatter",
                     "visible_shadow"):
            if hasattr(cp, attr):
                setattr(cp, attr, False)
        made.append(cp.name)
    return made


def _clay_wire(sc, plan, undo):
    mode = plan["mode"]
    mat, nt, out = _new_material(undo, mode.replace("+", "_"))
    if mode == "wire":
        surface = _emission(nt, (1.0, 1.0, 1.0))
    else:
        surface = nt.nodes.new("ShaderNodeBsdfPrincipled")
        surface.inputs["Base Color"].default_value = plan["clay"] + (1.0,)
        surface.inputs["Roughness"].default_value = 0.55
    shader = surface.outputs[0]
    wires = []
    if "wire" in mode:
        attr = nt.nodes.new("ShaderNodeAttribute")
        attr.attribute_type = "OBJECT"
        attr.attribute_name = _WIRE_PROP
        mix = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(attr.outputs["Fac"], mix.inputs[0])
        nt.links.new(surface.outputs[0], mix.inputs[1])
        nt.links.new(_emission(nt, plan["wire"]).outputs[0], mix.inputs[2])
        shader = mix.outputs[0]
    nt.links.new(shader, out.inputs["Surface"])
    for vl in sc.view_layers:
        _set(undo, vl, "material_override", mat)
    if "wire" in mode:
        wires = _wire_copies(sc, undo, plan["wirePx"])
    return {"mode": mode, "overrideMaterial": mat.name, "wireObjects": len(wires),
            "clayColor": list(plan["clay"]) if mode != "wire" else None,
            "wireColor": list(plan["wire"]) if "wire" in mode else None,
            "wirePx": plan["wirePx"] if "wire" in mode else None}


def silhouette_apply(sc, sil, undo):
    """Workbench, flat light, one colour per object (the silhouette's, or its collection's), anti-aliasing off, a
    background colour, the Standard view, no dither - two values in the file. Every write is in undo."""
    _set(undo, sc.render, "engine", "BLENDER_WORKBENCH")
    sh = sc.display.shading
    _set(undo, sh, "light", "FLAT")
    _set(undo, sh, "color_type", "OBJECT")
    for attr in ("show_object_outline", "show_cavity", "show_shadows", "show_specular_highlight", "show_xray",
                 "show_backface_culling", "use_dof"):
        if hasattr(sh, attr):
            _set(undo, sh, attr, False)
    if hasattr(sc.display, "render_aa"):
        _set(undo, sc.display, "render_aa", "OFF")
    _set(undo, sc.render, "film_transparent", False)
    # DITHER OFF: Blender's default dither (1.0) moved 1,452 of 16,384 pixels of a black-on-white cube one level off
    # 0 and 255 (measured on 5.0.1, 128 px) - four values in a file that must hold two
    if hasattr(sc.render, "dither_intensity"):
        _set(undo, sc.render, "dither_intensity", 0.0)
    # THE COMPOSITOR AND SEQUENCER OFF: a node tree a scene keeps (a glare, a blur) turned a black-on-white render into
    # 30-odd greys after the compositing suite had run in the same Blender
    for attr in ("use_compositing", "use_sequencer"):
        if hasattr(sc.render, attr):
            _set(undo, sc.render, attr, False)
    vs = sc.view_settings
    _set(undo, vs, "view_transform", "Standard")
    _set(undo, vs, "look", "None")
    _set(undo, vs, "exposure", 0.0)
    _set(undo, vs, "gamma", 1.0)
    world = bpy.data.worlds.new(_TMP + "Background")
    undo.append(lambda w=world: bpy.data.worlds.remove(w))
    world.color = sil["background"]
    _set(undo, sc, "world", world)
    by_obj = {}
    for cname, col in sil["collections"].items():
        for o in bpy.data.collections[cname].all_objects:
            by_obj.setdefault(o.name, col)
    coloured = 0
    for o in sc.objects:
        if o.type in ("MESH", "CURVE", "SURFACE", "META", "FONT", "CURVES", "POINTCLOUD", "VOLUME"):
            _set(undo, o, "color", tuple(by_obj.get(o.name, sil["color"])) + (1.0,))
            coloured += 1
    return {"silhouette": True, "color": list(sil["color"]), "background": list(sil["background"]),
            "collections": {k: list(v) for k, v in sil["collections"].items()} or None, "objectsColoured": coloured,
            "engine": "BLENDER_WORKBENCH", "silhouetteNote": "a Workbench render, flat light, anti-aliasing off: every "
            "pixel is the background or an object's colour. Shader displacement does not draw in Workbench."}


def _trees_in_use(sc):
    seen, out = set(), []

    def walk(tree):
        if tree is None or tree.as_pointer() in seen:
            return
        seen.add(tree.as_pointer())
        out.append(tree)
        for n in tree.nodes:
            if n.type == "GROUP" and getattr(n, "node_tree", None) is not None:
                walk(n.node_tree)
    mats = set()
    for o in _render_visible_meshes(sc, ("MESH", "CURVE", "SURFACE", "META", "FONT")):
        for slot in o.material_slots:
            if slot.material is not None and slot.material.use_nodes:
                mats.add(slot.material)
    for m in sorted(mats, key=lambda m: m.name):
        walk(m.node_tree)
    return out, sorted(m.name for m in mats)


def _relink(nt, undo, link, src):
    """Replace one link's source with `src` (a socket) for this render; undo puts the original back."""
    old_from, to = link.from_socket, link.to_socket
    nt.links.remove(link)
    new = nt.links.new(src, to)

    def back(nt=nt, new=new, old_from=old_from, to=to):
        try:
            nt.links.remove(new)
        except (RuntimeError, ReferenceError):
            pass
        nt.links.new(old_from, to)
    undo.append(back)


def preview_apply(sc, undo):
    """Bevel and AO nodes out of every material the render sees, for this render: a Bevel's Normal becomes the shading
    normal (a Geometry node - a bevel of radius 0), an AO's AO output 1 and its Color output what feeds its Color input.
    Cycles pays for both per sample with extra rays - the cost a weathering chain adds to a preview."""
    trees, mats = _trees_in_use(sc)
    counts = {"bevelNodes": 0, "aoNodes": 0, "linksSwapped": 0}
    for nt in trees:
        for node in [n for n in nt.nodes if n.type in ("BEVEL", "AMBIENT_OCCLUSION")]:
            outs = [lk for lk in nt.links if lk.from_node == node]
            if not outs:
                continue
            temps = []
            if node.type == "BEVEL":
                counts["bevelNodes"] += 1
                src_in = node.inputs.get("Normal")
                if src_in is not None and src_in.is_linked:
                    src = src_in.links[0].from_socket
                else:
                    geo = nt.nodes.new("ShaderNodeNewGeometry")
                    temps.append(geo)
                    src = geo.outputs["Normal"]
                for lk in outs:
                    _relink(nt, undo, lk, src)
                    counts["linksSwapped"] += 1
            else:
                counts["aoNodes"] += 1
                one = nt.nodes.new("ShaderNodeValue")
                one.outputs[0].default_value = 1.0
                temps.append(one)
                cin = node.inputs.get("Color")
                if cin is not None and cin.is_linked:
                    csrc = cin.links[0].from_socket
                else:
                    rgb = nt.nodes.new("ShaderNodeRGB")
                    rgb.outputs[0].default_value = tuple(cin.default_value) if cin is not None else (1, 1, 1, 1)
                    temps.append(rgb)
                    csrc = rgb.outputs[0]
                for lk in outs:
                    _relink(nt, undo, lk, one.outputs[0] if lk.from_socket.name == "AO" else csrc)
                    counts["linksSwapped"] += 1
            for t in temps:
                undo.append(lambda nt=nt, t=t: nt.nodes.remove(t))
    counts["materials"] = mats
    counts["previewNote"] = ("Bevel and AO nodes were taken out of the shading for this render only and put back after. "
                             "Edge wear and grime that read them render flat here; the final needs preview false.")
    return counts


def solo_apply(sc, solo, undo):
    hidden = []
    for o in sc.objects:
        if o.type == "LIGHT" and o is not solo and not o.hide_render:
            _set(undo, o, "hide_render", True)
            hidden.append(o.name)
    world = sc.world.name if sc.world is not None else None
    if sc.world is not None:
        _set(undo, sc, "world", None)
    return {"light": solo.name, "lightsHidden": hidden, "worldRemoved": world,
            "soloNote": "only this light lit the render: the other lights were hidden and the world taken off for it, "
                        "and put back after. A mesh with an emissive material still emits."}


def review_apply(sc, plan, undo):
    """Apply a review_plan's writes for one render -> the reply block (None when the plan asks for nothing)."""
    out = {}
    if plan["mode"] != "final":
        out.update(_clay_wire(sc, plan, undo))
    if plan["silhouette"] is not None:
        out.update(silhouette_apply(sc, plan["silhouette"], undo))
    if plan["preview"]:
        out["preview"] = preview_apply(sc, undo)
    if plan["solo"] is not None:
        out["soloLight"] = solo_apply(sc, plan["solo"], undo)
    return out or None


def _png_pixels(path):
    """(width, height, flat RGBA floats 0-1 as stored) of an image file, the datablock removed after."""
    img = bpy.data.images.load(path, check_existing=False)
    try:
        w, h = img.size
        px = list(img.pixels[:])
    finally:
        bpy.data.images.remove(img)
    return w, h, px


def cast_check(sc, threshold, tail="NOTHING was rendered.", bounds=None):
    """Render a neutral grey sphere under the scene's world and lights alone, at 64 px, and measure its colour.

    The sphere (a Diffuse BSDF at 0.5) sits at the centre of the render-visible meshes, half their size; every other
    render-visible object is hidden for it, and the film is transparent so the sphere's pixels are those with alpha.
    bounds (centre, radius) places the sphere instead (look_studio passes its subject's, so its own floor does not count).
    -> {meanRGB (0-255 as written), hueDegrees, strength ((max - min) / max of the mean), warning}."""
    import tempfile
    undo = []
    try:
        subjects = _render_visible_meshes(sc, ("MESH", "CURVE", "SURFACE", "META", "FONT"))
        if bounds is not None:
            centre, radius = mathutils.Vector(bounds[0]), max(float(bounds[1]) / 2.0, 1e-3)
        elif subjects:
            pts = [o.matrix_world @ mathutils.Vector(c) for o in subjects for c in o.bound_box]
            lo = mathutils.Vector([min(p[i] for p in pts) for i in range(3)])
            hi = mathutils.Vector([max(p[i] for p in pts) for i in range(3)])
            centre, radius = (lo + hi) / 2.0, max((hi - lo).length / 4.0, 1e-3)
        else:
            cm = sc.camera.matrix_world
            centre = cm.to_translation() + cm.to_quaternion() @ mathutils.Vector((0.0, 0.0, -3.0))
            radius = 0.5
        for o in sc.objects:
            if o.type != "LIGHT" and o.type != "CAMERA" and not o.hide_render:
                _set(undo, o, "hide_render", True)
        mesh = bpy.data.meshes.new(_TMP + "CastSphere")
        import bmesh
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, **({"radius": radius} if bpy.app.version >= (3, 0)
                                                                         else {"diameter": radius}))
        for f in bm.faces:
            f.smooth = True
        bm.to_mesh(mesh)
        bm.free()
        undo.append(lambda m=mesh: bpy.data.meshes.remove(m))
        sphere = bpy.data.objects.new(_TMP + "CastSphere", mesh)
        sc.collection.objects.link(sphere)
        undo.append(lambda o=sphere: remove_objects([o]))
        sphere.location = centre
        mat, nt, out = _new_material(undo, "CastGrey")
        dif = nt.nodes.new("ShaderNodeBsdfDiffuse")
        dif.inputs["Color"].default_value = (0.5, 0.5, 0.5, 1.0)
        nt.links.new(dif.outputs[0], out.inputs["Surface"])
        mesh.materials.append(mat)
        for vl in sc.view_layers:
            _set(undo, vl, "material_override", None)
        r = sc.render
        path = os.path.join(tempfile.gettempdir(), "mif_cast_%d.png" % os.getpid())
        _set(undo, r, "filepath", path)
        _set(undo, r, "resolution_x", 64)
        _set(undo, r, "resolution_y", 64)
        _set(undo, r, "resolution_percentage", 100)
        _set(undo, r, "film_transparent", True)
        _set(undo, r, "use_border", False)
        img = r.image_settings
        if hasattr(img, "media_type"):
            _set(undo, img, "media_type", "IMAGE")
        _set(undo, img, "file_format", "PNG")
        _set(undo, img, "color_mode", "RGBA")
        if "CYCLES" in r.engine and hasattr(sc, "cycles"):
            _set(undo, sc.cycles, "samples", min(int(sc.cycles.samples), 16))
        cam = sc.camera
        cam_saved = (cam.location.copy(), cam.rotation_euler.copy(), cam.rotation_mode, cam.rotation_quaternion.copy(),
                     tuple(cam.rotation_axis_angle))
        # THE CAMERA LOOKS AT THE SPHERE FROM WHERE IT STANDS, filling the frame: the scene camera's direction keeps the
        # light the subject gets from that side, the framing makes every pixel count.
        loc = cam.matrix_world.to_translation()
        d = centre - loc
        cam_data_saved = [(cam.data, a, getattr(cam.data, a)) for a in ("type", "ortho_scale", "lens", "sensor_fit",
                                                                         "shift_x", "shift_y")]
        undo.append(lambda s=cam_data_saved: [setattr(h, a, v) for h, a, v in s])
        # THE MODE FIRST: assigning rotation_mode converts the current rotation into the new mode, so the saved values
        # go back after it, all three of them
        undo.append(lambda c=cam, s=cam_saved: (setattr(c, "rotation_mode", s[2]), setattr(c, "location", s[0]),
                                                setattr(c, "rotation_euler", s[1]), setattr(c, "rotation_quaternion", s[3]),
                                                setattr(c, "rotation_axis_angle", s[4])))
        if cam.parent is None and d.length > radius * 1.2:
            cam.rotation_mode = "XYZ"
            cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
            cam.data.shift_x = cam.data.shift_y = 0.0
            if cam.data.type == "PERSP":
                cam.data.sensor_fit = "AUTO"
                half = math.asin(min(radius * 1.15 / d.length, 0.99))
                cam.data.lens = cam.data.sensor_width / (2.0 * math.tan(half))
            elif cam.data.type == "ORTHO":
                cam.data.ortho_scale = radius * 2.3
        t0 = time.time()
        bpy.ops.render.render(write_still=True)
        secs = time.time() - t0
        w, h, px = _png_pixels(path)
        try:
            os.remove(path)
        except OSError:
            pass
        rs = gs = bs = 0.0
        n = 0
        for i in range(0, len(px), 4):
            if px[i + 3] > 0.99:
                rs += px[i]
                gs += px[i + 1]
                bs += px[i + 2]
                n += 1
        if n == 0:
            raise MifOpError("the cast check's grey sphere covered no pixel of its 64 px render (the camera did not see "
                             "it), so there is no cast to report. %s" % tail)
        mean = [rs / n * 255.0, gs / n * 255.0, bs / n * 255.0]
        hi_c, lo_c = max(mean), min(mean)
        strength = (hi_c - lo_c) / hi_c if hi_c > 0 else 0.0
        import colorsys
        hue = colorsys.rgb_to_hsv(*[c / 255.0 for c in mean])[0] * 360.0
        out = {"meanRGB": [round(c, 2) for c in mean], "pixels": n, "hueDegrees": round(hue, 1) if strength > 0.005
               else None, "strength": round(strength, 4), "threshold": threshold, "sphereRadius": round(radius, 4),
               "sphereCentre": [round(v, 4) for v in centre], "seconds": round(secs, 3),
               "viewTransform": sc.view_settings.view_transform,
               "castNote": "a grey sphere (Diffuse 0.5) at the subject's centre under the world and lights alone, the "
                           "other objects hidden; meanRGB is as written (display-referred). strength is (max - min) / "
                           "max of the mean: 0 is neutral."}
        if strength > threshold:
            out["warning"] = ("the stage tints a neutral grey: mean %s, strength %.3f over %.3f, hue %.0f degrees - a "
                              "final judged under it shows this cast, not the asset's colour"
                              % ([round(c) for c in mean], strength, threshold, hue))
        return out
    finally:
        review_restore(undo)


# --------------------------------------------------------------------------------------------- render_pixel_probe
# WHY IS THIS PIXEL BLACK (1.4, FOUND 2026-11-18, not root-caused): black faces in Cycles on a visor's inner faces and a
# keyboard front, only while a floor plane rendered; clean_mesh and mesh_quality were clean. A diagnosis needed the object
# and face the camera ray hit, the shading normal against the geometric one, the ambient occlusion there and the BSDF's
# inputs - and guessed at all four.

_PP_KEYS = ("x", "y", "pixels", "camera", "aoDistance", "aoRays")
_PP_MAX = 16
_PP_BSDF_INPUTS = (("baseColor", "Base Color"), ("metallic", "Metallic"), ("roughness", "Roughness"),
                   ("alpha", "Alpha"), ("normal", "Normal"), ("emission", ("Emission Color", "Emission")),
                   ("emissionStrength", "Emission Strength"), ("transmission", ("Transmission Weight", "Transmission")))


def camera_ray(sc, cam, x, y):
    """(origin, unit direction) of the camera ray through pixel (x, y) - top-left origin, pixel centres, at the scene's
    render size (resolution x percentage). Shift and sensor fit come in through the camera's own view_frame."""
    r = sc.render
    w = r.resolution_x * r.resolution_percentage / 100.0
    h = r.resolution_y * r.resolution_percentage / 100.0
    u, v = (x + 0.5) / w, 1.0 - (y + 0.5) / h
    tr, br, bl, tl = cam.data.view_frame(scene=sc)
    local = bl + (br - bl) * u + (tl - bl) * v
    mw = cam.matrix_world
    if cam.data.type == "ORTHO":
        origin = mw @ mathutils.Vector((local.x, local.y, 0.0))
        d = mw.to_quaternion() @ mathutils.Vector((0.0, 0.0, -1.0))
    else:
        origin = mw.to_translation()
        d = (mw @ local) - origin
    return origin, d.normalized()


def _hit_detail(eval_obj, face_index, loc_world):
    """(geometric normal, shading normal, smooth, triangle corners' local positions) at a hit on an evaluated object."""
    from .ops_kit import _corner_normals                                                   # noqa: PLC0415
    mesh = eval_obj.to_mesh()
    try:
        inv = eval_obj.matrix_world.inverted()
        p = inv @ loc_world
        poly = mesh.polygons[face_index]
        mesh.calc_loop_triangles()
        cn = _corner_normals(mesh)
        best, best_d = None, None
        for tri in mesh.loop_triangles:
            if tri.polygon_index != face_index:
                continue
            a, b, c = (mesh.vertices[i].co for i in tri.vertices)
            bary = mathutils.geometry.barycentric_transform(p, a, b, c, mathutils.Vector((1, 0, 0)),
                                                            mathutils.Vector((0, 1, 0)), mathutils.Vector((0, 0, 1)))
            dmin = min(bary)
            if best is None or dmin > best_d:
                best, best_d = (tri, bary), dmin
        nmat = eval_obj.matrix_world.to_3x3().inverted().transposed()
        geo = (nmat @ poly.normal).normalized()
        if best is None:
            return geo, geo, bool(poly.use_smooth)
        tri, bary = best
        bary = [max(0.0, v) for v in bary]
        s = sum(bary) or 1.0
        ln = sum((cn[li] * (bw / s) for li, bw in zip(tri.loops, bary)), mathutils.Vector())
        shade = (nmat @ ln).normalized() if ln.length > 0 else geo
        return geo, shade, bool(poly.use_smooth)
    finally:
        eval_obj.to_mesh_clear()


def _ao(sc, dg, point, normal, distance, rays):
    """(unoccluded share of cosine-weighted hemisphere rays about `normal`, nearest occluder name, its distance)."""
    from .ops_kit import _fibonacci_hemisphere                                             # noqa: PLC0415
    rot = normal.to_track_quat("Z", "Y")
    start = point + normal * max(distance * 1e-4, 1e-5)
    free, near, near_d = 0, None, None
    for d in _fibonacci_hemisphere(rays):
        hit, loc, _n, _i, obj, _m = sc.ray_cast(dg, start, rot @ mathutils.Vector(d), distance=distance)
        if not hit:
            free += 1
            continue
        dd = (loc - start).length
        if near_d is None or dd < near_d:
            near, near_d = obj.name, dd
    return free / float(rays), near, near_d


def _bsdf_inputs(mat):
    """The Principled BSDF the material output reads: each input's value, or what is linked into it."""
    if mat is None:
        return None, "no material on this face: it renders with Blender's default grey surface"
    if not mat.use_nodes or mat.node_tree is None:
        return {"useNodes": False}, None
    out = next((n for n in mat.node_tree.nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output), None) or \
        next((n for n in mat.node_tree.nodes if n.type == "OUTPUT_MATERIAL"), None)
    surf = out.inputs["Surface"] if out is not None else None
    if surf is None or not surf.is_linked:
        return {"surfaceLinked": False}, "the material output's Surface is not linked: the face renders black"
    node = surf.links[0].from_node
    row = {"surfaceNode": node.name, "surfaceType": node.type}
    bsdf = node if node.type == "BSDF_PRINCIPLED" else next(
        (n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        return row, None
    row["principled"] = bsdf.name
    for key, names in _PP_BSDF_INPUTS:
        sock = next((bsdf.inputs[n] for n in ((names,) if isinstance(names, str) else names) if n in bsdf.inputs), None)
        if sock is None:
            continue
        if sock.is_linked:
            row[key] = {"linkedFrom": sock.links[0].from_node.name, "nodeType": sock.links[0].from_node.type}
        elif hasattr(sock, "default_value"):
            v = sock.default_value
            row[key] = round(float(v), 4) if isinstance(v, float) else [round(float(c), 4) for c in v]
    return row, None


def op_render_pixel_probe(params):
    """What the camera sees through a pixel of the render, and the usual reasons it is black.

    params:
      x, y (int)                 the pixel, top-left origin, at the scene's render size (resolution x percentage); or
      pixels [[x, y], ...]       up to 16
      camera (str)               default the scene camera
      aoDistance (float)         the AO rays' length, default the world's AO distance (1 m without one)
      aoRays (int)               default 64
    Per pixel: hit, object, faceIndex (of the evaluated mesh), location, distance, geometricNormal, shadingNormal and the
    angle between them, backfaceHit (the camera sees the face's back), shadingFacesAway (the shading normal points away
    from the camera while the face does not - the classic black face), ao and aoGeometric (unoccluded share about each
    normal), nearestOccluder, material and its Principled inputs, and diagnosis - the findings in words. Writes nothing.
    """
    reject_unknown(params, _PP_KEYS, "render_pixel_probe")
    tail = "NOTHING was probed."
    sc = bpy.context.scene
    cam = sc.camera
    if params.get("camera") is not None:
        cam = get_object(take(params, "camera", kind=str))
        if cam.type != "CAMERA":
            raise MifOpError("'%s' is not a camera. %s" % (cam.name, tail))
    if cam is None:
        raise MifOpError("there is no scene camera - pass camera. %s" % tail)
    if params.get("pixels") is not None:
        if params.get("x") is not None or params.get("y") is not None:
            raise MifOpError("pass x and y, or pixels - not both. %s" % tail)
        raw = params["pixels"]
        if not isinstance(raw, list) or not raw or len(raw) > _PP_MAX:
            raise MifOpError("pixels is a list of 1-%d [x, y] pairs. %s" % (_PP_MAX, tail))
        pts = [finite_floats(list(p), "pixels") if isinstance(p, (list, tuple)) and len(p) == 2 else None for p in raw]
        if any(p is None for p in pts):
            raise MifOpError("each pixel is [x, y], got %r. %s" % (raw, tail))
    else:
        if params.get("x") is None or params.get("y") is None:
            raise MifOpError("x and y are required (top-left origin, at the render size) - or pixels. %s" % tail)
        pts = [[take_float(params, "x"), take_float(params, "y")]]
    r = sc.render
    w = int(r.resolution_x * r.resolution_percentage / 100.0)
    h = int(r.resolution_y * r.resolution_percentage / 100.0)
    for px, py in pts:
        if not (0 <= px < w and 0 <= py < h):
            raise MifOpError("pixel [%g, %g] is outside the %d x %d render. %s" % (px, py, w, h, tail))
    world = sc.world
    ao_default = float(getattr(getattr(world, "light_settings", None), "distance", 1.0) or 1.0)
    ao_dist = take_float(params, "aoDistance", default=ao_default)
    rays = take_int(params, "aoRays", default=64)
    if not ao_dist > 0 or not 4 <= rays <= 1024:
        raise MifOpError("aoDistance must be above 0 and aoRays 4-1024. %s" % tail)
    dg = bpy.context.evaluated_depsgraph_get()
    rows = []
    for px, py in pts:
        origin, d = camera_ray(sc, cam, px, py)
        hit, loc, nrm, idx, obj, _m = sc.ray_cast(dg, origin, d, distance=cam.data.clip_end)
        row = {"pixel": [px, py], "hit": bool(hit)}
        if not hit:
            row["diagnosis"] = ["the ray meets no geometry: the pixel shows the world background"]
            rows.append(row)
            continue
        ev = obj.evaluated_get(dg)
        geo, shade, smooth = _hit_detail(ev, idx, loc)
        to_cam = -d
        notes = []
        back = geo.dot(to_cam) < 0.0
        away = shade.dot(to_cam) < 0.0 and not back
        ang = math.degrees(geo.angle(shade)) if geo.length and shade.length else 0.0
        if back:
            notes.append("the camera sees the BACK of this face: its normal points away (inside-out or a single-sided "
                         "face seen from behind) - Cycles shades the back with the flipped normal, a black face's usual "
                         "cause on a closed part; recalculate outside or flip it")
        if away:
            notes.append("the shading normal points away from the camera while the face points at it (%.1f degrees "
                         "between them): custom or split normals, or a Normal input, bend it past the horizon - the "
                         "face renders dark" % ang)
        elif ang > 30.0:
            notes.append("the shading normal is %.1f degrees off the face's: smooth shading or custom normals over a "
                         "large angle darken it" % ang)
        ao, occ, occ_d = _ao(sc, dg, loc, shade if not back else -shade, ao_dist, rays)
        ao_geo, _o, _d = _ao(sc, dg, loc, geo if not back else -geo, ao_dist, rays)
        if ao < 0.15:
            notes.append("ambient occlusion is %.2f: the hemisphere above the point is closed%s - %s"
                         % (ao, " by '%s' %.4f away" % (occ, occ_d) if occ else "",
                            "the surface sits inside or against another surface" if (occ_d or 1) < ao_dist * 0.01
                            else "it is enclosed"))
        mat = None
        try:
            mesh = ev.data
            mi = mesh.polygons[idx].material_index if idx < len(mesh.polygons) else 0
            mat = ev.material_slots[mi].material if mi < len(ev.material_slots) else None
        except (AttributeError, IndexError):
            pass
        bsdf, bsdf_note = _bsdf_inputs(mat)
        if bsdf_note:
            notes.append(bsdf_note)
        if isinstance(bsdf, dict):
            bc = bsdf.get("baseColor")
            if isinstance(bc, list) and max(bc[:3]) < 0.02 and not bsdf.get("metallic"):
                notes.append("the Principled Base Color is black")
        row.update({"object": obj.name, "faceIndex": idx, "location": rnd(list(loc), 5),
                    "distance": round((loc - origin).length, 5), "geometricNormal": rnd(list(geo), 5),
                    "shadingNormal": rnd(list(shade), 5), "normalAngleDegrees": round(ang, 3), "smooth": smooth,
                    "backfaceHit": back, "shadingFacesAway": away, "ao": round(ao, 4), "aoGeometric": round(ao_geo, 4),
                    "nearestOccluder": occ, "nearestOccluderDistance": round(occ_d, 6) if occ_d is not None else None,
                    "material": mat.name if mat is not None else None, "bsdf": bsdf,
                    "diagnosis": notes or ["nothing here explains a black pixel: the face points at the camera, its "
                                           "shading normal agrees, the hemisphere is open and the material is lit - "
                                           "check the lights and the world (render_info blockers)"]})
        rows.append(row)
    return {"camera": cam.name, "renderSize": [w, h], "aoDistance": ao_dist, "aoRays": rays, "pixels": rows,
            "faceIndexNote": "faceIndex is the EVALUATED mesh's face: with modifiers that change topology it is not the "
                             "edit mesh's index"}


# --------------------------------------------------------------------------------------------- silhouette_stats
# PROJECTED-AREA SHARES (1.4, FOUND 2026-11-18, round 2): proportion rules - a 60/30/10 split of mass, an accent colour's
# share of the silhouette - needed the share of the silhouette each group of parts covers from a camera, and no op
# measured projected area; the build rasterised its own geometry driver-side (a mech's cannon and cockpit cluster came
# to about 9-10 % of its silhouette, the number that bounded the accent budget).

_SS_KEYS = ("groups", "camera", "direction", "resolution")
_SS_SELECT = ("objects", "pattern", "collection")


def world_bvh(objs, dg):
    """A BVHTree of the evaluated meshes of `objs` in world space, or None when they hold no face."""
    from mathutils.bvhtree import BVHTree                                                   # noqa: PLC0415
    verts, polys = [], []
    for o in objs:
        ev = o.evaluated_get(dg)
        try:
            mesh = ev.to_mesh()
        except RuntimeError:
            continue
        if mesh is None:
            continue
        try:
            base = len(verts)
            mw = ev.matrix_world
            verts += [mw @ v.co for v in mesh.vertices]
            polys += [[base + i for i in p.vertices] for p in mesh.polygons]
        finally:
            ev.to_mesh_clear()
    return BVHTree.FromPolygons(verts, polys) if polys else None


def _ss_groups(raw, tail):
    """[(name, [objects])] from groups {name: [names] | {objects | pattern | collection}} or [{name, ...}]."""
    from .ops_scene import visibility_selection                                             # noqa: PLC0415
    if isinstance(raw, list):
        items = []
        for g in raw:
            if not isinstance(g, dict) or not isinstance(g.get("name"), str):
                raise MifOpError("a groups entry is {name, objects | pattern | collection}, got %r. %s" % (g, tail))
            items.append((g["name"], {k: v for k, v in g.items() if k != "name"}))
    elif isinstance(raw, dict):
        items = list(raw.items())
    else:
        raise MifOpError("groups is {name: [object names] | {objects | pattern | collection}}, got %r. %s" % (raw, tail))
    if not items:
        raise MifOpError("groups names no group. %s" % tail)
    out, owner = [], {}
    for name, spec in items:
        sel = {"objects": spec} if isinstance(spec, list) else spec
        if not isinstance(sel, dict) or len([k for k in _SS_SELECT if sel.get(k) is not None]) != 1 \
                or set(sel) - set(_SS_SELECT):
            raise MifOpError("group '%s' is a list of names or one of {objects, pattern, collection}, got %r. %s"
                             % (name, spec, tail))
        try:
            objs, _s = visibility_selection(sel)
        except MifOpError as exc:
            raise MifOpError("group '%s': %s" % (name, str(exc).replace("NOTHING was changed.", tail)))
        meshes = [o for o in objs if o.type in ("MESH", "CURVE", "SURFACE", "META", "FONT")]
        if not meshes:
            raise MifOpError("group '%s' holds no mesh-like object (%s). %s" % (name, ", ".join(o.name for o in objs[:8]),
                                                                              tail))
        for o in meshes:
            if o.name in owner:
                raise MifOpError("'%s' is in group '%s' and group '%s' - each object counts in one group. %s"
                                 % (o.name, owner[o.name], name, tail))
            owner[o.name] = name
        out.append((str(name), meshes))
    return out


def op_silhouette_stats(params):
    """Each group of parts' share of the silhouette from a view - own projected area, the area left visible after the
    rest of the scene occludes it, and the share of the whole silhouette - in pixels at a stated resolution.

    params:
      groups (required)   {name: [object names] | {objects | pattern | collection}} - or [{name, objects | ...}]
      camera (str)        a camera to look through (default the scene camera), at the render's aspect; or
      direction [x,y,z]   look ALONG this direction, orthographic, framing every render-visible mesh
      resolution (int)    the longest edge in pixels, 32-1024, default 256
    Rays through every pixel centre against a BVH per group and one of every other render-visible mesh. Writes nothing.
    """
    reject_unknown(params, _SS_KEYS, "silhouette_stats")
    tail = "NOTHING was measured."
    sc = bpy.context.scene
    if params.get("groups") is None:
        raise MifOpError("groups is required: {name: [object names] | {objects | pattern | collection}}. %s" % tail)
    groups = _ss_groups(params["groups"], tail)
    res = take_int(params, "resolution", default=256)
    if not 32 <= res <= 1024:
        raise MifOpError("resolution is the longest edge, 32-1024 pixels, got %d. %s" % (res, tail))
    if params.get("direction") is not None and params.get("camera") is not None:
        raise MifOpError("pass camera or direction, not both. %s" % tail)
    dg = bpy.context.evaluated_depsgraph_get()
    grouped = {o.name for _n, objs in groups for o in objs}
    others = [o for o in _render_visible_meshes(sc, ("MESH", "CURVE", "SURFACE", "META", "FONT"))
              if o.name not in grouped]
    trees = [(n, world_bvh(objs, dg)) for n, objs in groups] + [("__others__", world_bvh(others, dg))]
    if params.get("direction") is not None:
        raw = params["direction"]
        if not isinstance(raw, (list, tuple)) or len(raw) != 3:
            raise MifOpError("direction is [x, y, z], got %r. %s" % (raw, tail))
        d = mathutils.Vector(finite_floats(list(raw), "direction"))
        if d.length == 0:
            raise MifOpError("direction is a zero vector. %s" % tail)
        d.normalize()
        pts = [o.matrix_world @ mathutils.Vector(c) for o in [x for _n, objs in groups for x in objs] + others
               for c in o.bound_box]
        rot = d.to_track_quat("-Z", "Y")
        inv = rot.inverted()
        local = [inv @ p for p in pts]
        lo = mathutils.Vector([min(p[i] for p in local) for i in range(3)])
        hi = mathutils.Vector([max(p[i] for p in local) for i in range(3)])
        span = max(hi.x - lo.x, hi.y - lo.y) * 1.05 or 1.0
        cx, cy = (lo.x + hi.x) / 2.0, (lo.y + hi.y) / 2.0
        w = h = res
        pix = span / res
        back = hi.z + 1.0
        view = {"kind": "direction", "direction": rnd(list(d), 5), "spanMeters": round(span, 5),
                "pixelMeters": round(pix, 6)}

        def ray(x, y):
            return rot @ mathutils.Vector((cx + (x + 0.5 - w / 2.0) * pix, cy - (y + 0.5 - h / 2.0) * pix, back)), d
    else:
        cam = sc.camera
        if params.get("camera") is not None:
            cam = get_object(take(params, "camera", kind=str))
            if cam.type != "CAMERA":
                raise MifOpError("'%s' is not a camera. %s" % (cam.name, tail))
        if cam is None:
            raise MifOpError("there is no scene camera - pass camera or direction. %s" % tail)
        r = sc.render
        rw, rh = r.resolution_x * r.pixel_aspect_x, r.resolution_y * r.pixel_aspect_y
        w, h = (res, max(1, int(round(res * rh / rw)))) if rw >= rh else (max(1, int(round(res * rw / rh))), res)
        sx, sy = r.resolution_x * r.resolution_percentage / 100.0 / w, r.resolution_y * r.resolution_percentage / 100.0 / h
        view = {"kind": "camera", "camera": cam.name}

        def ray(x, y):
            return camera_ray(sc, cam, (x + 0.5) * sx - 0.5, (y + 0.5) * sy - 0.5)
    own = [0] * len(trees)
    vis = [0] * len(trees)
    sil = 0
    t0 = time.time()
    for y in range(h):
        for x in range(w):
            o, dvec = ray(x, y)
            best, best_d = None, None
            for i, (_n, tree) in enumerate(trees):
                if tree is None:
                    continue
                hit = tree.ray_cast(o, dvec)
                if hit[0] is None:
                    continue
                own[i] += 1
                if best_d is None or hit[3] < best_d:
                    best, best_d = i, hit[3]
            if best is not None:
                vis[best] += 1
                sil += 1
    if sil == 0:
        raise MifOpError("nothing covers a pixel of the %d x %d view: the groups and the scene are out of frame. %s"
                         % (w, h, tail))
    rows = []
    for i, (name, objs) in enumerate(groups):
        rows.append({"group": name, "objects": [o.name for o in objs][:50], "objectCount": len(objs),
                     "projectedPx": own[i], "visiblePx": vis[i], "occludedShare": round(1.0 - vis[i] / own[i], 4)
                     if own[i] else None, "shareOfSilhouette": round(vis[i] / float(sil), 4)})
    other = {"objectCount": len(others), "projectedPx": own[-1], "visiblePx": vis[-1],
             "shareOfSilhouette": round(vis[-1] / float(sil), 4)}
    return {"view": view, "width": w, "height": h, "silhouettePx": sil, "groups": rows, "others": other,
            "seconds": round(time.time() - t0, 3),
            "note": "projectedPx is the group's own footprint, visiblePx what is left after everything nearer the view "
                    "(the other groups and every other render-visible mesh) covers it, shareOfSilhouette visiblePx over "
                    "the pixels anything covers. One ray per pixel centre: shares are good to about 1/sqrt(pixels)."}


# --------------------------------------------------------------------------------------------- look_studio
# A NEUTRAL REVIEW SETUP IN ONE CALL (1.4, FOUND 2026-11-18): the bundled studio.exr at any strength showed a bright ring
# right behind the subject in a 3/4 view, set_world has no "background hidden from the camera", and builds hand-made a
# plain backdrop with HDRI lighting or fell back to a flat world colour. look_studio lights with the HDRI, shows camera
# rays a flat backdrop colour instead (Light Path Is Camera Ray), and adds three lights, a floor and a camera framed from
# the subject's bounds - every object in one collection, removed as one group with the scene's world and camera put back.

_STUDIO_KEYS = ("subject", "hdri", "hdriStrength", "hdriRotation", "neutralHdri", "background", "lights", "lightScale",
            "floor", "camera", "azimuth", "elevation", "lens", "castCheck", "remove")
_LS = "MifLookStudio"


def _ls_remove(sc):
    """Remove the studio this op built: its collection's objects, the collection, its world and floor material; the
    world and camera it replaced go back. -> the names removed, or None when there is no studio."""
    coll = bpy.data.collections.get(_LS)
    if coll is None:
        return None
    prev_world, prev_cam = coll.get("mif_prev_world"), coll.get("mif_prev_camera")
    objs = list(coll.all_objects)
    datas = [o.data for o in objs if o.data is not None]
    removed = remove_objects(objs)
    for d in datas:
        try:
            if d.users != 0:
                continue
            for kind, store in ((bpy.types.Light, bpy.data.lights), (bpy.types.Camera, bpy.data.cameras),
                                (bpy.types.Mesh, bpy.data.meshes)):
                if isinstance(d, kind):
                    store.remove(d)
                    break
        except (ReferenceError, RuntimeError):
            pass
    bpy.data.collections.remove(coll)
    world = bpy.data.worlds.get(_LS + "_World")
    if world is not None and sc.world == world:
        sc.world = bpy.data.worlds.get(prev_world) if prev_world else None
    if world is not None:
        bpy.data.worlds.remove(world)
    mat = bpy.data.materials.get(_LS + "_Floor")
    if mat is not None and mat.users == 0:
        bpy.data.materials.remove(mat)
    if sc.camera is None and prev_cam and bpy.data.objects.get(prev_cam) is not None:
        sc.camera = bpy.data.objects[prev_cam]
    return {"removed": removed, "worldRestored": sc.world.name if sc.world else None,
            "cameraRestored": sc.camera.name if sc.camera else None}


def _ls_world(sc, hdri_path, strength, rotation, background, neutral):
    w = bpy.data.worlds.new(_LS + "_World")
    w.use_nodes = True
    nt = w.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputWorld")
    coord = nt.nodes.new("ShaderNodeTexCoord")
    mapping = nt.nodes.new("ShaderNodeMapping")
    mapping.inputs["Rotation"].default_value[2] = rotation
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(hdri_path, check_existing=True)
    lit = nt.nodes.new("ShaderNodeBackground")
    lit.inputs["Strength"].default_value = strength
    seen = nt.nodes.new("ShaderNodeBackground")
    seen.inputs["Color"].default_value = tuple(background) + (1.0,)
    seen.inputs["Strength"].default_value = 1.0
    path = nt.nodes.new("ShaderNodeLightPath")
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(coord.outputs["Generated"], mapping.inputs["Vector"])
    nt.links.new(mapping.outputs["Vector"], env.inputs["Vector"])
    if neutral:
        # THE HDRI'S OWN TINT OUT: the bundled studio.exr measured hue 201, strength 0.045 on the grey cast sphere
        # (5.0.1) - a review stage should not tint. Saturation 0 keeps where its light comes from and how bright.
        grey = nt.nodes.new("ShaderNodeHueSaturation")
        grey.inputs["Saturation"].default_value = 0.0
        nt.links.new(env.outputs["Color"], grey.inputs["Color"])
        nt.links.new(grey.outputs["Color"], lit.inputs["Color"])
    else:
        nt.links.new(env.outputs["Color"], lit.inputs["Color"])
    nt.links.new(path.outputs["Is Camera Ray"], mix.inputs[0])
    nt.links.new(lit.outputs[0], mix.inputs[1])
    nt.links.new(seen.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    return w


def op_look_studio(params):
    """A neutral review setup in one call: HDRI lighting the camera does not see, three lights, a floor and a camera
    framed on the subject - every object in the MifLookStudio collection, removable as one group.

    params:
      subject (str | [str])     the objects to light and frame - default every render-visible mesh
      hdri (str)                a bundled studio-light world name or an image file - default "studio"
      hdriStrength (float)      default 1.0;  hdriRotation (radians) default 0
      neutralHdri (bool)        the HDRI's colour taken out (saturation 0), its light kept - default true
      background [r,g,b]        what camera rays see in place of the HDRI - default [0.18, 0.18, 0.18]
      lights (bool)             key, fill and rim area lights, white, aimed at the subject - default true
      lightScale (float)        multiplies their energy - default 1.0
      floor (bool)              a grey plane under the subject - default true
      camera (bool)             a camera framed on the bounds, made the scene camera - default true
      azimuth / elevation (deg) the camera's direction from the front (-Y), default 35 and 15;  lens (mm) default 50
      castCheck                 true | {threshold}: measure the stage's colour cast on a grey sphere after building
      remove (bool)             take the studio out instead: its objects, collection, world and floor material, the
                                world and camera it replaced put back
    Building again replaces the studio this op built before (replaced: true).
    """
    reject_unknown(params, _STUDIO_KEYS, "look_studio")
    tail = "NOTHING was changed."
    sc = bpy.context.scene
    if take_bool(params, "remove", default=False):
        extra = sorted(k for k in params if k != "remove")
        if extra:
            raise MifOpError("remove takes no other key (%s given). %s" % (", ".join(extra), tail))
        gone = _ls_remove(sc)
        if gone is None:
            raise MifOpError("there is no %s collection to remove. %s" % (_LS, tail))
        return dict(gone, collection=_LS)
    raw = params.get("subject")
    if raw is None:
        subjects = [o for o in _render_visible_meshes(sc, ("MESH", "CURVE", "SURFACE", "META", "FONT"))
                    if not any(c.name == _LS for c in o.users_collection)]
    else:
        names = [raw] if isinstance(raw, str) else raw
        if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
            raise MifOpError("subject is an object name or a list of them, got %r. %s" % (raw, tail))
        subjects = [get_object(n) for n in names]
    if not subjects:
        raise MifOpError("there is no render-visible mesh to light and frame - pass subject. %s" % tail)
    from .ops_world import bundled_world_hdri                                                # noqa: PLC0415
    word = take(params, "hdri", default="studio", kind=str)
    hdri_path = bundled_world_hdri(word) or bpy.path.abspath(str(word))
    if not os.path.isfile(hdri_path):
        from .ops_world import bundled_world_hdris                                           # noqa: PLC0415
        raise MifOpError("hdri %r is neither a file nor one of this Blender's bundled worlds (%s). %s"
                         % (word, ", ".join(sorted(bundled_world_hdris())), tail))
    strength = take_float(params, "hdriStrength", default=1.0)
    rotation = take_float(params, "hdriRotation", default=0.0)
    neutral = take_bool(params, "neutralHdri", default=True)
    background = _rgb(params.get("background"), (0.18, 0.18, 0.18), "background", tail)
    want_lights = take_bool(params, "lights", default=True)
    scale = take_float(params, "lightScale", default=1.0)
    want_floor = take_bool(params, "floor", default=True)
    want_cam = take_bool(params, "camera", default=True)
    az = math.radians(take_float(params, "azimuth", default=35.0))
    el = math.radians(take_float(params, "elevation", default=15.0))
    lens = take_float(params, "lens", default=50.0)
    if strength < 0 or scale < 0 or not 1.0 <= lens <= 5000.0:
        raise MifOpError("hdriStrength and lightScale are 0 or more, lens 1-5000 mm. %s" % tail)
    cast = params.get("castCheck")
    if cast is not None and not isinstance(cast, (bool, dict)):
        raise MifOpError("castCheck is true or {threshold}, got %r. %s" % (cast, tail))
    cast_thr = None
    if cast:
        cast_thr = finite_float((cast if isinstance(cast, dict) else {}).get("threshold", 0.05), "castCheck.threshold")
        if not want_cam and sc.camera is None:
            raise MifOpError("castCheck renders through the scene camera, and there is none and camera is false. %s"
                             % tail)
        if "WORKBENCH" in sc.render.engine:
            raise MifOpError("castCheck measures lighting, and the scene renders with Workbench. %s" % tail)

    dg = bpy.context.evaluated_depsgraph_get()
    pts = [o.evaluated_get(dg).matrix_world @ mathutils.Vector(c) for o in subjects for c in o.evaluated_get(dg).bound_box]
    lo = mathutils.Vector([min(p[i] for p in pts) for i in range(3)])
    hi = mathutils.Vector([max(p[i] for p in pts) for i in range(3)])
    centre, radius = (lo + hi) / 2.0, max((hi - lo).length / 2.0, 1e-3)

    replaced = _ls_remove(sc) is not None
    coll = bpy.data.collections.new(_LS)
    sc.collection.children.link(coll)
    coll["mif_prev_world"] = sc.world.name if sc.world is not None else ""
    coll["mif_prev_camera"] = sc.camera.name if sc.camera is not None else ""
    made = []
    world = _ls_world(sc, hdri_path, strength, rotation, background, neutral)
    sc.world = world

    def place(obj, frm, to):
        obj.location = frm
        obj.rotation_mode = "XYZ"
        obj.rotation_euler = (mathutils.Vector(to) - mathutils.Vector(frm)).to_track_quat("-Z", "Y").to_euler()

    def at(az_r, el_r, dist):
        return centre + mathutils.Vector((math.sin(az_r) * math.cos(el_r), -math.cos(az_r) * math.cos(el_r),
                                          math.sin(el_r))) * dist
    lights = []
    if want_lights:
        d = 4.0 * radius
        for name, daz, del_, k in (("Key", 45.0, 40.0, 1.0), ("Fill", -60.0, 15.0, 0.35), ("Rim", 160.0, 35.0, 0.6)):
            data = bpy.data.lights.new(_LS + "_" + name, "AREA")
            data.size = radius * 1.5
            data.color = (1.0, 1.0, 1.0)
            data.energy = 5.0 * d * d * k * scale
            ob = bpy.data.objects.new(_LS + "_" + name, data)
            coll.objects.link(ob)
            place(ob, at(az + math.radians(daz), math.radians(del_), d), centre)
            lights.append({"name": ob.name, "energy": round(data.energy, 3), "size": round(data.size, 4)})
            made.append(ob.name)
    floor = None
    if want_floor:
        mesh = bpy.data.meshes.new(_LS + "_Floor")
        s = radius * 12.0
        mesh.from_pydata([(-s, -s, 0), (s, -s, 0), (s, s, 0), (-s, s, 0)], [], [(0, 1, 2, 3)])
        ob = bpy.data.objects.new(_LS + "_Floor", mesh)
        coll.objects.link(ob)
        ob.location = (centre.x, centre.y, lo.z)
        mat = bpy.data.materials.get(_LS + "_Floor") or bpy.data.materials.new(_LS + "_Floor")
        mat.use_nodes = True
        bsdf = next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if bsdf is not None:
            bsdf.inputs["Base Color"].default_value = tuple(background) + (1.0,)
            bsdf.inputs["Roughness"].default_value = 0.8
        mesh.materials.append(mat)
        floor = {"name": ob.name, "z": round(lo.z, 5), "size": round(2 * s, 4)}
        made.append(ob.name)
    cam_row = None
    if want_cam:
        cd = bpy.data.cameras.new(_LS + "_Camera")
        cd.lens = lens
        cd.sensor_fit = "AUTO"
        ob = bpy.data.objects.new(_LS + "_Camera", cd)
        coll.objects.link(ob)
        r = sc.render
        aspect = (r.resolution_y * r.pixel_aspect_y) / max(r.resolution_x * r.pixel_aspect_x, 1e-6)
        half_h = math.atan(cd.sensor_width / 2.0 / lens)
        half = math.atan(math.tan(half_h) * min(aspect, 1.0 / aspect))
        dist = radius / math.sin(half) * 1.1
        cd.clip_end = max(cd.clip_end, dist + radius * 20.0)
        place(ob, at(az, el, dist), centre)
        sc.camera = ob
        cam_row = {"name": ob.name, "location": rnd(list(ob.location), 5), "lens": lens, "distance": round(dist, 5)}
        made.append(ob.name)
    bpy.context.view_layer.update()
    out = {"collection": coll.name, "objects": made, "replaced": replaced, "world": world.name,
           "worldReplaced": coll["mif_prev_world"] or None, "hdri": hdri_path, "hdriStrength": strength,
           "neutralHdri": neutral,
           "background": list(background), "subject": [o.name for o in subjects][:50],
           "bounds": {"centre": rnd(list(centre), 5), "radius": round(radius, 5)}, "lights": lights, "floor": floor,
           "camera": cam_row,
           "note": "the HDRI lights the subject and shows in reflections; camera rays see the flat background colour "
                   "instead (Light Path Is Camera Ray), so the bright ring of the HDRI behind the subject is gone. "
                   "look_studio remove:true takes it all out and puts the world and camera back."}
    if cast_thr is not None:
        out["castCheck"] = cast_check(sc, cast_thr, tail="The studio was built.", bounds=(centre, radius))
    return out


# --------------------------------------------------------------------------------------------- highlight_point
# WHERE A LIGHT'S HIGHLIGHT LANDS (1.4, FOUND 2026-11-18, round 3): an eye's catch light was placed by reflecting the
# camera ray about the wanted normal driver-side and checked by a render with the other lights hidden by hand. This
# answers both questions - where the mirror highlight of a light lands on a surface for a camera, and where to put the
# light so it lands at a chosen point - and render_still soloLight renders that light alone.

_HP_KEYS = ("object", "light", "camera", "surfacePoint", "distance", "apply")


def _smooth_tris(obj, dg):
    """[(world corners, world corner normals)] of an evaluated mesh's loop triangles - shading normals, interpolated."""
    from .ops_kit import _corner_normals                                                   # noqa: PLC0415
    ev = obj.evaluated_get(dg)
    mesh = ev.to_mesh()
    try:
        mesh.calc_loop_triangles()
        cn = _corner_normals(mesh)
        mw = ev.matrix_world
        nm = mw.to_3x3().inverted_safe().transposed()
        out = []
        for t in mesh.loop_triangles:
            pts = [mw @ mesh.vertices[i].co for i in t.vertices]
            nrm = [(nm @ cn[li]).normalized() for li in t.loops]
            out.append((pts, nrm))
        return out
    finally:
        ev.to_mesh_clear()


def _light_dir(light, p):
    """Unit direction from p toward the light (a SUN's from its rotation), and the light's position (None for a sun)."""
    if light.data.type == "SUN":
        return (light.matrix_world.to_quaternion() @ mathutils.Vector((0.0, 0.0, 1.0))).normalized(), None
    pos = light.matrix_world.to_translation()
    return (pos - p).normalized(), pos


def op_highlight_point(params):
    """Where a light's mirror highlight lands on a surface for a camera - or, with surfacePoint, where to put the light
    so the highlight lands there.

    params:
      object (str, required)   the mesh the highlight lands on
      light (str, required)    the light (a SUN by its direction, any other by its position)
      camera (str)             default the scene camera
      surfacePoint [x,y,z]     INVERSE: the world point wanted (snapped to the surface): the light position whose mirror
                               highlight lands there - the camera ray reflected about the surface's shading normal
      distance (float)         inverse: how far from the point to put the light - default the light's current distance
      apply (bool)             inverse: move the light there and aim it at the point (default false: answer only)
    Forward: the point where the shading normal bisects the directions to the camera and to the light, searched over the
    evaluated mesh's triangles with interpolated normals, then refined inside the best ones; errorDegrees is the angle
    left between the normal and that bisector. visibleFromCamera and litByLight cast a ray each; pixel is where it lands
    in the render (top-left origin).
    """
    reject_unknown(params, _HP_KEYS, "highlight_point")
    tail = "NOTHING was changed."
    sc = bpy.context.scene
    obj = get_object(take(params, "object", kind=str, required=True))
    if obj.type not in ("MESH", "CURVE", "SURFACE", "META", "FONT"):
        raise MifOpError("'%s' is a %s - the highlight lands on a mesh. %s" % (obj.name, obj.type, tail))
    light = get_object(take(params, "light", kind=str, required=True))
    if light.type != "LIGHT":
        raise MifOpError("'%s' is a %s, not a light. %s" % (light.name, light.type, tail))
    cam = sc.camera
    if params.get("camera") is not None:
        cam = get_object(take(params, "camera", kind=str))
        if cam.type != "CAMERA":
            raise MifOpError("'%s' is not a camera. %s" % (cam.name, tail))
    if cam is None:
        raise MifOpError("there is no scene camera - pass camera. %s" % tail)
    inverse = params.get("surfacePoint") is not None
    for k in ("distance", "apply"):
        if params.get(k) is not None and not inverse:
            raise MifOpError("%s goes with surfacePoint (where to put the light). %s" % (k, tail))
    want_apply = take_bool(params, "apply", default=False)
    dist = take_float(params, "distance", default=None)
    if dist is not None and not dist > 0:
        raise MifOpError("distance must be above 0. %s" % tail)
    if inverse and want_apply and light.data.type == "SUN" and dist is not None:
        raise MifOpError("a SUN has no position, so distance does not apply - apply turns it. %s" % tail)
    want_point = None
    if inverse:
        raw = params["surfacePoint"]
        if not isinstance(raw, (list, tuple)) or len(raw) != 3:
            raise MifOpError("surfacePoint is [x, y, z], got %r. %s" % (raw, tail))
        want_point = mathutils.Vector(finite_floats(list(raw), "surfacePoint"))
    dg = bpy.context.evaluated_depsgraph_get()
    tris = _smooth_tris(obj, dg)
    if not tris:
        raise MifOpError("'%s' has no faces. %s" % (obj.name, tail))
    c_pos = cam.matrix_world.to_translation()

    def interp(tri, a, b):
        pts, nrm = tri
        c = 1.0 - a - b
        p = pts[0] * a + pts[1] * b + pts[2] * c
        n = (nrm[0] * a + nrm[1] * b + nrm[2] * c).normalized()
        return p, n
    out = {"object": obj.name, "light": light.name, "lightType": light.data.type, "camera": cam.name}
    if not inverse:
        def score(p, n):
            to_c = (c_pos - p).normalized()
            to_l, _pos = _light_dir(light, p)
            if n.dot(to_c) <= 0 or n.dot(to_l) <= 0:
                return -2.0
            return n.dot((to_c + to_l).normalized())
        ranked = sorted(((score(*interp(t, 1 / 3.0, 1 / 3.0)), i) for i, t in enumerate(tris)), reverse=True)[:12]
        best = None
        steps = 16
        for _s, i in ranked:
            for ia in range(steps + 1):
                for ib in range(steps + 1 - ia):
                    p, n = interp(tris[i], ia / float(steps), ib / float(steps))
                    s = score(p, n)
                    if best is None or s > best[0]:
                        best = (s, p, n)
        if best is None or best[0] <= -1.0:
            raise MifOpError("no point of '%s' faces both the camera and the light '%s', so no highlight lands on it. %s"
                             % (obj.name, light.name, tail))
        s, p, n = best
        to_l, lpos = _light_dir(light, p)
        out.update({"point": rnd(list(p), 5), "normal": rnd(list(n), 5),
                    "errorDegrees": round(math.degrees(math.acos(max(-1.0, min(1.0, s)))), 4)})
    else:
        hit = sc.ray_cast(dg, want_point + (c_pos - want_point).normalized() * 1e-3, (want_point - c_pos).normalized())
        best = None
        for t in tris:
            pts = t[0]
            cp = mathutils.geometry.closest_point_on_tri(want_point, *pts)
            dd = (cp - want_point).length
            if best is None or dd < best[0]:
                best = (dd, t, cp)
        _dd, t, p = best
        bary = mathutils.geometry.barycentric_transform(p, *t[0], mathutils.Vector((1, 0, 0)),
                                                         mathutils.Vector((0, 1, 0)), mathutils.Vector((0, 0, 1)))
        p, n = interp(t, bary[0], bary[1])
        view = (p - c_pos).normalized()
        if n.dot(-view) <= 0:
            raise MifOpError("the surface at that point faces away from the camera, so no highlight there can be seen. %s"
                             % tail)
        refl = (view - 2.0 * view.dot(n) * n).normalized()
        cur = (light.matrix_world.to_translation() - p).length
        d = dist if dist is not None else max(cur, 1e-3)
        lpos = None if light.data.type == "SUN" else p + refl * d
        out.update({"point": rnd(list(p), 5), "normal": rnd(list(n), 5), "snappedBy": round(_dd, 6),
                    "reflectedDirection": rnd(list(refl), 5),
                    "lightPosition": rnd(list(lpos), 5) if lpos is not None else None, "distance": round(d, 5)})
        if hit[0] and (hit[1] - p).length > 1e-3:
            out["warning"] = "the camera sees '%s' in front of that point, not this surface" % hit[4].name
        if want_apply:
            from .ops_lightcam import _aim_error_degrees, _look_at_euler                   # noqa: PLC0415
            from .ops_common import basis_space, write_orientation                          # noqa: PLC0415
            if lpos is not None:
                space = basis_space(light)
                light.location = (space.inverted() @ lpos) if space is not None else lpos
                bpy.context.view_layer.update()
                write_orientation(light, _look_at_euler(tuple(light.matrix_world.to_translation()), tuple(p)), space)
            else:
                space = basis_space(light)
                write_orientation(light, _look_at_euler((0.0, 0.0, 0.0), tuple(-refl)), space)
            out["applied"] = True
            out["aimErrorDegrees"] = _aim_error_degrees(light, p) if lpos is not None else None
        to_l, lpos = (refl, None) if lpos is None else ((lpos - p).normalized(), lpos)
    # THE TWO VISIBILITY RAYS: the camera sees the point, the light reaches it
    p = mathutils.Vector(out["point"])
    n = mathutils.Vector(out["normal"])
    cam_hit = sc.ray_cast(dg, c_pos, (p - c_pos).normalized())
    out["visibleFromCamera"] = bool(cam_hit[0]) and (cam_hit[1] - p).length < 1e-3 * max(1.0, (p - c_pos).length)
    start = p + n * 1e-4
    to_l, lpos_now = _light_dir(light, p) if not inverse else (to_l, lpos)
    far = (lpos_now - start).length if lpos_now is not None else 1e6
    lhit = sc.ray_cast(dg, start, to_l, distance=far)
    out["litByLight"] = not lhit[0]
    if not out["litByLight"]:
        out["shadowedBy"] = lhit[4].name
    from bpy_extras.object_utils import world_to_camera_view                               # noqa: PLC0415
    ndc = world_to_camera_view(sc, cam, p)
    r = sc.render
    w = r.resolution_x * r.resolution_percentage / 100.0
    h = r.resolution_y * r.resolution_percentage / 100.0
    out["pixel"] = [round(ndc.x * w - 0.5, 2), round((1.0 - ndc.y) * h - 0.5, 2)]
    out["inFrame"] = 0.0 <= ndc.x <= 1.0 and 0.0 <= ndc.y <= 1.0 and ndc.z > 0
    if not out["visibleFromCamera"]:
        out.setdefault("warning", "the camera does not see this point (something nearer covers it, or the mesh is "
                                  "inside-out and the far side's normals face the camera) - the highlight does not show")
    out["note"] = ("the mirror direction: a rough surface spreads the highlight around it. render_still soloLight renders "
                   "this light alone to see it.")
    return out


OPS = {
    "set_freestyle": op_set_freestyle,
    "set_sun_position": op_set_sun_position,
    "render_viewport": op_render_viewport,
    "highlight_point": op_highlight_point,
    "look_studio": op_look_studio,
    "silhouette_stats": op_silhouette_stats,
    "render_pixel_probe": op_render_pixel_probe,
}
