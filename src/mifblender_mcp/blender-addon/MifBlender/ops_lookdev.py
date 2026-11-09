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

from .ops_common import (MifOpError, check_output_path, finite_float, finite_floats, get_object, reject_unknown,
                         require_editable, take, take_bool, take_float, take_int)


# --------------------------------------------------------------------------------------------- set_freestyle

_FS_KEYS = ("enabled", "viewLayer", "lineThickness", "thicknessMode", "creaseAngle", "lineSets", "removeLineSets",
            "edgeMarks")
_EDGE_TYPES = {"SILHOUETTE": "select_silhouette", "BORDER": "select_border", "CREASE": "select_crease",
               "RIDGE_VALLEY": "select_ridge_valley", "SUGGESTIVE_CONTOUR": "select_suggestive_contour",
               "MATERIAL_BOUNDARY": "select_material_boundary", "CONTOUR": "select_contour",
               "EXTERNAL_CONTOUR": "select_external_contour", "EDGE_MARK": "select_edge_mark"}
_LS_KEYS = ("name", "edgeTypes", "collection", "color", "thickness", "alpha", "visibility")


def _lineset_row(ls):
    st = ls.linestyle
    return {"name": ls.name, "show": bool(ls.show_render),
            "edgeTypes": sorted(k for k, a in _EDGE_TYPES.items() if getattr(ls, a, False)),
            "collection": ls.collection.name if ls.select_by_collection and ls.collection else None,
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
                               thickness, alpha, visibility VISIBLE | HIDDEN | RANGE}] - a line set of that name is
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
            if s.get("collection") is not None and bpy.data.collections.get(s["collection"]) is None:
                raise MifOpError("no collection named '%s'. %s" % (s["collection"], tail))
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
            "resolutionY", "percentage", "overlays")


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
    shading = str(take(params, "shading", default="SOLID", kind=str)).upper()
    ctype = str(take(params, "colorType", default="MATERIAL", kind=str)).upper()
    r = sc.render
    saved = {"engine": r.engine, "camera": sc.camera, "x": r.resolution_x, "y": r.resolution_y,
             "pct": r.resolution_percentage, "fp": r.filepath, "fmt": r.image_settings.file_format,
             "frame": sc.frame_current}
    win = bpy.context.window
    view3d = None
    if not bpy.app.background and win is not None:
        view3d = next((a for a in win.screen.areas if a.type == "VIEW_3D"), None)
    route = "viewport" if view3d is not None else "workbench"
    t0 = time.time()
    written = []
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
        for f, p in zip(frames, paths):
            sc.frame_set(f)
            r.filepath = p
            if route == "viewport":
                space = view3d.spaces.active
                keep = space.overlay.show_overlays
                space.overlay.show_overlays = take_bool(params, "overlays", default=False)
                region = next(rg for rg in view3d.regions if rg.type == "WINDOW")
                try:
                    with bpy.context.temp_override(window=win, area=view3d, region=region):
                        bpy.ops.render.opengl(write_still=True, view_context=False)
                finally:
                    space.overlay.show_overlays = keep
            else:
                bpy.ops.render.render(write_still=True)
            if os.path.isfile(p) and os.path.getmtime(p) >= t0 - 1:
                written.append(p)
    finally:
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
                                         "headless: a Workbench render with solid shading stands in for the viewport"),
           "frames": len(frames), "framesWritten": len(written), "files": written[:20], "camera": cam.name,
           "seconds": round(secs, 3), "secondsPerFrame": round(secs / max(len(frames), 1), 4)}
    if route != "viewport" and params.get("overlays"):
        out["overlaysNote"] = ("overlays were not drawn: headless has no viewport, and the Workbench render that stands in "
                               "for it has no overlays. A session with a 3D View draws them.")
    if len(written) != len(frames):
        out["warning"] = "%d of %d frames reached disk" % (len(written), len(frames))
    return out


OPS = {
    "set_freestyle": op_set_freestyle,
    "set_sun_position": op_set_sun_position,
    "render_viewport": op_render_viewport,
}
