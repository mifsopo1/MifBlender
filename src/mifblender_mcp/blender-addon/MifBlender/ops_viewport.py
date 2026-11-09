"""Viewport shading and framing - what makes the work VISIBLE while it is happening.

WHY THIS IS A REAL GAP AND NOT A CONVENIENCE. A recording of a build drew the report "it was all
grey and I couldn't see any of that happening in real time" - while three fluorescent fixtures, a
window shaft and two practicals were being created correctly. Nothing was wrong with the lighting.
The viewport was in SOLID shading, which ignores materials and lights by design, so the entire
lighting stage was invisible to the only person watching.

A bridge that can light a scene and cannot show it has not finished the job. Every other op here
changes what is IN the file; this one changes what the human can see of it, and for a tool driven
by an agent on somebody else's screen that is not a lesser thing.

MATERIAL vs RENDERED, since picking the wrong one is the usual disappointment: MATERIAL preview
uses a built-in studio light and shows materials but NOT your scene's lamps. Only RENDERED runs the
actual render engine, so it is the only mode in which a flickering light flickers.
"""
import bpy
from mathutils import Vector

from .ops_common import (MifOpError, check_axis_dict, finite_floats, get_object, reject_unknown,
                         take, take_bool, take_float)

_SHADE_KEYS = {"shading", "mode", "studioLight", "useSceneLights", "useSceneWorld",
               "showOverlays", "showGizmos", "colorType", "showXray", "showCavity", "showGrid"}
# NO "all". Framing everything is what this op does when no object is named - view3d.view_all
# is the else branch - so "all" named the DEFAULT and was read nowhere. Worse in combination:
# object plus all:true is a contradiction the caller can express, and the object silently
# won. Omitting object is the way to say it, and a caller who sends all now gets a refusal
# instead of agreement.
_FRAME_KEYS = {"object", "name", "camera"}
# No "target" alias for focus. It was accepted and no tool sent it, which param_reach
# correctly called unreachable - and a second name for the same thing is not worth a
# baseline entry. One name, one meaning.
_VIEW_KEYS = {"focus", "distance", "azimuth", "elevation", "lookFrom",
              "perspective", "lens"}


def _view3d_spaces():
    """Every 3D viewport in the current screen. There may be none in a headless run."""
    out = []
    screen = getattr(bpy.context, "screen", None)
    if screen is None:
        return out
    for area in screen.areas:
        if area.type != "VIEW_3D":
            continue
        for sp in area.spaces:
            if sp.type == "VIEW_3D":
                out.append((area, sp))
    return out


def op_set_viewport_shading(params):
    """Set the 3D viewport's shading mode, and report how many viewports actually changed.

    params:
      shading (alias mode)  WIREFRAME | SOLID | MATERIAL | RENDERED
      useSceneLights        RENDERED/MATERIAL: use the scene's own lamps rather than a studio light
      useSceneWorld         RENDERED/MATERIAL: use the scene's world rather than a studio HDRI
      studioLight           name of the studio light, when not using scene lights
      showOverlays          grids, outlines, relationship lines
      showGizmos
      colorType             MATERIAL | OBJECT | RANDOM | SINGLE | TEXTURE - SOLID mode only
      showXray              X-ray: see and select through the surface (shading.show_xray, 2026-10-04)
      showCavity            cavity shading, which shows creases and topology mistakes (shading.show_cavity)
      showGrid              the floor grid overlay: overlay.show_floor (perspective) and show_ortho_grid (the
                            orthographic views), both; a backdrop image is judged with it off (2026-10-05)
    """
    reject_unknown(params, _SHADE_KEYS, "set_viewport_shading")
    spaces = _view3d_spaces()
    if not spaces:
        raise MifOpError("there is no 3D viewport in this Blender - a background/headless session "
                         "has no screen at all, so there is nothing to shade. NOTHING was changed.")

    want = take(params, "shading", "mode", default=None, kind=str)
    if want:
        want = str(want).upper()
        valid = {"WIREFRAME", "SOLID", "MATERIAL", "RENDERED"}
        if want not in valid:
            raise MifOpError("shading must be one of %s, got '%s'. NOTHING was changed."
                             % (", ".join(sorted(valid)), want))

    # VALIDATED BEFORE THE LOOP, exactly like `shading` above, and a red row has been carrying the
    # reason since it was written. `sh.color_type = str(ct).upper()` sat INSIDE the write loop with
    # no wrapper - the outlier of three, since `shading` validates up here and `studioLight` catches
    # its RNA error and rewraps it - so an invalid value surfaced Blender's own TypeError with the
    # parameter's name nowhere in it. Worse, `sh.type = want` had already run in the same iteration,
    # so the call reported failure AND changed the scene. Measured on 4.4.0:
    #
    #     {mode: WIREFRAME, colorType: MifNoSuchColorType} -> ok:false, viewport on WIREFRAME
    #
    # A refusal that half-applied is the shape this file refuses everywhere else, and "NOTHING was
    # changed" is only true if the check runs before the first write.
    #
    # THE VALID SET IS READ OFF THE RNA rather than remembered - the same discipline ops_lightcam
    # uses for _LIGHT_KEYS, and for the same reason: the enum moves between builds and a hardcoded
    # list goes stale silently. If the property cannot be found at all the check stands aside rather
    # than inventing a refusal, because a reader that cannot see the enum is not a reader that
    # proved the value wrong.
    ct = take(params, "colorType", default=None, kind=str)
    if ct:
        ct = str(ct).upper()
        _rna = spaces[0][1].shading.bl_rna.properties.get("color_type")
        _allowed = {e.identifier for e in getattr(_rna, "enum_items", ())} if _rna else set()
        if _allowed and ct not in _allowed:
            raise MifOpError("colorType must be one of %s, got '%s'. NOTHING was changed."
                             % (", ".join(sorted(_allowed)), ct))

    # THE FLAGS TOO, BEFORE THE LOOP (2026-09-21). They were read inside it after sh.type had been written, so
    # {shading: WIREFRAME, showOverlays: "flase"} switched the first viewport to wireframe and refused. studioLight
    # cannot be checked in advance - Blender validates it on assignment - so every viewport is snapshotted, and a
    # refused light puts back all of them; it used to leave the first one changed and the rest not.
    flags = {k: take_bool(params, k, default=True) for k in ("useSceneLights", "useSceneWorld", "showOverlays",
                                                             "showGizmos", "showXray", "showCavity", "showGrid") if k in params}
    sl = take(params, "studioLight", default=None, kind=str)
    shade_attrs = ("type", "use_scene_lights", "use_scene_lights_render", "use_scene_world",
                   "use_scene_world_render", "studio_light", "color_type", "show_xray", "show_cavity")
    snapshot = [(sp, {a: getattr(sp.shading, a) for a in shade_attrs if hasattr(sp.shading, a)},
                 sp.overlay.show_overlays, sp.show_gizmo, (sp.overlay.show_floor, sp.overlay.show_ortho_grid))
                for _area, sp in spaces]

    changed, before = 0, spaces[0][1].shading.type
    try:
        for _area, sp in spaces:
            sh = sp.shading
            if want:
                sh.type = want
            if "useSceneLights" in flags:
                for attr in ("use_scene_lights", "use_scene_lights_render"):
                    if hasattr(sh, attr):
                        setattr(sh, attr, flags["useSceneLights"])
            if "useSceneWorld" in flags:
                for attr in ("use_scene_world", "use_scene_world_render"):
                    if hasattr(sh, attr):
                        setattr(sh, attr, flags["useSceneWorld"])
            if sl:
                try:
                    sh.studio_light = str(sl)
                except (TypeError, ValueError) as exc:
                    raise MifOpError("studioLight: no studio light called '%s' in this Blender: %s. NOTHING was "
                                     "changed - every viewport was put back." % (sl, exc))
            if ct:
                # Already taken, upper-cased and checked against the RNA enum above the loop.
                sh.color_type = ct
            if "showXray" in flags:
                sh.show_xray = flags["showXray"]
            if "showCavity" in flags:
                sh.show_cavity = flags["showCavity"]
            if "showOverlays" in flags:
                sp.overlay.show_overlays = flags["showOverlays"]
            if "showGizmos" in flags:
                sp.show_gizmo = flags["showGizmos"]
            if "showGrid" in flags:
                sp.overlay.show_floor = flags["showGrid"]
                sp.overlay.show_ortho_grid = flags["showGrid"]
            changed += 1
    except MifOpError:
        for sp, attrs, overlays, gizmo, grid in snapshot:
            for a, v in attrs.items():
                try:
                    setattr(sp.shading, a, v)
                except (TypeError, ValueError, AttributeError):   # noqa: PERF203 - put back what can be
                    pass
            sp.overlay.show_overlays = overlays
            sp.show_gizmo = gizmo
            sp.overlay.show_floor, sp.overlay.show_ortho_grid = grid
        raise

    now = spaces[0][1].shading
    return {
        "viewportsChanged": changed,
        "before": before,
        "shading": now.type,
        "useSceneLights": bool(getattr(now, "use_scene_lights", False)),
        "useSceneWorld": bool(getattr(now, "use_scene_world", False)),
        # THE TWO FLAGS THIS OP WROTE AND NEVER REPORTED. useSceneLights and useSceneWorld were
        # always here; showOverlays and showGizmos were set and then invisible, so the
        # inert-parameter probe read both as doing nothing. From the first viewport, like the rest.
        "showOverlays": bool(spaces[0][1].overlay.show_overlays),
        "showGizmos": bool(spaces[0][1].show_gizmo),
        # AND THE TWO THIS OP WROTE THAT NOTHING READ (advisory 12; audit_write_only_state, 2026-09-21). Called with no
        # parameters this op writes nothing and answers the current shading, so it is the reader for all of them.
        "colorType": getattr(now, "color_type", None),
        "studioLight": getattr(now, "studio_light", None),
        "showXray": bool(getattr(now, "show_xray", False)),
        "showCavity": bool(getattr(now, "show_cavity", False)),
        # true when either grid shows: the floor (perspective) or the orthographic grid; both are written together
        "showGrid": bool(spaces[0][1].overlay.show_floor or spaces[0][1].overlay.show_ortho_grid),
        "modeNote": ("MATERIAL preview uses a studio light and does NOT show your scene's lamps; "
                     "only RENDERED runs the render engine, so it is the only mode in which a "
                     "flickering light actually flickers."),
    }


def _view_eye(r3d):
    """Where the viewport is looking FROM, derived rather than read off a matrix.

    THIS USED TO CALL RegionView3D.update() AND TAKE THE EYE OFF view_matrix, AND THAT KILLED THE
    PROCESS. On a headless Blender that has not rendered yet, update() is an
    EXCEPTION_ACCESS_VIOLATION inside GPU_matrix_frustum_set, reached through
    rna_RegionView3D_update -> ED_view3d_update_viewmat: it writes into a GPU matrix stack that a
    background session has not built. An access violation is not a Python exception, so there was
    nothing to catch - blender.exe died mid-answer and the caller saw a closed socket. Ledger B13.

    A RENDER IS WHAT MAKES IT SAFE, and that is why nothing caught it for so long: by the time
    test_blender_scene reaches its viewport rows it has rendered several times, so the ops pass
    there and crash for anyone who calls them first. Measured on 4.4: fresh session crashes, after
    one 32x32 render it survives, and a new scene does NOT help. There is no readable difference
    either - area 1586x954, WINDOW region 1586x928 and window_matrix[0][0] 1.388889 are identical
    before and after the render - so no property-based guard was available to write.

    SO THE MATRIX IS NOT CONSULTED AT ALL NOW. The viewport is an orbit and the eye follows from
    the orbit, which is stored state needing no redraw and no GPU. Measured against
    view_matrix.inverted().translation on 3.6.23 / 4.4.0 / 5.2.1, in a session that had rendered so
    both answers existed:

        PERSP    view_location + view_rotation @ (0, 0, view_distance)    exact to 2.1e-06
        CAMERA   the scene camera's world position                        exact
        ORTHO    view_location itself - there is no depth offset          exact

    All three branches are required. ORTHO is the one that reads wrong: it is neither the orbit nor
    the camera, and using the orbit form there would have put the eye 9.6 units from where the view
    actually is, in a response that reports it as fact.

    It also removes the staleness the 2026-09-13 note was about - the eye no longer depends on
    anything having redrawn, rather than depending on update() having been called to force it."""
    mode = r3d.view_perspective
    if mode == "CAMERA":
        cam = getattr(bpy.context.scene, "camera", None)
        if cam is not None:
            return cam.matrix_world.translation
        # No scene camera to look through. Falling back to the pivot rather than inventing a
        # position: frame_viewport refuses this case outright, so it is reachable only if something
        # cleared the camera between the two calls.
        return r3d.view_location
    if mode == "ORTHO":
        return r3d.view_location
    return r3d.view_location + (r3d.view_rotation @ Vector((0.0, 0.0, float(r3d.view_distance))))


def _view_readback(r3d, sp):
    """The view as it IS - one read-back for both ops."""
    eye = _view_eye(r3d)
    return {
        "focus": [round(v, 4) for v in r3d.view_location],
        "distance": round(float(r3d.view_distance), 4),
        "eye": [round(v, 4) for v in eye],
        "perspective": r3d.view_perspective,
        "lens": round(float(sp.lens), 3),
    }


def op_frame_viewport(params):
    """Point the viewport at something - the whole scene, one object, or through the camera.

    Framing is not cosmetic when somebody is watching: a build that happens off-screen is a build
    nobody can see happen.
    """
    reject_unknown(params, _FRAME_KEYS, "frame_viewport")
    spaces = _view3d_spaces()
    if not spaces:
        raise MifOpError("there is no 3D viewport in this Blender. NOTHING was changed.")
    area, sp = spaces[0]

    if take_bool(params, "camera", default=False):
        if bpy.context.scene.camera is None:
            raise MifOpError("there is no scene camera to look through. NOTHING was changed.")
        sp.region_3d.view_perspective = "CAMERA"
        return dict(_view_readback(sp.region_3d, sp), view="CAMERA", camera=bpy.context.scene.camera.name)

    name = take(params, "object", "name", default=None, kind=str)
    region = next((r for r in area.regions if r.type == "WINDOW"), None)
    if region is None:
        raise MifOpError("the viewport has no WINDOW region to frame into. NOTHING was changed.")

    try:
        with bpy.context.temp_override(area=area, region=region, space_data=sp,
                                       screen=bpy.context.screen, scene=bpy.context.scene,
                                       view_layer=bpy.context.view_layer):
            if name:
                obj = get_object(name)
                for o in bpy.context.view_layer.objects:
                    o.select_set(False)
                obj.select_set(True)
                bpy.context.view_layer.objects.active = obj
                bpy.ops.view3d.view_selected()
            else:
                bpy.ops.view3d.view_all()
            sp.region_3d.view_perspective = "PERSP"
    except RuntimeError as exc:
        raise MifOpError("could not frame the viewport: %s" % exc)
    # THE VIEW IT PRODUCED, not only what it was asked to frame (2026-09-13): the pivot and distance after
    # view_selected / view_all are what a caller can check - framing one cube must put the pivot on it.
    return dict(_view_readback(sp.region_3d, sp), view="PERSP", framed=name or "all objects",
                objectCount=len(bpy.context.view_layer.objects))


def op_set_viewport_view(params):
    """Place the viewport's own camera - the orbit pivot, the distance and the angle.

    WHY THIS EXISTS SEPARATELY FROM frame_viewport. Framing answers "show me this object"; this
    answers "stand HERE and look THERE", which is what a walkthrough needs. The view has to move itself
    while a scene is being built so nobody has to drive it, and framing
    alone cannot do a slow push down a room.

    THE VIEWPORT IS AN ORBIT, NOT A CAMERA. region_3d has view_location (the pivot), view_distance
    (how far back) and view_rotation (a quaternion). There is no "position" to set directly - the
    eye is derived from those three, so a caller who thinks in eye-positions gets lookFrom, which
    is converted here rather than left as an exercise.

    params:
      focus {x,y,z}            the point being orbited - what you are looking AT
      distance (float)         metres back from the focus
      azimuth (float)          RADIANS around Z. 0 looks along +Y, increasing turns anticlockwise
      elevation (float)        RADIANS above the horizon; positive looks DOWN at the focus
      lookFrom {x,y,z}         eye position INSTEAD of azimuth/elevation/distance - all three are
                               derived from it, and passing both is refused
      perspective              PERSP | ORTHO | CAMERA
      lens (float)             the view's focal length in mm, within the 3D view's own range (1 to 250); outside it
                               is refused rather than clamped
    """
    import math
    import mathutils

    reject_unknown(params, _VIEW_KEYS, "set_viewport_view")
    # RANGE-CHECKED against the view's own limits, read off the type's RNA, before anything is looked up: written
    # unchecked, Blender clamped it with no refusal and the reply read the clamped value back (found 2026-10-25).
    lens = take_float(params, "lens", default=None)
    if lens is not None:
        prop = bpy.types.SpaceView3D.bl_rna.properties["lens"]
        if not prop.hard_min <= lens <= prop.hard_max:
            raise MifOpError("lens must be %g to %g mm for a 3D view, got %g. NOTHING was changed."
                             % (prop.hard_min, prop.hard_max, lens))
    spaces = _view3d_spaces()
    if not spaces:
        raise MifOpError("there is no 3D viewport in this Blender. NOTHING was changed.")
    _area, sp = spaces[0]
    r3d = sp.region_3d

    have_polar = any(k in params for k in ("azimuth", "elevation", "distance"))
    if "lookFrom" in params and have_polar:
        raise MifOpError("pass lookFrom OR azimuth/elevation/distance, not both - lookFrom already "
                         "determines all three. NOTHING was changed.")

    # EVERYTHING IS VALIDATED BEFORE ANY OF IT IS APPLIED. This block used to write
    # r3d.view_location as soon as `focus` was parsed and then run FOUR more refusals below it,
    # three of them ending "NOTHING was changed" - so {"focus":[1,2,3],"lookFrom":[1,2,3]} moved
    # the viewport pivot and then told the caller nothing had happened. Same for a negative
    # distance, and for perspective:"CAMERA" with no scene camera, which could fire after the
    # pivot, the distance AND the rotation had all been set.
    #
    # That sentence is what every refusal in MifBridge is held to, so a false one is worse than an
    # ordinary bug. The house rule is that a refusal fires before a mutation; this now obeys it by
    # computing the whole target state first and committing it only once nothing can still refuse.
    focus = params.get("focus")
    f = _vec3_of(focus, "focus") if focus is not None else tuple(r3d.view_location)

    new_distance = None
    new_rotation = None
    if "lookFrom" in params:
        eye = _vec3_of(params["lookFrom"], "lookFrom")
        d = mathutils.Vector((f[0] - eye[0], f[1] - eye[1], f[2] - eye[2]))
        if d.length == 0.0:
            raise MifOpError("lookFrom is the focus point, so there is no direction to look. "
                             "NOTHING was changed.")
        new_distance = d.length
        # The view looks down its own -Z, same convention as a camera object.
        new_rotation = d.to_track_quat("-Z", "Y")
    else:
        dist = take_float(params, "distance", default=None)
        if dist is not None:
            if dist <= 0:
                raise MifOpError("distance must be positive, got %r. NOTHING was changed." % dist)
            new_distance = dist
        az = take_float(params, "azimuth", default=None)
        el = take_float(params, "elevation", default=None)
        if az is not None or el is not None:
            az = 0.0 if az is None else az
            el = 0.35 if el is None else el
            # Direction from eye toward focus, from polar angles: azimuth 0 looks along +Y.
            d = mathutils.Vector((math.sin(az) * math.cos(el),
                                  math.cos(az) * math.cos(el),
                                  -math.sin(el)))
            new_rotation = d.to_track_quat("-Z", "Y")

    persp = take(params, "perspective", default=None, kind=str)
    want = None
    if persp:
        want = str(persp).upper()
        if want not in ("PERSP", "ORTHO", "CAMERA"):
            raise MifOpError("perspective must be PERSP, ORTHO or CAMERA, got '%s'. "
                             "NOTHING was changed." % persp)
        if want == "CAMERA" and bpy.context.scene.camera is None:
            raise MifOpError("there is no scene camera to look through. NOTHING was changed.")

    # COMMIT. Nothing below here can refuse.
    if focus is not None:
        r3d.view_location = f
    if new_distance is not None:
        r3d.view_distance = new_distance
    if new_rotation is not None:
        r3d.view_rotation = new_rotation
    if want:
        r3d.view_perspective = want
    if lens is not None:
        sp.lens = lens

    return _view_readback(r3d, sp)


def _vec3_of(v, key):
    if isinstance(v, dict):
        # AT LEAST ONE OF x/y/z, AND NOTHING ELSE - see ops_create for why. Fifth copy of this
        # parser in the addon and the defect was in every one: a dict read with .get(axis,
        # default) turns {"mif":"typo"} into the DEFAULT vector and reports success.
        check_axis_dict(v, key, ("x", "y", "z"))
        return tuple(finite_floats([v.get("x", 0.0), v.get("y", 0.0), v.get("z", 0.0)], key))
    if isinstance(v, (list, tuple)) and len(v) == 3:
        return tuple(finite_floats(v, key))
    raise MifOpError("'%s' must be {x,y,z} or a 3-list, got %r." % (key, v))


OPS = {
    "set_viewport_shading": op_set_viewport_shading,
    "set_viewport_view": op_set_viewport_view,
    "frame_viewport": op_frame_viewport,
}
