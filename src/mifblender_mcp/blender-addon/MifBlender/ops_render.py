"""Render settings and a still render - the gap that turns the other three into a picture.

WHY THIS ONE NEXT. Lights, cameras and keyframes were closed first because they gate everything;
rendering is what makes them visible. Without it an agent can build and light a scene and still
cannot produce a single image without leaving the typed path.

=============================================================================
RENDERING BLOCKS THE MAIN THREAD, AND THAT IS NOT A DETAIL
=============================================================================
This addon dispatches on Blender's main thread, so a render holds it for the whole exposure -
Blender is frozen and the bridge answers nothing until it finishes. A Cycles render at default
samples on a real scene is minutes, which will hit the addon's job timeout and look like a hung
bridge rather than a slow render.

So: samples and resolution are FIRST-CLASS parameters rather than buried, the response reports how
long it actually took, and the docstring says plainly to start small. A caller who wants a
production frame should set the settings here and press F12 themselves.

=============================================================================
ok:true IS NOT AN IMAGE
=============================================================================
bpy.ops.render.render(write_still=True) returns {'FINISHED'} whether or not a file appeared - a bad
path, a permissions problem or a disabled file format all still report FINISHED. So this stats the
file afterwards and reports its real size, the same way the UE arm's ui_scenario_capture does:
"wroteFile" is a measurement, not the operator's opinion.
"""
import base64
import math
import tempfile
import os
import re
import subprocess
import time

import bpy
from mathutils import Vector

from . import childproc                          # no window; one heavy job at a time under a memory ceiling

from .ops_common import (abs_output_path, MifOpError, get_object, simulation_guard, _is_id_struct, check_output_path, check_writable_dir, finite_float, finite_floats, finite_int,
                         reject_unknown, rna_properties, take,
                         take_bool, take_float, take_int, write_rna_property)

_SETTINGS_KEYS = {
    "engine", "resolutionX", "resolutionY", "percentage", "samples",
    "filePath", "output", "fileFormat", "filmTransparent", "colorMode",
    "useDenoising", "exposure", "colorDepth", "eeveeRaytracing", "eevee", "cycles", "render", "ffmpeg",
    # THE OUTPUT FILE'S OWN SETTINGS (2026-10-04): scene.render.image_settings by identifier - exr_codec, the colour
    # management override and its nested linear_colorspace_settings.name / view_settings.view_transform, use_preview
    "image",
    # HOW CYCLES DRAWS HAIR (2026-10-05): scene.cycles_curves - shape RIBBONS | THICK | THICK_LINEAR (5.0+), subdivisions
    "cyclesCurves",
    # THE CYCLES TEXTURE CACHE, 5.2 AND LATER (2026-10-05): GENERATE or CLEAR the .tx files, and measure a preview
    "textureCache", "textureCacheMemory",
}

# THE ENGINES' OWN SETTINGS, BY RNA IDENTIFIER (2026-09-21). This op curated fourteen keys and nothing reached the
# hundred-odd others on scene.eevee and scene.cycles - found building a fire, where EEVEE draws every volume on an
# 8-pixel grid (volumetric_tile_size) and so turned a baked flame into a soft blob, with no typed op able to set a
# finer grid, more volume samples, or Cycles' volume step rate. So `eevee` and `cycles` take {identifier: value},
# written through ops_common.write_rna_property - each value checked against the property's own type - and inside
# this op's snapshot, so a refusal puts the whole call back. render_info {rna: true} is the read twin.
#
# A SETTING THIS OP ALREADY CURATES IS NOT TAKEN TWICE. `samples` writes taa_render_samples on EEVEE and samples on
# Cycles, `eeveeRaytracing` the raytracing switches, `useDenoising` Cycles' denoiser: naming one both ways in one call
# is two answers to one question, refused as set_modifier refuses it.
_ENGINE_CURATED = {
    "eevee": {"taa_render_samples": "samples", "use_raytracing": "eeveeRaytracing", "use_ssr": "eeveeRaytracing",
              "use_ssr_refraction": "eeveeRaytracing"},
    "cycles": {"samples": "samples", "use_denoising": "useDenoising"},
    "render": {"engine": "engine", "resolution_x": "resolutionX", "resolution_y": "resolutionY",
               "resolution_percentage": "percentage", "filepath": "filePath", "film_transparent": "filmTransparent"},
    "ffmpeg": {},
    # the format, depth and mode are this op's own keys: fileFormat picks the media type that accepts the format
    "image": {"file_format": "fileFormat", "color_depth": "colorDepth", "color_mode": "colorMode",
              "media_type": "fileFormat"},
    "cyclesCurves": {},
}
# A VIDEO'S CONTAINER, CODECS AND SOUNDTRACK (2026-09-23): scene.render.ffmpeg, the fourth group - format, codec,
# audio_codec, audio_bitrate, audio_mixrate, audio_channels and every other setting there, by identifier. A video with
# audio_codec NONE is silent whatever the scene plays, which is how a mixdown gets MUXED into a render: this group, then
# save_file, then render_animation. Measured 2026-09-23 on 3.6.23, 4.2.17, 4.4.0, 5.0.1 and 5.2.1: an MPEG4/H264/AAC
# render of a sequencer tone at scene volume 0.5 carries the tone 6.02 dB down, read back out of the file.
# HAIR AS CYCLES RENDERS IT (2026-10-05): scene.cycles_curves is a struct the Cycles add-on registers OUTSIDE
# scene.cycles (CyclesCurveRenderSettings) - shape RIBBONS or THICK, THICK_LINEAR from 5.0 (measured: absent on 4.2.23
# and 4.4.0, present on 5.0.1 and 5.2.2), and subdivisions - so hair rendered as 3D curves instead of ribbons had no
# typed route. The sixth group, the same writer as the others; absent with the add-on off.
_GROUP_PATH = {"eevee": "scene.eevee", "cycles": "scene.cycles", "render": "scene.render",
               "ffmpeg": "scene.render.ffmpeg", "image": "scene.render.image_settings",
               "cyclesCurves": "scene.cycles_curves"}
_GROUPS = ("eevee", "cycles", "render", "ffmpeg", "image", "cyclesCurves")


def _group_holder(sc, group):
    if group == "ffmpeg":
        return sc.render.ffmpeg
    if group == "image":
        return sc.render.image_settings
    if group == "cyclesCurves":
        return getattr(sc, "cycles_curves", None)
    return getattr(sc, group, None)


# A DOTTED IDENTIFIER REACHES A NESTED SETTINGS STRUCT (2026-10-04): EEVEE's ray-tracing resolution lives on
# scene.eevee.ray_tracing_options (4.2 and later; MEASURED absent on 3.6.23, present on 4.4.0 and 5.2.1), and an output
# file's colour management override on image_settings.linear_colorspace_settings / view_settings / display_settings
# (all three builds). A flat identifier map could not write any of them. Each step must be a struct POINTER, never a
# datablock: following an ID would write into whatever it points at, which is another op's business.
def _walk_dotted(holder, dotted, group):
    """(the struct holding the last name, the last name) for 'a.b.c', or a refusal naming the structs on the way."""
    parts = dotted.split(".")
    path = _GROUP_PATH[group]
    for part in parts[:-1]:
        prop = holder.bl_rna.properties.get(part)
        fixed = getattr(prop, "fixed_type", None) if prop is not None else None
        nested = sorted(q.identifier for q in holder.bl_rna.properties
                        if q.type == "POINTER" and q.identifier != "rna_type"
                        and not _is_id_struct(getattr(q, "fixed_type", None)))
        if prop is None or prop.type != "POINTER" or _is_id_struct(fixed):
            raise MifOpError("%s has no settings struct %r on Blender %s, so %r cannot be reached. The nested structs "
                             "it has: %s. NOTHING was changed."
                             % (path, part, bpy.app.version_string, dotted, ", ".join(nested) or "(none)"))
        holder = getattr(holder, part)
        path += "." + part
    return holder, parts[-1], path


def _set_file_format(img, fmt):
    """image_settings.file_format = fmt -> the media_type it took, or None where this build has none.

    FROM 5.0 A FORMAT BELONGS TO A MEDIA TYPE (IMAGE, MULTI_LAYER_IMAGE, VIDEO), and file_format only accepts the
    current type's formats: measured 2026-09-23, FFMPEG is refused on 5.0.1 and 5.2.1 with enum "FFMPEG" not found in
    ('JPEG', 'OPEN_EXR', 'PNG', ...) until media_type is VIDEO - while the static enum still lists it, so the check
    above every write passes and the assignment raised a bare TypeError after the engine and path had moved. So the
    build decides: the format is tried under each media type it has, and the first that accepts it stays. Raises
    TypeError when none does; the caller puts the call's writes back."""
    try:
        img.file_format = fmt
        return getattr(img, "media_type", None)
    except TypeError:
        if not hasattr(img, "media_type"):
            raise
    before = img.media_type
    for item in img.bl_rna.properties["media_type"].enum_items:
        if item.identifier == before:
            continue
        img.media_type = item.identifier
        try:
            img.file_format = fmt
            return item.identifier
        except TypeError:
            continue
    img.media_type = before
    raise TypeError("no media type of this Blender (%s) accepts file_format %r"
                    % (", ".join(i.identifier for i in img.bl_rna.properties["media_type"].enum_items), fmt))


def _video_settings(sc):
    """What a video render will be, or None when the output is not a video container."""
    img = sc.render.image_settings
    if img.file_format not in _CONTAINER_FORMATS:
        return None
    ff = sc.render.ffmpeg
    out = {"container": ff.format, "codec": ff.codec, "audioCodec": ff.audio_codec,
           "audioChannels": ff.audio_channels, "audioSampleRate": ff.audio_mixrate,
           "audioBitrateKbps": ff.audio_bitrate, "carriesAudio": ff.audio_codec != "NONE"}
    if img.file_format != "FFMPEG":
        out["carriesAudio"] = False        # the AVI formats write no audio stream at all
    if not out["carriesAudio"]:
        out["silentNote"] = ("this video carries NO sound, whatever the scene plays - set ffmpeg {audio_codec: AAC} "
                             "(or FLAC, PCM, ...) with fileFormat FFMPEG to mux the scene's audio into it")
    return out
# scene.render's OWN GROUP (2026-09-21, the same day): motion blur - use_motion_blur and motion_blur_shutter, on
# scene.render from 4.2 for EEVEE and Cycles alike - was the next setting a fire's sparks needed and nothing reached.
# Some of its properties belong to another op, which the refusal names rather than letting two ops own one setting.
_RENDER_ELSEWHERE = {"fps": "set_frame_range", "fps_base": "set_frame_range", "frame_map_old": "set_frame_range",
                     "frame_map_new": "set_frame_range", "use_compositing": "set_compositing",
                     "use_sequencer": "set_compositing"}
_RENDER_KEYS = {"filePath", "output", "frame", "samples", "resolutionX", "resolutionY",
                "percentage", "writeStill", "returnImage", "previewMaxPx",
                "previewMaxBytes", "allowSimulation", "computeBackend", "region", "regionPx", "crop"}
_STILL_BACKENDS = ("NONE", "OPTIX", "CUDA", "HIP", "METAL", "ONEAPI")


def _still_backend(sc, params):
    """render_still computeBackend, THE CHECKS: (prefs, backend, devices, was) or None. Writes nothing - the write is
    _still_backend_apply, after every other refusal render_still can make (audit_mutate_then_deny).

    Refuses when the engine is not Cycles, the name is unknown, or the backend lists no device here.
    MEASURED 2026-10-05 headless on 5.2.2 with no GPU: compute_device_type takes CUDA though get_devices_for_type
    ('CUDA') is empty (its enum lists nothing to check against), and the render then runs on the CPU in silence - which
    is why an empty backend is refused rather than written."""
    raw = take(params, "computeBackend", default=None, kind=str)
    if raw is None:
        return None
    backend = str(raw).upper()
    if backend not in _STILL_BACKENDS:
        raise MifOpError("computeBackend must be one of %s, got %r. NOTHING was rendered."
                         % (", ".join(_STILL_BACKENDS), raw))
    if sc.render.engine != "CYCLES":
        raise MifOpError("computeBackend chooses Cycles' device and the scene renders with %s. Set engine CYCLES "
                         "(set_render_settings) first. NOTHING was rendered." % sc.render.engine)
    addon = bpy.context.preferences.addons.get("cycles")
    if addon is None:
        raise MifOpError("the Cycles add-on is not enabled, so there is no compute preference to set. NOTHING was "
                         "rendered.")
    prefs = addon.preferences
    was = str(prefs.compute_device_type)

    def listed(b):
        try:
            return [d for d in prefs.get_devices_for_type(compute_device_type=b) if d.type != "CPU"]
        except Exception:  # noqa: BLE001 - a backend this build or machine cannot load lists nothing
            return []
    devices = []
    if backend != "NONE":
        devices = listed(backend)
        if not devices:
            others = [b for b in _STILL_BACKENDS[1:] if b != backend and listed(b)]
            raise MifOpError("computeBackend %s lists no device on this machine, so Cycles would render on the CPU and "
                             "say nothing. Backends that list one here: %s. The preference is as it was (%s). NOTHING "
                             "was rendered." % (backend, ", ".join(others) or "none - render on the CPU (NONE)", was))
    return prefs, backend, devices, was


def _still_backend_apply(sc, plan):
    """Write what _still_backend checked: the preference, the devices ticked, the scene's device. The reply's block."""
    prefs, backend, _devices, was = plan
    prefs.compute_device_type = backend
    prefs.get_devices()
    used = []
    for d in prefs.devices:
        if backend != "NONE" and d.type == backend:
            d.use = True
            used.append(d.name)
    sc.cycles.device = "GPU" if backend != "NONE" else "CPU"
    return {"backend": backend, "device": sc.cycles.device, "devices": used, "preferenceWas": was,
            "preferenceNow": str(prefs.compute_device_type),
            "note": ("compute_device_type is a USER PREFERENCE, not a scene setting: it is %s now for every Cycles "
                     "render in this Blender session (and kept if the preferences are saved); the scene's device is "
                     "%s." % (backend, sc.cycles.device))}

# The longest edge of a returned preview, when the caller does not say. Not a limit on the RENDER -
# the file on disk is always full size - only on the copy handed back over the socket.
_PREVIEW_MAX_PX_DEFAULT = 128
# A floor, so previewMaxPx:1 cannot ask for something that carries no information at all.
_PREVIEW_MIN_PX = 32
# A transport guard, not a render limit. Base64 expands the PNG by roughly one third and the
# receiving client's result budget may be smaller than the bridge's socket budget. The caller may
# raise this deliberately, but the default stays below the smallest measured client failure.
_PREVIEW_MAX_BYTES_DEFAULT = 30000
_PREVIEW_MIN_BYTES = 1024


def _encode_preview(path, max_px, max_bytes):
    """(dict) describing a base64 PNG of `path`, downscaled to max_px on its longest edge.

    NEVER RAISES. A render that reached disk is a success, and failing to hand back a picture of it
    must not turn that into a failure - the caller still has the file. Anything that goes wrong is
    reported as imageError beside a null image, which is a different thing from silence.

    THE LOADED IMAGE IS REMOVED. bpy.data.images.load puts a datablock in the file; leaving it there
    would be this op quietly changing the scene to answer a question about it, which is exactly what
    the rest of this addon refuses to do.
    """
    out = {"image": None, "imageError": None}
    img = None
    tmp = None
    try:
        img = bpy.data.images.load(path)
        src_w, src_h = int(img.size[0]), int(img.size[1])
        out["renderedWidth"], out["renderedHeight"] = src_w, src_h
        if src_w <= 0 or src_h <= 0:
            out["imageError"] = "the rendered file loaded with a zero dimension (%dx%d)" % (src_w, src_h)
            return out

        longest = max(src_w, src_h)
        if longest > max_px:
            factor = float(max_px) / float(longest)
            new_w = max(1, int(round(src_w * factor)))
            new_h = max(1, int(round(src_h * factor)))
            img.scale(new_w, new_h)
            out["downscaledFrom"] = [src_w, src_h]
        out["imageWidth"], out["imageHeight"] = int(img.size[0]), int(img.size[1])

        # PNG regardless of what the render wrote. The caller asked for something to LOOK at, and a
        # preview in EXR or a raw format is not that.
        tmp = os.path.join(tempfile.gettempdir(),
                           "mif_preview_%d_%d.png" % (os.getpid(), int(time.time() * 1000)))
        img.filepath_raw = tmp
        img.file_format = "PNG"
        img.save()

        preview_bytes = os.path.getsize(tmp)
        estimated_b64_bytes = 4 * ((preview_bytes + 2) // 3)
        out["imageBytes"] = preview_bytes
        out["estimatedBase64Bytes"] = estimated_b64_bytes
        out["imageFormat"] = "png"
        if preview_bytes > max_bytes:
            out["imageOmitted"] = True
            out["imageOmittedReason"] = (
                "the %d-byte preview exceeds previewMaxBytes=%d and would expand to about %d "
                "base64 bytes. Reduce previewMaxPx or use filePath to inspect the full render."
                % (preview_bytes, max_bytes, estimated_b64_bytes)
            )
            return out

        with open(tmp, "rb") as fh:
            raw = fh.read()
        out["image"] = base64.b64encode(raw).decode("ascii")
        out["imageOmitted"] = False
    except Exception as exc:                                        # noqa: BLE001
        out["imageError"] = "%s: %s" % (type(exc).__name__, str(exc)[:160])
    finally:
        if img is not None:
            try:
                bpy.data.images.remove(img)
            except Exception as exc:                                # noqa: BLE001
                # SAID, NOT SWALLOWED. A datablock left behind is a change to the file, and the
                # caller is entitled to know this op left one.
                out["previewDatablockLeaked"] = str(exc)[:120]
        if tmp:
            try:
                os.remove(tmp)
            except OSError:
                pass
    return out


# ---------------------------------------------------------------------------
# compare_to_reference
# ---------------------------------------------------------------------------

# The comparison resolution. Shape is legible here and the pixel loop stays cheap; comparing at
# render resolution is slower and no more informative about a silhouette.
_COMPARE_PX_DEFAULT = 128
_COMPARE_PX_MIN = 16
_COMPARE_PX_MAX = 512


def _load_scaled_luma(path, px):
    """(luma list, alpha list, w, h) for an image scaled to fit px on its longest edge.

    THE DATABLOCK IS REMOVED BY THE CALLER - this returns plain Python lists precisely so the image
    does not have to stay alive, because leaving one behind would be this op quietly adding
    something to the file to answer a question about it.
    """
    img = bpy.data.images.load(path)
    try:
        w, h = int(img.size[0]), int(img.size[1])
        if w <= 0 or h <= 0:
            raise MifOpError("'%s' loaded with a zero dimension (%dx%d)" % (path, w, h))
        longest = max(w, h)
        if longest > px:
            f = float(px) / float(longest)
            img.scale(max(1, int(round(w * f))), max(1, int(round(h * f))))
            w, h = int(img.size[0]), int(img.size[1])
        n = w * h * 4
        buf = [0.0] * n
        # foreach_get is a C-level bulk copy - a Python loop over pixels costs orders of magnitude
        # more, which is the same reason ops_mesh reaches for it on vertex coordinates.
        img.pixels.foreach_get(buf)
        luma = [0.0] * (w * h)
        alpha = [0.0] * (w * h)
        for i in range(w * h):
            r, g, b, a = buf[i * 4], buf[i * 4 + 1], buf[i * 4 + 2], buf[i * 4 + 3]
            luma[i] = 0.2126 * r + 0.7152 * g + 0.0722 * b
            alpha[i] = a
        return luma, alpha, w, h
    finally:
        try:
            bpy.data.images.remove(img)
        except Exception:                                           # noqa: BLE001
            pass


def op_compare_to_reference(params):
    """How close is an image to a reference? Two numbers, because they fail differently.

    params:
      image (str, required)      the render to judge
      reference (str, required)  the picture it is supposed to match
      comparePx (int)            longest edge both are scaled to, default 128
      threshold (float)          luminance above which a pixel counts as subject, default 0.1.
                                 Ignored when the image has real alpha, which is a better mask.
    """
    reject_unknown(params, {"image", "reference", "comparePx", "threshold"},
                   "compare_to_reference")
    img_path = bpy.path.abspath(str(take(params, "image", required=True, kind=str)))
    ref_path = bpy.path.abspath(str(take(params, "reference", required=True, kind=str)))
    for label, p in (("image", img_path), ("reference", ref_path)):
        if not os.path.isfile(p):
            raise MifOpError("%s '%s' does not exist. NOTHING was compared." % (label, p))

    px = finite_int(take_float(params, "comparePx", default=_COMPARE_PX_DEFAULT), "comparePx")
    if px < _COMPARE_PX_MIN or px > _COMPARE_PX_MAX:
        raise MifOpError("comparePx must be between %d and %d - below that a silhouette carries no "
                         "information and above it the comparison is slower for no gain. "
                         "NOTHING was compared." % (_COMPARE_PX_MIN, _COMPARE_PX_MAX))
    thresh = finite_float(take_float(params, "threshold", default=0.1), "threshold")

    a_luma, a_alpha, aw, ah = _load_scaled_luma(img_path, px)
    b_luma, b_alpha, bw, bh = _load_scaled_luma(ref_path, px)

    # DIFFERENT ASPECT RATIOS ARE NOT COMPARED PIXEL-FOR-PIXEL. Two images scaled to the same longest
    # edge can still differ in the other dimension, and lining those up by index would compare a
    # pixel against whatever happened to land at the same offset - a number that looks like a
    # measurement and means nothing.
    if (aw, ah) != (bw, bh):
        raise MifOpError(
            # NO GLOBAL CLAIM ABOUT STATE. This refusal happens AFTER both images were loaded and
            # removed again, so "NOTHING was compared" - which is what this said - was a promise
            # about the file that this function is not in a position to make cleanly.
            # audit_mutate_then_deny caught it: a load is a write, and a write above a refusal that
            # promises nothing happened is the shape it exists to find. What is true is narrower
            # and is what it says now.
            "the two images have different aspect ratios (%dx%d vs %dx%d after scaling), so a "
            "pixel-for-pixel comparison would be meaningless. No comparison was performed. Render "
            "at the reference's aspect ratio first - set_render_settings resolutionX/resolutionY."
            % (aw, ah, bw, bh))

    n = aw * ah
    # Real alpha is a better subject mask than any luminance threshold, so it is preferred when
    # BOTH images have it - a render on a transparent background and a cut-out reference.
    a_has_alpha = any(v < 0.999 for v in a_alpha)
    b_has_alpha = any(v < 0.999 for v in b_alpha)
    use_alpha = a_has_alpha and b_has_alpha

    if use_alpha:
        a_mask = [v > 0.5 for v in a_alpha]
        b_mask = [v > 0.5 for v in b_alpha]
        mask_source = "alpha (both images have transparency, which is a better mask than a threshold)"
    else:
        a_mask = [v > thresh for v in a_luma]
        b_mask = [v > thresh for v in b_luma]
        mask_source = ("luminance > %.3f (one or both images are fully opaque, so alpha could not "
                       "be used)" % thresh)

    inter = union = 0
    total_diff = 0.0
    for i in range(n):
        if a_mask[i] and b_mask[i]:
            inter += 1
        if a_mask[i] or b_mask[i]:
            union += 1
        total_diff += abs(a_luma[i] - b_luma[i])

    iou = (float(inter) / float(union)) if union else None

    # A MASK COVERING ALMOST THE WHOLE FRAME IS NOT A SILHOUETTE. Measured the first time this ran:
    # a cube and the same cube at 0.3x scale both scored IoU 1.0, because the opaque grey world
    # background is brighter than the threshold, so every pixel counted as subject and the
    # intersection over union of two full frames is 1.0 by definition. It would have passed happily
    # against a render compared with itself; it took rendering something genuinely DIFFERENT and
    # still getting 1.0 to see it.
    #
    # So a degenerate mask returns NULL rather than a number. A meaningless 1.0 is worse than an
    # admitted null, because a null cannot be optimised against by mistake.
    a_cover = float(sum(1 for v in a_mask if v)) / float(n) if n else 0.0
    b_cover = float(sum(1 for v in b_mask if v)) / float(n) if n else 0.0
    degenerate = None
    if max(a_cover, b_cover) > 0.95:
        degenerate = ("the subject mask covers %.0f%% of the frame, so it is not separating a "
                      "subject from a background - almost certainly an OPAQUE BACKGROUND brighter "
                      "than the threshold. silhouetteIoU is null rather than a meaningless number. "
                      "FIX: render with a transparent film (set_render_settings filmTransparent, or "
                      "Render Properties > Film > Transparent) and the alpha channel becomes a real "
                      "mask, which this op prefers automatically. Raising `threshold` can also work "
                      "when the subject is much brighter than the background."
                      % (max(a_cover, b_cover) * 100.0))
    elif max(a_cover, b_cover) < 0.01:
        degenerate = ("the subject mask covers under 1%% of the frame - the render may be blank, or "
                      "`threshold` (%.3f) may be above everything in it. silhouetteIoU is null "
                      "rather than a number derived from a handful of pixels." % thresh)
    if degenerate:
        iou = None

    return {
        "image": img_path,
        "reference": ref_path,
        "comparedAt": [aw, ah],
        "silhouetteIoU": round(iou, 4) if iou is not None else None,
        "meanAbsDiff": round(total_diff / float(n), 4) if n else None,
        "maskSource": mask_source,
        "subjectPixels": {"image": sum(1 for v in a_mask if v),
                          "reference": sum(1 for v in b_mask if v)},
        "iouNote": ("silhouetteIoU ignores colour and lighting and asks only whether the subject "
                    "occupies the same space in frame - it is the number for matching a blockout. "
                    "1.0 is identical, and there is no threshold here that means 'good enough': "
                    "that depends on what you are matching."),
        "limitNote": ("this is PIXEL similarity, not semantic similarity. A render matching a "
                      "blockout's silhouette exactly with entirely wrong materials scores well on "
                      "IoU and should. A high score is not 'this looks right'."),
        "maskCoverage": {"image": round(a_cover, 4), "reference": round(b_cover, 4)},
        "degenerateMask": degenerate,
        "emptyMaskNote": (None if union else
                          "NEITHER image has any subject pixels at this threshold, so IoU is "
                          "undefined rather than 0 - raise threshold, or check the render is not "
                          "blank."),
    }

def _engine_ids():
    """The render engines THIS build offers - asked of Blender, never read off the RNA.

    FOUND BY A HAND CHECK, 2026-09-13. This used to return
    bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items: the static read of a DYNAMIC enum.
    On Blender 4.4.0 it listed only BLENDER_EEVEE_NEXT, so BLENDER_WORKBENCH - compiled into every
    build - and CYCLES were both refused as "unknown render engine", and the alias table in
    op_set_render_settings could never resolve two of its three names. No suite saw it: the ops that
    need Cycles set scene.render.engine directly (ops_material bake), so this path was only ever tried
    with EEVEE or with a name meant to be refused. ops_anim's set_keyframe warns about the same read.

    Assigning an identifier that cannot exist makes Blender's own TypeError name every member the
    property offers - "... not found in ('BLENDER_EEVEE_NEXT', 'BLENDER_WORKBENCH', 'CYCLES')". The
    failed assignment changes nothing, and the engine is put back anyway in case a build ever differs.
    If the message stops listing members, the old static read is the fallback, plus the engine already
    selected - never an empty set, which would refuse every engine including the current one.
    """
    render = bpy.context.scene.render
    before = render.engine
    try:
        render.engine = "MIF_NO_SUCH_ENGINE"
    except TypeError as exc:
        listed = str(exc).rsplit("not found in", 1)
        if len(listed) == 2:
            found = set(re.findall(r"'([A-Za-z0-9_]+)'", listed[1]))
            if found:
                return found
    finally:
        if render.engine != before:
            render.engine = before
    return {i.identifier for i in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items} | {before}


def _apply_common(sc, params):
    """Settings shared by set_render_settings and the overrides render_still accepts."""
    applied = {}
    rx = take_float(params, "resolutionX", default=None)
    ry = take_float(params, "resolutionY", default=None)
    pct = take_float(params, "percentage", default=None)
    samples = take_float(params, "samples", default=None)

    # WHERE THE SAMPLE COUNT LIVES IS RESOLVED BEFORE ANYTHING IS WRITTEN. It lives in a different
    # place per engine, and writing the wrong one is a silent no-op: setting cycles.samples on
    # EEVEE changes nothing and reports nothing. That refusal used to fire AFTER resolution,
    # resolutionY and percentage had already been written - and in render_still it also runs before
    # the frame and output-path checks, so a call ending "NOTHING was rendered" had permanently
    # changed the scene's resolution on the way past. Deciding the target first costs nothing:
    # which attribute exists is a property of the engine, not of the value being written.
    samples_target = None
    if samples is not None:
        if "CYCLES" in sc.render.engine and hasattr(sc, "cycles"):
            samples_target = (sc.cycles, "samples", "cycles.samples")
        elif hasattr(sc, "eevee"):
            for attr in ("taa_render_samples", "taa_samples"):
                if hasattr(sc.eevee, attr):
                    samples_target = (sc.eevee, attr, "eevee.%s" % attr)
                    break
        if samples_target is None:
            raise MifOpError("this Blender's %s engine exposes no sample count this op knows how "
                             "to write, so `samples` would have been silently ignored. Set it "
                             "through the engine's own property instead. NOTHING was changed."
                             % sc.render.engine)

    # CONVERTED ABOVE THE COMMIT, all of them, before a single one is written. finite_int CAN
    # refuse - Blender's int properties are 32-bit and a caller can send 2**40 - so converting each
    # one inside its own `if` put a refusal between two writes: set_render_settings{resolutionX:
    # 123, resolutionY: 2**40} moved the width to 123 and then answered "NOTHING was changed".
    # Measured, and introduced by the fix for the raw ValueError these used to throw.
    want_x = finite_int(rx, "resolutionX") if rx is not None else None
    want_y = finite_int(ry, "resolutionY") if ry is not None else None
    want_pct = finite_int(pct, "percentage") if pct is not None else None
    want_samples = finite_int(samples, "samples") if samples_target is not None else None

    # COMMIT. Nothing below can refuse.
    if want_x is not None:
        sc.render.resolution_x = want_x
        applied["resolutionX"] = sc.render.resolution_x
    if want_y is not None:
        sc.render.resolution_y = want_y
        applied["resolutionY"] = sc.render.resolution_y
    if want_pct is not None:
        sc.render.resolution_percentage = want_pct
        applied["percentage"] = sc.render.resolution_percentage
    if samples_target is not None:
        holder, attr, label = samples_target
        setattr(holder, attr, want_samples)
        applied["samples"] = int(getattr(holder, attr))
        applied["samplesOn"] = label
    return applied


def _eevee_raytracing(sc):
    """EEVEE's scene raytracing as this build holds it: eevee.use_raytracing from 4.2, use_ssr AND use_ssr_refraction on
    EEVEE Legacy - see set_render_settings' eeveeRaytracing for the measurement behind both."""
    eevee = getattr(sc, "eevee", None)
    if eevee is None:
        return None
    if hasattr(eevee, "use_raytracing"):
        return bool(eevee.use_raytracing)
    return bool(getattr(eevee, "use_ssr", False) and getattr(eevee, "use_ssr_refraction", False))


def _render_snapshot(sc):
    """Every scene setting set_render_settings can write, as (holder, attr, value), in the order they go back.

    WHY A SNAPSHOT AND NOT JUST ORDERING (2026-09-21). Most of this op's values are checked before its first write,
    but two cannot be: a colorDepth or colorMode is legal or not PER FILE FORMAT, and Blender only says which on
    assignment - with the new format already in place. Their refusals used to say "anything set before it in this
    call stands", so {engine, resolutionX: 324, fileFormat: JPEG, colorMode: RGBA} left the scene on the other engine
    at 324 wide. Now the whole call goes back. ORDER MATTERS: the engine first, and the file format before the depth
    and mode it constrains.
    """
    rs, img = sc.render, sc.render.image_settings
    # media_type BEFORE file_format: from 5.0 a format is only accepted under its own media type (_set_file_format)
    items = [(rs, "engine"), (rs, "resolution_x"), (rs, "resolution_y"), (rs, "resolution_percentage"),
             (rs, "filepath"), (img, "media_type"), (img, "file_format"), (img, "color_depth"), (img, "color_mode"),
             (rs, "film_transparent"), (sc.view_settings, "exposure")]
    cycles = getattr(sc, "cycles", None)
    if cycles is not None:
        items += [(cycles, "samples"), (cycles, "use_denoising")]
    eevee = getattr(sc, "eevee", None)
    if eevee is not None:
        items += [(eevee, a) for a in ("taa_render_samples", "taa_samples", "use_raytracing", "use_ssr",
                                       "use_ssr_refraction")]
    return [(h, a, getattr(h, a)) for h, a in items if hasattr(h, a)]


def _render_restore(snap):
    for holder, attr, value in snap:
        try:
            setattr(holder, attr, value)
        except (TypeError, ValueError, AttributeError):      # noqa: PERF203 - put back what can be
            pass


def _detached(value):
    """A value that stays what it was - an array read off a struct is a LIVE view, and restoring from it restores
    nothing; a flag enum reads as a set."""
    if isinstance(value, (set, frozenset)):
        return set(value)
    if hasattr(value, "__len__") and not isinstance(value, str) and not hasattr(value, "bl_rna"):
        return tuple(value)
    return value


def _engine_settings_wanted(sc, params):
    """set_render_settings' `eevee`, `cycles`, `render` and `ffmpeg`, checked above every write -> [(group, holder,
    key, value)].

    What can be known before writing is refused here: the engine's settings struct is missing (its add-on is off),
    the group is not a non-empty object, a key is not a writable property of it, or a key is a setting this op also
    curates and the call names it both ways (_ENGINE_CURATED). The VALUE is checked by write_rna_property as it is
    written, which is why those writes come last and inside the snapshot."""
    wanted = []
    for group in _GROUPS:
        if params.get(group) is None:
            continue
        want = params[group]
        holder = _group_holder(sc, group)
        if holder is None:
            raise MifOpError("this Blender (%s) has no %s - that engine's add-on is off - so `%s` has nothing to "
                             "write on. NOTHING was changed." % (bpy.app.version_string, _GROUP_PATH[group], group))
        if not isinstance(want, dict) or not want:
            raise MifOpError("`%s` takes a non-empty object of RNA identifier -> value, got %r. NOTHING was changed."
                             % (group, want))
        items = list(want.items())
        if group == "ffmpeg":
            # THE CONTAINER BEFORE THE CODEC: Blender offers the codecs the current container takes, so a codec written
            # ahead of its container can be refused for an order the caller never chose
            items.sort(key=lambda kv: (kv[0] != "format", kv[0] != "codec"))
        if group == "image":
            # THE OVERRIDE SWITCH BEFORE THE NESTED SETTINGS IT GOVERNS
            items.sort(key=lambda kv: (kv[0] != "color_management", "." in str(kv[0])))
        for key, value in items:
            if isinstance(key, str) and "." in key:
                nested, last, path = _walk_dotted(holder, key, group)
                prop = nested.bl_rna.properties.get(last)
                if prop is None or prop.is_readonly or last.startswith("bl_") or last == "rna_type":
                    writable = sorted(q.identifier for q in nested.bl_rna.properties
                                      if not q.is_readonly and not q.identifier.startswith("bl_")
                                      and q.identifier != "rna_type")
                    raise MifOpError("%s has no writable setting %r on Blender %s. It has: %s. NOTHING was changed."
                                     % (path, last, bpy.app.version_string, ", ".join(writable) or "(none)"))
                wanted.append((group, nested, last, value, key, path))
                continue
            owner = _RENDER_ELSEWHERE.get(key) if group == "render" else None
            if owner:
                raise MifOpError("render.%s belongs to %s, which checks it against what else it moves. NOTHING was "
                                 "changed." % (key, owner))
            curated = _ENGINE_CURATED[group].get(key)
            if curated is not None and group == "image":
                # ALWAYS this op's own key: from 5.0 a format is accepted only under its media type, and fileFormat
                # is the path that picks the media type (_set_file_format); a depth or mode is checked per format there
                raise MifOpError("image.%s is `%s` - pass that instead; it is written in the order the format needs. "
                                 "NOTHING was changed." % (key, curated))
            if curated is not None and params.get(curated) is not None:
                raise MifOpError("%s.%s and `%s` in one call write the same setting - two answers to one question. "
                                 "Pass one. NOTHING was changed." % (group, key, curated))
            prop = holder.bl_rna.properties.get(key) if isinstance(key, str) else None
            if prop is None or prop.is_readonly or key.startswith("bl_") or key == "rna_type":
                writable = sorted(p.identifier for p in holder.bl_rna.properties
                                  if not p.is_readonly and not p.identifier.startswith("bl_")
                                  and p.identifier != "rna_type")
                raise MifOpError("%s has no writable setting %r on Blender %s. It has: %s. NOTHING was changed."
                                 % (_GROUP_PATH[group], key, bpy.app.version_string, ", ".join(writable) or "(none)"))
            wanted.append((group, holder, key, value, key, _GROUP_PATH[group]))
    return wanted


# THE CYCLES TEXTURE CACHE (5.2 and later). render.generate_texture_cache writes a tiled, mipmapped .tx file for every
# image the shader nodes use into a blender_tx folder beside the image, and clear_texture_cache deletes them; Cycles
# then loads only the tiles a render touches. Measured 2026-10-05 on 5.2.2: a 2048 px PNG on a plane rendered at 64 px
# with 1 sample peaked at 17 MB with use_texture_cache off and 1 MB with it on after GENERATE. Neither the operators
# nor render.use_texture_cache exist on 5.0.1 or before.
_TX_ACTIONS = ("GENERATE", "CLEAR")


def _texture_cache_plan(sc, params):
    """(action, measure) for textureCache / textureCacheMemory, or None - every refusal before any write."""
    want = take(params, "textureCache", default=None, kind=str)
    mem = take_bool(params, "textureCacheMemory", default=False) if params.get("textureCacheMemory") is not None \
        else False
    if want is None:
        if mem:
            raise MifOpError("textureCacheMemory measures a preview render before and after textureCache GENERATE - "
                             "pass that too. NOTHING was changed.")
        return None
    want = str(want).upper()
    if not hasattr(sc.render, "use_texture_cache") or "generate_texture_cache" not in dir(bpy.ops.render):
        raise MifOpError("the Cycles texture cache (render.generate_texture_cache, render.use_texture_cache) arrived in "
                         "Blender 5.2, and this is %s. NOTHING was changed." % bpy.app.version_string)
    if want not in _TX_ACTIONS:
        raise MifOpError("textureCache is GENERATE or CLEAR, got %r. NOTHING was changed." % want)
    if mem and want != "GENERATE":
        raise MifOpError("textureCacheMemory compares a preview with the cache off and with the cache GENERATE wrote - "
                         "it goes with GENERATE, not %s. NOTHING was changed." % want)
    if mem and sc.camera is None:
        raise MifOpError("textureCacheMemory renders a preview, and there is no scene camera. NOTHING was changed.")
    return want, mem


def _tx_files():
    """Every .tx file in the blender_tx folder beside an image file this file uses."""
    found = set()
    for img in bpy.data.images:
        if img.source not in ("FILE", "SEQUENCE", "TILED") or not img.filepath:
            continue
        folder = os.path.join(os.path.dirname(os.path.abspath(bpy.path.abspath(img.filepath))), "blender_tx")
        if os.path.isdir(folder):
            found.update(os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".tx"))
    return found


def _preview_memory_mb(sc, use_cache):
    """The largest Mem / Peak figure Cycles reports while rendering a 64 px, 1-sample CPU preview of the scene."""
    stats = []

    def grab(*args):
        stats.extend(a for a in args if isinstance(a, str))

    keep = [(sc.render, "engine"), (sc.render, "resolution_x"), (sc.render, "resolution_y"),
            (sc.render, "resolution_percentage"), (sc.render, "use_texture_cache")]
    keep = [(h, a, getattr(h, a)) for h, a in keep]
    keep_c = [(sc.cycles, a, getattr(sc.cycles, a)) for a in ("samples", "device") if hasattr(sc, "cycles")]
    bpy.app.handlers.render_stats.append(grab)
    try:
        sc.render.engine = "CYCLES"
        sc.cycles.samples = 1
        sc.cycles.device = "CPU"
        sc.render.resolution_x = sc.render.resolution_y = 64
        sc.render.resolution_percentage = 100
        sc.render.use_texture_cache = use_cache
        bpy.ops.render.render()
    finally:
        bpy.app.handlers.render_stats.remove(grab)
        for h, a, v in keep + keep_c:
            setattr(h, a, v)
    best = None
    for line in stats:
        for m in re.finditer(r"(?:Mem|Peak):\s*([\d.]+)([KMG])", line):
            v = float(m.group(1)) * {"K": 1.0 / 1024.0, "M": 1.0, "G": 1024.0}[m.group(2)]
            best = v if best is None else max(best, v)
    return best


def _run_texture_cache(sc, plan):
    action, mem = plan
    out = {"action": action}
    before_files = _tx_files()
    if mem:
        out["memoryMB"] = {"cacheOff": _preview_memory_mb(sc, False)}
    if action == "GENERATE":
        bpy.ops.render.generate_texture_cache()
    else:
        bpy.ops.render.clear_texture_cache()
    after_files = _tx_files()
    if mem:
        out["memoryMB"]["cacheOn"] = _preview_memory_mb(sc, True)
        out["memoryNote"] = ("the largest Mem/Peak Cycles reported while rendering a 64 px, 1-sample CPU preview, with "
                             "use_texture_cache off and then on; the render settings were put back, and Render Result "
                             "holds the last preview.")
    out["txFilesWritten"] = sorted(after_files - before_files)[:50]
    out["txFilesRemoved"] = sorted(before_files - after_files)[:50]
    out["txFilesNow"] = len(after_files)
    out["useTextureCache"] = bool(sc.render.use_texture_cache)
    if not sc.render.use_texture_cache:
        out["note"] = ("render.use_texture_cache is off, so a render does not read these files - render "
                       "{use_texture_cache: true}.")
    return out


def op_set_render_settings(params):
    """Configure the render, and report what the scene ACTUALLY holds afterwards."""
    reject_unknown(params, _SETTINGS_KEYS, "set_render_settings")
    sc = bpy.context.scene
    tx_plan = _texture_cache_plan(sc, params)
    before = {"engine": sc.render.engine,
              "resolution": [sc.render.resolution_x, sc.render.resolution_y],
              "percentage": sc.render.resolution_percentage,
              "filePath": sc.render.filepath,
              "fileFormat": sc.render.image_settings.file_format,
              "eeveeRaytracing": _eevee_raytracing(sc)}

    # PARSED AND CHECKED ABOVE EVERY WRITE. Placing this beside the assignment it guards looked
    # right and was not: _apply_common runs first, so an unusable filePath refused with "NOTHING was
    # changed" after the engine and sample count had already moved. audit_mutate_then_deny caught it
    # in the same minute it was written, which is the whole argument for gating that audit at zero.
    out_path = take(params, "filePath", "output", default=None, kind=str)
    resolved_path = None
    if out_path:
        resolved_path = bpy.path.abspath(str(out_path))
        check_output_path(out_path, resolved_path, "changed")

    # COLOUR MODE, CHECKED HERE FOR THE REASON THE BLOCK ABOVE GIVES. It used to be a bare
    # assignment two hundred lines down, so `{percentage: 33, colorMode: "CMYK"}` wrote the
    # percentage, then threw a raw TypeError with no parameter name in it - a call that reported
    # failure and changed the scene. Found by the second seat 2026-09-15 (B1334); measured:
    # percentage 50 before, ok:false, percentage 33 after.
    #
    # ONLY THE FORMAT-INDEPENDENT HALF IS CHECKABLE HERE. BW/RGB/RGBA is the whole enum, and which
    # of them a given file format accepts is decided on assignment - the same thing colorDepth's
    # block says about depths. So a value that is not a colour mode at all costs nothing, and one
    # that is a colour mode this FORMAT refuses is still caught at the write, wrapped rather than
    # raw. Two guards, because they catch two different mistakes.
    cm_want = take(params, "colorMode", default=None, kind=str)
    if cm_want:
        cm_want = str(cm_want).upper()
        cm_valid = {i.identifier for i in
                    bpy.types.ImageFormatSettings.bl_rna.properties["color_mode"].enum_items}
        if cm_want not in cm_valid:
            raise MifOpError("unknown colorMode %r. Valid: %s. NOTHING was changed."
                             % (take(params, "colorMode", default=None, kind=str),
                                ", ".join(sorted(cm_valid))))

    # FILE FORMAT, CHECKED HERE FOR THE SAME REASON AS THE TWO BLOCKS ABOVE. It was validated
    # where it is WRITTEN, which is below _apply_common and below the filepath assignment - so
    # `{engine: "CYCLES", filePath: "...", fileFormat: <invalid>}` left a scene on Cycles rendering
    # to a new path after a call that reported failure. Reported by the second seat as defect M,
    # measured on 4.4.0, and it is the third key in this handler with that shape: filePath's own
    # comment says "Placing this beside the assignment it guards looked right and was not", and
    # colorMode's says the same about B1334 yesterday.
    #
    # ONLY THE CHECK MOVES. The assignment stays below, because colorDepth's rollback reads
    # fmt_before immediately before it and that ordering has already been got wrong once.
    fmt_want = take(params, "fileFormat", default=None, kind=str)
    if fmt_want:
        fmt_valid = {i.identifier for i in
                     bpy.types.ImageFormatSettings.bl_rna.properties["file_format"].enum_items}
        if str(fmt_want).upper() not in fmt_valid:
            raise MifOpError("unknown fileFormat '%s'. Valid: %s. NOTHING was changed."
                             % (fmt_want, ", ".join(sorted(fmt_valid))))

    # EEVEE'S SCENE RAYTRACING, WHICH NOTHING COULD REACH. The second seat (mailbox id 81) found set_material_settings'
    # refraction flags on and a glass ball still refracting only the world: the SCENE switch was never exposed.
    # MEASURED 2026-09-16 - a glass sphere (transmission 1, roughness 0) before a checker wall, material raytraced
    # refraction on, 160x120, scene switch off vs on: 4.2.17 25.139 / 4.4.0 25.262 / 5.0.1 25.262 / 5.2.1 25.275 (0-255
    # mean difference) through eevee.use_raytracing. 3.6.23 has no use_raytracing - EEVEE Legacy refracts through
    # screen-space reflections, and eevee.use_ssr + use_ssr_refraction off vs on measured 29.0 there. So one key, the
    # property this build has, and the response names what was written. Resolved here, above every write.
    rt_want = take_bool(params, "eeveeRaytracing") if params.get("eeveeRaytracing") is not None else None
    rt_attrs = []
    if rt_want is not None:
        if hasattr(sc.eevee, "use_raytracing"):
            rt_attrs = ["use_raytracing"]
        elif hasattr(sc.eevee, "use_ssr") and hasattr(sc.eevee, "use_ssr_refraction"):
            rt_attrs = ["use_ssr", "use_ssr_refraction"]
        else:
            raise MifOpError("this Blender (%s) has neither eevee.use_raytracing nor eevee.use_ssr_refraction, so there "
                             "is no scene switch for eeveeRaytracing to write. NOTHING was changed."
                             % bpy.app.version_string)

    # AND THE LAST FOUR, READ HERE (2026-09-21) - they were read after the engine, resolution, path and format had
    # been written, so {engine, resolutionX: 321, filmTransparent: "flase"} switched the engine and refused.
    want_film = take_bool(params, "filmTransparent", default=False) if "filmTransparent" in params else None
    want_exposure = take_float(params, "exposure", default=None)
    want_denoise = take_bool(params, "useDenoising", default=True) if "useDenoising" in params else None
    depth = take(params, "colorDepth", default=None)
    # finite_int: int(depth) raised a raw OverflowError for Infinity (5.0.1, 2026-09-20).
    depth_s = str(finite_int(depth, "colorDepth")) if depth is not None else None
    engine_writes = _engine_settings_wanted(sc, params)
    snap = _render_snapshot(sc)
    snap += [(holder, key, _detached(getattr(holder, key))) for _group, holder, key, _value, _label, _path
             in engine_writes]

    engine = take(params, "engine", default=None, kind=str)
    if engine:
        want = str(engine).upper()
        valid = _engine_ids()
        if want not in valid:
            # Blender renamed EEVEE's identifier between versions - BLENDER_EEVEE became
            # BLENDER_EEVEE_NEXT and back again - so a friendly alias is resolved against what THIS
            # build actually offers rather than against a remembered name.
            alias = {"EEVEE": [e for e in valid if "EEVEE" in e],
                     "CYCLES": [e for e in valid if "CYCLES" in e],
                     "WORKBENCH": [e for e in valid if "WORKBENCH" in e]}.get(want, [])
            if alias:
                want = sorted(alias)[0]
            else:
                raise MifOpError("unknown render engine '%s' for this Blender. Valid: %s. NOTHING "
                                 "was changed." % (engine, ", ".join(sorted(valid))))
        sc.render.engine = want

    # PUT IT BACK if the rest refuses. _apply_common validates `samples` against whatever
    # engine is now selected - it has to run after the switch - and every refusal it makes promises
    # "NOTHING was changed". Without this, set_render_settings({engine: "CYCLES", samples: ...}) on
    # a build whose Cycles exposes no sample count left the scene on Cycles and said nothing had
    # changed, which is the one thing that sentence is meant to rule out. The whole snapshot since
    # 2026-09-21, not only the engine - the same restore the colour checks below use.
    try:
        applied = _apply_common(sc, params)
    except MifOpError:
        _render_restore(snap)
        raise

    if resolved_path is not None:
        # Validated at the top. This op is where a bad path gets REMEMBERED and render_still is
        # where it fails, so checking only there meant this one answered ok and left the failure to
        # surface later, in a different endpoint, against a caller already told it worked.
        sc.render.filepath = resolved_path
    fmt = take(params, "fileFormat", default=None, kind=str)
    # THE "BEFORE" IS THE SNAPSHOT, taken above every write. The first version of the colorDepth
    # rollback below read the "before" value AFTER this block had already changed it, so it
    # restored JPEG to JPEG and did nothing - caught by a probe that set PNG/16, asked for
    # JPEG/32, and found PNG had not come back. Depth is in the snapshot too, because changing the
    # format CLAMPS it: going to JPEG drops a stored 16 to 8, so putting the format back
    # without the depth restores half the state.
    if fmt:
        # ALREADY VALIDATED AT THE TOP - see the fmt_want block above. The enum check used to live
        # here, which is what made a bad format refuse AFTER the engine and path had moved.
        # From 5.0 the build's media type decides whether the format is accepted - see _set_file_format - and a
        # format no media type takes puts every write of this call back.
        img = sc.render.image_settings
        media_before = getattr(img, "media_type", None)
        try:
            took = _set_file_format(img, str(fmt).upper())
        except TypeError as exc:
            _render_restore(snap)
            raise MifOpError("fileFormat %r cannot be set on this Blender (%s): %s. NOTHING was changed - every "
                             "setting this call wrote was put back." % (fmt, bpy.app.version_string, exc))
        if took is not None and took != media_before:
            applied["mediaType"] = took
    # COLOUR DEPTH, AND IT MUST BE WRITTEN AFTER file_format ABOVE. render_info has always
    # reported colorDepth and nothing could set it - an 8-bit normal map or HDR pass is
    # quantised, exists, and looks roughly right, which is the failure this repo is built
    # around.
    #
    # THE ENUM LIES AND THE SETTER DOES NOT. bl_rna.properties["color_depth"].enum_items
    # reports 8,10,12,16,32 for EVERY format including JPEG - measured on 3.6 and 5.0 - so
    # validating against it would accept "32" on a JPEG. Assigning it raises TypeError
    # naming the REAL set: enum "32" not found in ('8'). So the attempt IS the validation,
    # and Blender's own message is passed through because it knows what this format allows
    # and the introspection does not.
    #
    # Order matters for the same reason: a depth written before the format is checked
    # against the OLD format, so fileFormat:OPEN_EXR with colorDepth:32 in one call would
    # fail against whatever was set before.
    if depth_s is not None:
        # EVERYTHING IS PUT BACK IF THE DEPTH IS REFUSED. The only way to know a depth is legal
        # is to assign it, and that needs the new format already in place - so the refusal can
        # only come after writes. It used to put back the format alone and say "anything set
        # before it in this call stands"; since 2026-09-21 the snapshot taken above every write
        # goes back whole, engine and resolution included, and the call changes nothing.
        try:
            sc.render.image_settings.color_depth = depth_s
        except (TypeError, ValueError) as exc:
            bad_fmt = sc.render.image_settings.file_format
            _render_restore(snap)
            raise MifOpError(
                "colorDepth %r is not one the %s format accepts: %s. The enum's own item "
                "list reports 8, 10, 12, 16 and 32 for EVERY format and is wrong - Blender "
                "validates on assignment instead, so this is its own message. NOTHING was "
                "changed - every setting this call wrote was put back."
                % (depth, bad_fmt, exc))
    cm = take(params, "colorMode", default=None, kind=str)
    if cm:
        # VALIDATED BEFORE THE WRITE, and then the write is wrapped as well - this is the only enum
        # in the handler that had NEITHER, and its two neighbours have one each. Found by the second
        # seat 2026-09-15 (B1334). A bare assignment gave a raw Python traceback that never said
        # `colorMode`:
        #     TypeError: bpy_struct: item.attr = val: enum "CMYK" not found in ('BW', 'RGB', 'RGBA')
        # and - the expensive half - everything written EARLIER in the same call stood, with the
        # refusal saying nothing about it. Measured: {percentage:33, colorMode:"CMYK"} answered
        # ok:false and read back percentage 33. A call that reported failure and changed the scene.
        #
        # BOTH GUARDS, because they catch different things. The RNA enum is the fileFormat shape and
        # catches a value that is not a colour mode at all. But BW/RGB/RGBA are format-DEPENDENT the
        # way colorDepth is - a format with no alpha refuses RGBA on assignment while the enum still
        # lists it - so the attempt is still the real validation, and Blender's own message is the
        # one that knows. Same reasoning as the colorDepth block above, applied to its neighbour.
        try:
            sc.render.image_settings.color_mode = str(cm).upper()
        except (TypeError, ValueError) as exc:
            bad_fmt = sc.render.image_settings.file_format
            _render_restore(snap)
            raise MifOpError(
                "colorMode %r is not one the %s format accepts: %s. The enum's item list is the "
                "same for every format and does not narrow per format - Blender validates on "
                "assignment, so this is its own message. NOTHING was changed - every setting "
                "this call wrote was put back."
                % (cm, bad_fmt, exc))
    if want_film is not None:
        sc.render.film_transparent = want_film
    if want_exposure is not None:
        sc.view_settings.exposure = want_exposure
    if want_denoise is not None:
        if hasattr(sc, "cycles") and "CYCLES" in sc.render.engine:
            sc.cycles.use_denoising = want_denoise
        elif hasattr(sc, "eevee") and hasattr(sc.eevee, "use_taa_reprojection"):
            pass  # EEVEE has no denoiser of the same kind; not silently pretended otherwise
        applied["denoisingRequested"] = want_denoise
    if rt_want is not None:
        for attr in rt_attrs:
            setattr(sc.eevee, attr, bool(rt_want))
        applied["eeveeRaytracingWrote"] = ["eevee.%s" % a for a in rt_attrs]
        if "EEVEE" not in sc.render.engine:
            applied["eeveeRaytracingNote"] = ("written, and this scene renders with %s, which does not read it - it "
                                              "applies if the engine becomes EEVEE." % sc.render.engine)
    if engine_writes:
        # LAST, AND INSIDE THE SNAPSHOT: write_rna_property checks each value against its property's own type as it
        # writes, so a refusal here comes after the writes above - and puts every one of them back.
        written = {}
        try:
            for group, holder, key, value, label, path in engine_writes:
                written.setdefault(group, {})[label] = write_rna_property(
                    holder, key, value, "NOTHING was changed - every setting this call wrote was put back.",
                    path, path, frozenset())
        except Exception:
            _render_restore(snap)
            raise
        applied.update(written)
        idle = [g for g in written if g in ("eevee", "cycles", "cyclesCurves")
                and ("EEVEE" if g == "eevee" else "CYCLES") not in sc.render.engine]
        if idle:
            applied["engineSettingsNote"] = ("written, and this scene renders with %s, which does not read %s - "
                                             "it applies if the engine changes."
                                             % (sc.render.engine, " or ".join(_GROUP_PATH[g] for g in idle)))
        img_set = sc.render.image_settings
        if any(k == "color_management" or "." in k for k in written.get("image", {})) \
                and img_set.color_management != "OVERRIDE":
            applied["imageColorNote"] = ("written, and image_settings.color_management is %s, so the output follows "
                                         "the scene's colour management and does not read these - set image "
                                         "{color_management: 'OVERRIDE'} for them to apply." % img_set.color_management)
        if "ffmpeg" in written and sc.render.image_settings.file_format != "FFMPEG":
            applied["ffmpegNote"] = ("written, and the output is %s, which does not read scene.render.ffmpeg - it "
                                     "applies once fileFormat is FFMPEG." % sc.render.image_settings.file_format)

    if tx_plan is not None:
        applied["textureCache"] = _run_texture_cache(sc, tx_plan)
    return {
        "before": before,
        "after": {"engine": sc.render.engine,
                  "resolution": [sc.render.resolution_x, sc.render.resolution_y],
                  "percentage": sc.render.resolution_percentage,
                  "filePath": sc.render.filepath,
                  "fileFormat": sc.render.image_settings.file_format,
                  "mediaType": getattr(sc.render.image_settings, "media_type", None),
                  "video": _video_settings(sc),
                  "filmTransparent": bool(sc.render.film_transparent),
                  "eeveeRaytracing": _eevee_raytracing(sc)},
        "applied": applied,
        "hasCamera": bpy.context.scene.camera is not None,
        "cameraNote": (None if bpy.context.scene.camera is not None else
                       "there is NO scene camera, so a render would fail - create_camera with "
                       "makeActive, or set one, before rendering."),
    }


# A RENDER REGION (1.3 gap audit, 2026-10-19): a test render of the area that changed - a face, a material, a fix -
# instead of the whole frame. region [minX, minY, maxX, maxY] in 0-1 of the frame, Blender's border convention (origin
# bottom left), or regionPx [x0, y0, x1, y1] in output pixels from the TOP left (an image's convention); crop (default
# true) writes only the region, false the full frame with the rest empty. RenderSettings.use_border / border_* /
# use_crop_to_border (present on 3.6-5.2). render_still sets them for the render and puts the scene's back after;
# render_animation sets them in the child process only (the session's scene is not touched).
def _parse_region(params, sc, tail):
    reg, px = params.get("region"), params.get("regionPx")
    if reg is None and px is None:
        if params.get("crop") is not None:
            raise MifOpError("crop shapes a region render - pass region or regionPx too. %s" % tail)
        return None
    if reg is not None and px is not None:
        raise MifOpError("give region (0-1) or regionPx (pixels), not both. %s" % tail)
    raw = reg if reg is not None else px
    if (not isinstance(raw, (list, tuple)) or len(raw) != 4
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in raw)):
        raise MifOpError("%s is [x0, y0, x1, y1], got %r. %s" % ("region" if reg is not None else "regionPx", raw, tail))
    w = sc.render.resolution_x * sc.render.resolution_percentage / 100.0
    h = sc.render.resolution_y * sc.render.resolution_percentage / 100.0
    box = finite_floats(list(raw), "region" if reg is not None else "regionPx")
    if reg is not None:
        x0, y0, x1, y1 = box
    else:
        x0, x1 = box[0] / w, box[2] / w
        y0, y1 = 1.0 - box[3] / h, 1.0 - box[1] / h
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise MifOpError("the region %r is not a box inside the frame (0-1 after conversion: %.4f %.4f %.4f %.4f). %s"
                         % (list(raw), x0, y0, x1, y1, tail))
    crop = params.get("crop", True)
    if not isinstance(crop, bool):
        raise MifOpError("crop is true or false, got %r. %s" % (crop, tail))
    return {"border": [x0, y0, x1, y1], "crop": crop,
            "pixels": {"x": int(round(x0 * w)), "y": int(round((1.0 - y1) * h)),
                       "width": int(round((x1 - x0) * w)), "height": int(round((y1 - y0) * h)),
                       "frameWidth": int(w), "frameHeight": int(h), "origin": "top left"}}


def _apply_region(sc, region):
    r = sc.render
    saved = (r.use_border, r.border_min_x, r.border_min_y, r.border_max_x, r.border_max_y, r.use_crop_to_border)
    r.use_border = True
    r.border_min_x, r.border_min_y, r.border_max_x, r.border_max_y = region["border"]
    r.use_crop_to_border = region["crop"]
    return saved


def _restore_region(sc, saved):
    r = sc.render
    (r.use_border, r.border_min_x, r.border_min_y, r.border_max_x, r.border_max_y, r.use_crop_to_border) = saved


_REGION_SCRIPT = """import bpy
for sc in bpy.data.scenes:
    r = sc.render
    r.use_border = True
    r.border_min_x, r.border_min_y, r.border_max_x, r.border_max_y = @@BORDER@@
    r.use_crop_to_border = @@CROP@@
"""


def op_render_still(params):
    """Render the current frame to a file, and report whether a file actually appeared.

    BLOCKS Blender for the whole render. Start with small resolution/samples; a heavy frame will
    exceed the addon's main-thread job timeout and read as a hung bridge.

    region [x0, y0, x1, y1] (0-1, origin bottom left) or regionPx (pixels, origin top left) renders only that box;
    crop (default true) writes the box alone. The scene's own border is put back after; the reply's region names the
    box in pixels.
    """
    reject_unknown(params, _RENDER_KEYS, "render_still")
    sc = bpy.context.scene
    if sc.camera is None:
        raise MifOpError("there is no scene camera, so there is nothing to render from. Create one "
                         "with create_camera (makeActive defaults true). NOTHING was rendered.")

    # THE TARGET IS RESOLVED BEFORE ANYTHING IS WRITTEN. This used to assign sc.render.filepath and
    # then check whether a path existed, so the settings _apply_common had already applied and the
    # frame it had already stepped to were left behind by a refusal reading "NOTHING was rendered".
    # True of the render and silent about the scene, which is the distinction the audit draws.
    out_path = take(params, "filePath", "output", default=None, kind=str)
    target = bpy.path.abspath(str(out_path)) if out_path else bpy.path.abspath(sc.render.filepath)
    if not target:
        raise MifOpError("no output path is set - pass filePath, or set one with "
                         "set_render_settings. NOTHING was rendered.")

    check_output_path(out_path, target, "rendered")

    # The directory too: it needs the target and the filesystem, nothing from the scene, and below
    # the settings write it was refusing "NOTHING was rendered" with the render settings and the
    # current frame already moved.
    parent = os.path.dirname(target)
    if parent and not os.path.isdir(parent):
        try:
            os.makedirs(parent, exist_ok=True)
        except OSError as exc:
            raise MifOpError("cannot create the output directory %s: %s. NOTHING was rendered."
                             % (parent, exc))

    # THE PREVIEW'S LIMITS ARE CHECKED BEFORE THE RENDER. They were read after it, so a bad previewMaxPx or
    # previewMaxBytes refused a call whose render had already run and written its file - ok:false for a success
    # (the message even said so, "The render completed, but..."). Both need only the params, so they are refused
    # here, before a setting is applied or a frame is stepped to.
    want_image = take_bool(params, "returnImage", default=False)
    max_px = max_bytes = None
    if want_image:
        max_px = finite_int(take_float(params, "previewMaxPx", default=_PREVIEW_MAX_PX_DEFAULT), "previewMaxPx")
        if max_px < _PREVIEW_MIN_PX:
            raise MifOpError(
                "previewMaxPx must be at least %d - below that the image carries no information worth "
                "returning. NOTHING was rendered." % _PREVIEW_MIN_PX)
        max_bytes = finite_int(take_float(params, "previewMaxBytes", default=_PREVIEW_MAX_BYTES_DEFAULT),
                               "previewMaxBytes")
        if max_bytes < _PREVIEW_MIN_BYTES:
            raise MifOpError("previewMaxBytes must be at least %d. NOTHING was rendered." % _PREVIEW_MIN_BYTES)

    # READ BEFORE THE OVERRIDES ARE APPLIED (2026-09-21). frame and writeStill were read after _apply_common had
    # written the resolution and samples, so {resolutionX: 64, frame: "banana"} left the scene at 64 wide and
    # rendered nothing.
    frame = take_float(params, "frame", default=None)
    frame_i = finite_int(frame, "frame") if frame is not None else None
    write_still = take_bool(params, "writeStill", default=True)
    # A FRAME TO RENDER AT STEPS EVERY LIVE SIMULATION on the way (ops_common.simulation_guard).
    sim_reply = simulation_guard(params, [frame_i] if frame_i is not None else [], "render_still frame",
                                 tail="NOTHING was rendered, and the frame did not move.")
    backend_plan = _still_backend(sc, params)
    region = _parse_region(params, sc, "NOTHING was rendered.")

    applied = _apply_common(sc, params)
    backend_reply = _still_backend_apply(sc, backend_plan) if backend_plan is not None else None
    if frame_i is not None:
        sc.frame_set(frame_i)
    if out_path:
        sc.render.filepath = target

    # WHERE BLENDER SAYS IT WILL WRITE, asked rather than guessed. frame_path() applies the format
    # extension, the frame numbering and any # padding in the path - all three of which make
    # `target + file_extension` wrong, in both directions, for a path ending in a separator or
    # carrying a # run. Kept as the first candidate with the guesses behind it.
    candidates = []
    try:
        fp = sc.render.frame_path(frame=sc.frame_current)
        if fp:
            candidates.append(fp)
    except (AttributeError, TypeError, RuntimeError):
        pass                      # older API or an unusual output path; the guesses still apply
    candidates.append(target)
    ext = sc.render.file_extension or ""
    if ext and not target.lower().endswith(ext.lower()):
        candidates.append(target + ext)
    _seen = set()
    candidates = [c for c in candidates if not (c in _seen or _seen.add(c))]

    # EVERY CANDIDATE IS STAT'D BEFORE THE RENDER, not just `target`. The previous version captured
    # before_size for `target` alone and then tested `cand != target or st != before_size` - and
    # for the usual winning candidate, target + ext, `cand != target` is TRUE, so the freshness
    # test was satisfied by the candidate's NAME rather than by anything about the file. A stale
    # render from a previous run therefore reported wroteFile:true with its old byte count, which
    # is the one thing this measurement exists to rule out.
    before = {}
    for c in candidates:
        try:
            if os.path.isfile(c):
                before[c] = (os.path.getsize(c), os.path.getmtime(c))
        except OSError:
            pass

    region_saved = _apply_region(sc, region) if region is not None else None
    t0 = time.time()
    try:
        bpy.ops.render.render(write_still=write_still)
    finally:
        if region_saved is not None:
            _restore_region(sc, region_saved)
    elapsed = time.time() - t0

    # THE MEASUREMENT. render() returns FINISHED whether or not anything reached disk, so the file
    # is stat'd rather than trusted - and compared against what was there before.
    wrote, size, stale = None, -1, None
    for cand in candidates:
        if not os.path.isfile(cand):
            continue
        try:
            st, mt = os.path.getsize(cand), os.path.getmtime(cand)
        except OSError:
            continue
        was = before.get(cand)
        # Fresh means: it did not exist, or it changed. mtime is the primary signal because a
        # re-render of the same scene legitimately produces the same byte count.
        fresh = was is None or mt > was[1] or st != was[0]
        if st > 0 and fresh:
            wrote, size, stale = cand, st, False
            break
        if wrote is None:
            wrote, size, stale = cand, st, True

    # THE PICTURE, if it was asked for. Only ever from a file this call is satisfied it WROTE -
    # `wrote and stale is False` is the same predicate wroteFile reports below, not a looser one.
    # Encoding a stale file would hand back an image of somebody else's render and call it this one.
    result_preview = {}
    # want_image, max_px and max_bytes were read and refused BEFORE the render - see the note there. Reading
    # returnImage again here would find it already taken and silently hand back no preview.
    if want_image:
        if wrote and size > 0 and stale is False:
            result_preview = _encode_preview(wrote, max_px, max_bytes)
        else:
            # ASKED FOR AND NOT DELIVERABLE, said plainly. Silence here would read as "the render
            # produced no image", when what happened is that nothing fresh reached disk to encode.
            result_preview = {
                "image": None,
                "imageError": ("no freshly written file to encode - see wroteFile and "
                               "staleFileFound below. The render, not the preview, is what failed."),
            }

    # THE SAMPLES THIS RENDER USED, whether or not the call set them. `applied` carried the count only when `samples` was
    # passed, so a render at the scene's own setting said nothing about it (second seat, report 87 item 7).
    used_samples, used_samples_on = _samples_in_use(sc)
    out = {
        "rendered": True,
        "frame": sc.frame_current,
        "engine": sc.render.engine,
        "samples": used_samples,
        "samplesOn": used_samples_on,
        "resolution": [int(sc.render.resolution_x * sc.render.resolution_percentage / 100.0),
                       int(sc.render.resolution_y * sc.render.resolution_percentage / 100.0)],
        "camera": sc.camera.name,
        "elapsedSeconds": round(elapsed, 3),
        "filePath": wrote or target,
        # FRESH, not merely present. `stale is False` means the file changed across the render;
        # a file that was already there and did not change is reported as NOT written, with
        # staleFileFound naming it, because "there is a png at that path" and "this call rendered
        # one" are different claims and only the second is what a caller asked about.
        "wroteFile": bool(wrote) and size > 0 and stale is False,
        "fileBytes": size,
        "staleFileFound": bool(wrote) and stale is True,
        "applied": applied,
        "sizeNote": ("wroteFile and fileBytes are stat'd off disk. render() returns FINISHED even "
                     "when nothing was written - a bad path or a disabled format both look like "
                     "success from the operator alone."),
        "blockingNote": ("this held Blender's main thread for %.1fs; the bridge answered nothing "
                         "during it." % elapsed),
    }
    # MERGED LAST so the preview cannot quietly overwrite a field the render measured. The keys are
    # disjoint by construction; this ordering is what keeps that true if either side gains one.
    for k, v in result_preview.items():
        out.setdefault(k, v)
    out.update(sim_reply)
    if region is not None:
        out["region"] = dict(region["pixels"], crop=region["crop"], border=[round(v, 6) for v in region["border"]],
                             sceneBorderRestored=True)
    if backend_reply is not None:
        out["computeBackend"] = backend_reply
    if result_preview:
        out["previewNote"] = (
            "image is a compact base64 PNG preview, not the artifact. It is downscaled to "
            "previewMaxPx and omitted before base64 encoding when its PNG exceeds previewMaxBytes. "
            "The full-size file remains at filePath; renderedWidth/renderedHeight describe it and "
            "imageWidth/imageHeight describe the preview attempt.")
    return out


def _samples_in_use(sc):
    """(samples, where) for the engine this scene renders with - one reader for render_info and render_still.

    Samples live in a different place per engine and this is the read half of the same problem _apply_common solves
    for writing, so it reports WHERE it found the number rather than just the number - a caller comparing against what
    they set needs to know they are the same field. WORKBENCH reads neither Cycles' nor EEVEE's count: its anti-aliasing
    is display.render_aa, a sample count or OFF / FXAA, and was reported here as EEVEE's number until 2026-09-16."""
    eng = sc.render.engine
    if "CYCLES" in eng and hasattr(sc, "cycles"):
        return int(sc.cycles.samples), "cycles.samples"
    if "WORKBENCH" in eng:
        aa = getattr(getattr(sc, "display", None), "render_aa", None)
        return (int(aa) if isinstance(aa, str) and aa.isdigit() else None), "display.render_aa"
    if hasattr(sc, "eevee"):
        for attr in ("taa_render_samples", "taa_samples"):
            if hasattr(sc.eevee, attr):
                return int(getattr(sc.eevee, attr)), "eevee.%s" % attr
    return None, None


def op_render_info(params):
    """Everything that decides what a render will look like, including why it will be black.

    THE READ HALF THAT DID NOT EXIST. set_render_settings reports only the five fields it can
    WRITE, and ops_render had no read op at all - one of six modules in that state. So "why is this
    render black / washed out / the wrong size" was unanswerable without run_python, which is the
    arbitrary-code switch a user may have turned off.

    ARRANGED AROUND THE BLACK-RENDER QUESTION, because that is what people actually ask. The four
    usual causes each get a direct field rather than being derivable from one: no scene camera
    (render_still refuses, but set_render_settings and everything else will not tell you), no
    world (a scene with no world contributes NO ambient light, so an interior is pure black outside
    its own fixtures and the lights get blamed), every light hidden from the render, and a view
    transform that is not what the caller assumed.

    rna (bool)   adds `eevee`, `cycles`, `render` and `ffmpeg`: every readable setting of scene.eevee,
                 scene.cycles, scene.render and scene.render.ffmpeg by identifier - the read twin of
                 set_render_settings' groups (2026-09-21; ffmpeg 2026-09-23). null for an engine whose add-on is off.

    `video` says what a video render will be - container, codecs, and whether it carries the scene's sound at all -
    and is null when the output is a still format.
    """
    reject_unknown(params, ("rna",), "render_info")
    want_rna = take_bool(params, "rna", default=False)
    sc = bpy.context.scene
    r = sc.render
    eng = r.engine

    samples, samples_on = _samples_in_use(sc)

    lights = [o for o in sc.objects if o.type == "LIGHT"]
    lit = [o for o in lights if not o.hide_render and getattr(o.data, "energy", 0) > 0]
    vs = getattr(sc, "view_settings", None)
    ds = getattr(sc, "display_settings", None)

    out = {
        "engine": eng,
        "samples": samples,
        "samplesOn": samples_on,
        "resolution": [r.resolution_x, r.resolution_y],
        "percentage": r.resolution_percentage,
        # What the file will ACTUALLY be, which is the number that matters and the one people get
        # wrong - percentage is a multiplier nobody remembers is set.
        "effectiveResolution": [int(r.resolution_x * r.resolution_percentage / 100.0),
                                int(r.resolution_y * r.resolution_percentage / 100.0)],
        "filePath": r.filepath,
        "fileFormat": r.image_settings.file_format,
        "mediaType": getattr(r.image_settings, "media_type", None),
        "video": _video_settings(sc),
        "colorMode": r.image_settings.color_mode,
        "colorDepth": r.image_settings.color_depth,
        "filmTransparent": bool(r.film_transparent),
        "fps": r.fps,
        "fpsBase": round(float(r.fps_base), 6),
        "frameStart": sc.frame_start,
        "frameEnd": sc.frame_end,
        "frameCurrent": sc.frame_current,
        # COLOUR MANAGEMENT, which silently changes every pixel and is the usual cause of "washed
        # out". Reported rather than assumed - the default view transform differs by Blender
        # version and a studio OCIO config renames all of these.
        "viewTransform": getattr(vs, "view_transform", None),
        "look": getattr(vs, "look", None),
        "exposure": round(float(getattr(vs, "exposure", 0.0)), 6),
        "gamma": round(float(getattr(vs, "gamma", 1.0)), 6),
        "displayDevice": getattr(ds, "display_device", None),
        "sceneCamera": sc.camera.name if sc.camera else None,
        "worldName": sc.world.name if sc.world else None,
        "lightCount": len(lights),
        "lightsContributing": len(lit),
        "viewLayer": bpy.context.view_layer.name,
    }

    # THE DIAGNOSIS, not just the census. Each entry is a measured reason this render will produce
    # nothing useful, in the order they bite. A caller gets the answer rather than the inputs to it.
    blockers = []
    if sc.camera is None:
        blockers.append("no scene camera - render_still will refuse and nothing can be framed")
    if sc.world is None:
        blockers.append("no world datablock - the scene contributes NO ambient light, so anything "
                        "not lit by its own fixture renders pure black")
    if not lit:
        blockers.append("no light contributes to the render - %d light(s) exist but all are "
                        "hidden from render or at zero energy" % len(lights))
    if out["effectiveResolution"][0] < 1 or out["effectiveResolution"][1] < 1:
        blockers.append("effective resolution is %s - percentage is %d%%"
                        % (out["effectiveResolution"], r.resolution_percentage))
    out["blockers"] = blockers
    out["wouldRenderSomething"] = not blockers
    # WHAT render_animation WOULD NOT ANIMATE, from the scene's own start frame - see _out_of_process_sim_rows. Not a
    # blocker: render_still renders in this session, where these simulations do step.
    sims = _out_of_process_sim_rows(sc, sc.frame_start)
    out["outOfProcessSimulations"] = sims or None
    out["outOfProcessNote"] = ("render_animation renders the SAVED file in a second Blender, and there %d simulation(s) "
                               "will not animate: %s. bake_physics, then save_file, then render_animation."
                               % (len(sims), ", ".join("%s (%s)" % (s["cache"], s["renders"]) for s in sims))
                               if sims else None)
    if want_rna:
        for group in _GROUPS:
            holder = _group_holder(sc, group)
            out[group] = rna_properties(holder) if holder is not None else None
    return out


# THE `scene` KEY IS BACK, and properly this time. It was removed on 2026-09-03 rather than shipped
# half-working: frame_path() was still being computed on bpy.context.scene while the child rendered
# another one, so every path in the response and every mtime in the postcondition would have
# described the wrong scene - confidently. Now the NAMED scene is resolved first and everything
# downstream reads it: the camera check, the frame range, the output paths, the writable-directory
# probe and the before-mtimes.
#
# AND THE NAME IS VALIDATED HERE BECAUSE BLENDER WILL NOT. Measured on 5.0.1: `blender -b file.blend
# -S BadName` SILENTLY renders the saved active scene instead - no error, no warning, exit 0. So a
# typo would render the wrong scene and report success against paths taken from the right one,
# which is the exact failure the missing output override exists to prevent.
_ANIM_KEYS = {"frameStart", "frameEnd", "start", "end", "frameStep", "step", "scene", "gpu", "gpuBackend", "memoryLimitGB",
              "region", "regionPx", "crop"}
_GPU_BACKENDS = ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI")

# THE GPU, ASKED FOR BY NAME (2026-09-13). The spawned Blender starts with the user's preferences, where Cycles
# renders on the CPU unless a compute device was ever ticked - and on this machine none was: every Cycles clip so
# far rendered on the CPU (lightbox: 660 frames at 1080p / 8 samples in 5,194 s; the bunker preview ~16 s a frame
# at 960x540). This script runs in the spawned process before the frames: it tries the backends in order, keeps the
# first that lists a non-CPU device, ticks every device and sets the scene to GPU; with none it leaves the CPU, so
# gpu:true is safe to send from any machine. EEVEE ignores all of it.
_GPU_SCRIPT = """import bpy
p = bpy.context.preferences.addons['cycles'].preferences
chosen = None
for backend in @@BACKENDS@@:
    try:
        p.compute_device_type = backend
        p.get_devices()
        if any(d.type != 'CPU' for d in p.devices):
            chosen = backend
            break
    except Exception:
        continue
if chosen:
    for d in p.devices:
        d.use = True
    tuned = set()
    for sc in bpy.data.scenes:
        sc.cycles.device = 'GPU'
        # THE DENOISER TOO (2026-09-13 23:05, measured on an RTX 3080 at 1080p / 8 samples): OpenImageDenoise ran
        # on the CPU and cost about seven seconds a frame on top of a two-second render; the OptiX denoiser is
        # part of the same GPU pass. Only where OptiX is the backend - the others keep what the scene had.
        if chosen == 'OPTIX' and sc.cycles.use_denoising:
            sc.cycles.denoiser = 'OPTIX'
        # PERSISTENT DATA, AND OpenImageDenoise ON THE GPU FOR THE OTHER BACKENDS (2026-09-17, measured through
        # render_animation on an RTX 4080 SUPER, 5.2.1, CUDA, 960x540 / 48 samples / denoising, 12 frames): 18.5 s as
        # it was, 14.3 s with persistent data - frames byte-identical - and 4.1 s with the denoiser on the GPU as well,
        # 0.034 of 255 from the CPU denoiser's frames where adjacent frames differ 0.27-1.40. The second seat measured
        # the pair on an RTX 3080 at 1080p / 48: 12-13.5 s a frame to 4.6-6.2 s (mailbox id 169). denoising_use_gpu
        # does not exist on every build, hence the hasattr.
        sc.render.use_persistent_data = True
        tuned.add('persistent data')
        if chosen != 'OPTIX' and sc.cycles.use_denoising and hasattr(sc.cycles, 'denoising_use_gpu'):
            sc.cycles.denoising_use_gpu = True
            tuned.add('denoiser on the GPU')
    print('[MifBlender] render on the GPU through %s%s%s: %s' % (
        chosen, ', OptiX denoiser' if chosen == 'OPTIX' else '', ''.join(', ' + t for t in sorted(tuned)),
        ', '.join(d.name for d in p.devices if d.type != 'CPU')))
else:
    print('[MifBlender] no GPU backend of %s lists a device here - rendering on the CPU' % (@@BACKENDS@@,))
"""


# THE OPTIX KERNEL CAN FAIL WHERE OPTIX LISTS A DEVICE, and the GPU script cannot see it coming. Reported by the second
# seat (mailbox id 152), measured on its machine: Blender 5.2.2, an RTX 3080 on NVIDIA 560.94 - OptiX lists the card, so
# the script above chooses it, and the render then logs "Failed to load OptiX kernel ... OPTIX_ERROR_INTERNAL_COMPILER_ERROR"
# and "New backend is missing implementation for PTX intrinsic optix.ptx.copysign.f32" and renders 0 of 100 frames, where
# CUDA on the same card renders. The order is not changed - OptiX's denoiser saves about seven seconds a frame on this
# machine (see _GPU_SCRIPT) - so render_animation takes gpuBackend, and render_status names it when the log shows this.
_OPTIX_FAILURE_MARKERS = ("Failed to load OptiX kernel", "OPTIX_ERROR_")


def _gpu_log_note(log_text):
    """A sentence when a render log shows the OptiX kernel failing to load, else None."""
    if not any(m in (log_text or "") for m in _OPTIX_FAILURE_MARKERS):
        return None
    return ("the log shows the OptiX kernel failing to load, which renders no frames on this GPU driver even though "
            "OptiX lists the card. Start the render again with gpuBackend:'CUDA' (render_animation), which uses the same "
            "card without OptiX.")
_STATUS_KEYS = {"jobId", "job", "id", "logLines"}

# THE JOB TABLE. A render outlives the request that started it, which is the whole point of this
# pair, so the addon has to remember it somewhere. Module-level and therefore PROCESS-LOCAL: it does
# NOT survive a Blender restart, and render_status says "unknown job" rather than "not finished" for
# an id it has never seen. Those are different answers and conflating them is how a caller waits
# forever for something nobody is doing.
_JOBS = {}
_JOB_SEQ = [0]

# Formats that write ONE file for the whole range instead of a file per frame. For these the frame
# count inside the container is not visible from the filesystem, and any per-frame progress number
# would be invented. Named explicitly rather than guessed from the extension.
_CONTAINER_FORMATS = {"FFMPEG", "AVI_JPEG", "AVI_RAW"}


def _log_tail(path, lines):
    """Last `lines` lines of the job log, read defensively - the log may not exist yet."""
    try:
        with open(path, "rb") as fh:
            try:
                fh.seek(-65536, os.SEEK_END)
            except OSError:
                fh.seek(0)                       # log shorter than the window
            text = fh.read().decode("utf-8", "replace")
        return text.splitlines()[-lines:]
    except Exception:                            # noqa: BLE001
        return []


# UNBAKED SIMULATIONS, OUT OF PROCESS. MEASURED 2026-09-16 on 3.6.23, 4.2.17, 4.4.0, 5.0.1 and 5.2.1 (docs/22 B61): a saved
# file rendered by a second Blender with this op's own argv shape (-b file -s 1 -e 40 -j 1 -a), Workbench 64 px, frame 1
# against frame 40. An unbaked rigid body and an unbaked soft body read 0.0 - they do not move - where the same scenes
# baked moved 0.076-0.085. An unbaked cloth matched its bake when the render started at its cache start; rendered from
# frame 20, its frame 40 differed from the full run's by 0.0043-0.0047. Unbaked emitter particles matched their bake
# from frame 1 and from frame 20, so they are not named. A dynamic paint surface renders flat unless it is baked to disk
# beside a saved file (B54). None of this shows in the live session: render_still and every in-session guard step the
# simulation there, which is how the second seat's squash passed its guards and shipped frame 18 == frame 60.
def _out_of_process_sim_rows(sc, start):
    """The simulations a render of the saved file will not animate, as rows - an empty list when there are none."""
    from . import ops_physics                                   # runtime: the two modules stay independent at import
    rows = []
    rbw = sc.rigidbody_world
    for label, holder, kind in ops_physics._scene_cache_ranges(sc):
        if kind == "dynamic_paint":
            pc = getattr(holder, "point_cache", None)
            if not (pc is not None and pc.is_baked and getattr(pc, "use_disk_cache", False)):
                rows.append({"cache": label, "renders": "flat",
                             "why": "a dynamic paint surface renders only from a disk-cache bake beside a saved .blend"})
            continue
        if bool(getattr(holder, "is_baked", False)):
            continue
        if kind == "rigidbody":
            moving = (rbw is not None and rbw.enabled and rbw.collection is not None
                      and any(o.rigid_body is not None and o.rigid_body.type == "ACTIVE" and o.rigid_body.enabled
                              for o in rbw.collection.objects))
            if moving:
                rows.append({"cache": label, "renders": "frozen",
                             "why": "an unbaked rigid body world does not simulate in a render of the saved file"})
        elif kind == "modifier":
            mtype = label.rsplit(":", 1)[-1]
            if mtype == "soft_body":
                rows.append({"cache": label, "renders": "frozen",
                             "why": "an unbaked soft body does not simulate in a render of the saved file"})
            elif mtype == "cloth" and start > int(holder.frame_start):
                rows.append({"cache": label, "renders": "from rest",
                             "why": "an unbaked cloth simulates only from its cache start (frame %d), and this render "
                                    "starts at frame %d" % (int(holder.frame_start), start)})
    return rows


def op_render_animation(params):
    """Render a frame RANGE out of process, and return immediately with a job to poll.

    WHY THIS CANNOT BE AN IN-PROCESS OP, which is the whole reason it looks like this. Every op in
    this addon runs on Blender's main thread, and server.DEFAULT_JOB_TIMEOUT is 150s. An animation
    render therefore freezes the addon completely - the socket stops answering, bl_status included -
    and the MCP gives up while the render CARRIES ON. That is the worst possible failure: a caller
    told nothing happened, and a machine pinned for ten minutes doing exactly what they asked. The
    timeout ladder elsewhere is deliberately Blender-gives-up-first so a failure has a real error
    rather than an abandoned job, and a long render inverts it.

    So this spawns `blender -b <saved file> -a` as a separate process and hands back a jobId.

    IT RENDERS THE FILE ON DISK, NOT THIS SESSION, and every refusal here follows from that:

      * bpy.data.filepath must be set. There is nothing to render out of process otherwise.
      * bpy.data.is_dirty must be False. A dirty session means the file on disk is NOT what you are
        looking at, so the render would silently produce the wrong scene - lit differently, framed
        differently, at a different resolution. Silently wrong is worse than refused.
      * THERE IS NO OUTPUT OVERRIDE, deliberately. The obvious `-o` flag would desynchronise the
        paths this op computes with scene.render.frame_path() from the paths the child actually
        writes, and the per-frame postcondition below is the only thing that makes a progress
        number real. Set the output with set_render_settings, save_file, then render here. Frame
        RANGE is safe to override and is accepted, because -s/-e change no path.

    THE POSTCONDITION IS PER-FRAME AND IS TAKEN BEFORE THE RENDER STARTS. Every expected frame path
    is enumerated and stat'd FIRST, recording what was already there. A frame counts as rendered
    only if its mtime is at or after the job start - a leftover file from a previous run passes any
    existence check, which is how a progress number reports 100% before a single pixel is drawn.

    FOR A CONTAINER FORMAT (FFMPEG, AVI) there is exactly ONE output path and the number of frames
    inside it is NOT knowable from the filesystem. render_status says so with framesVerifiable:false
    rather than reporting a count it never checked.

    params:
      frameStart / start (int)   default scene.frame_start
      frameEnd / end (int)       default scene.frame_end
      frameStep / step (int)     default scene.frame_step
      scene (str)                render a DIFFERENT scene in the same file. Default the active one.
                                 Validated here, because Blender's -S silently falls back to the
                                 saved active scene on a name it does not know.
      gpu (bool)                 render Cycles on the GPU, trying OPTIX, CUDA, HIP, METAL, ONEAPI in turn
      gpuBackend (str)           with gpu:true, only this backend - CUDA where the OptiX kernel fails to load
      memoryLimitGB (float)      end the render's process tree past this much memory (private commit on Windows,
                                 resident size elsewhere). Default MIF_CHILD_JOB_MEMORY_GB, else half of physical
                                 memory; 0 for none. render_status reports the peak and a kill.

    ONE HEAVY JOB AT A TIME (1.3, from use): parallel background renders and analysis jobs beside a working editor used
    up a 64 GB machine and crashed it. The render takes one of MIF_MAX_CHILD_JOBS (default 1) machine-wide slots - shared
    with the MCP server's builds, Insights exports and encodes (childproc.py) - and is refused, naming what holds them,
    when none is free. It starts with no console window (CREATE_NO_WINDOW on Windows).
    """
    reject_unknown(params, _ANIM_KEYS, "render_animation")
    mem_gb = params.get("memoryLimitGB")
    if mem_gb is not None and (isinstance(mem_gb, bool) or not isinstance(mem_gb, (int, float))
                               or not math.isfinite(mem_gb) or mem_gb < 0):
        raise MifOpError("memoryLimitGB is the render's memory ceiling in GB, 0 or more (0: none), not %r. NOTHING was "
                         "started." % (mem_gb,))
    # CHECKED FIRST, with the other shape errors, before the session is looked at.
    gpu_backend = take(params, "gpuBackend", default=None, kind=str)
    if gpu_backend is not None:
        gpu_backend = str(gpu_backend).upper()
        if gpu_backend not in _GPU_BACKENDS:
            raise MifOpError("gpuBackend must be one of %s, got '%s'. NOTHING was started."
                             % (", ".join(_GPU_BACKENDS), gpu_backend))
        if params.get("gpu") not in (True, 1, "true", "True", "1"):
            raise MifOpError("gpuBackend chooses the backend for gpu:true, and gpu is not true here - pass gpu:true with "
                             "it. NOTHING was started.")
    scene_name = take(params, "scene", kind=str)
    if scene_name:
        sc = bpy.data.scenes.get(str(scene_name))
        if sc is None:
            known = sorted(s.name for s in bpy.data.scenes)
            raise MifOpError("no scene named '%s' in this file. Present: %s. Blender's own -S flag "
                             "would SILENTLY render the saved active scene instead of failing, so "
                             "this is checked here. NOTHING was started."
                             % (scene_name, ", ".join(known)))
    else:
        sc = bpy.context.scene
    r = sc.render

    blend = bpy.data.filepath
    if not blend:
        raise MifOpError("this session has never been saved, so there is no file to render out of "
                         "process. Save it with save_file first (repointSession:true, or pass the "
                         "path here after saving). NOTHING was started.")
    # THE DIRTY GUARD CANNOT DO ITS JOB IN BACKGROUND MODE, so it does not pretend to.
    #
    # bpy.data.is_dirty is ALWAYS True under `blender -b` - measured on 4.4.0 and 5.0.1, where it
    # reads True before a save, immediately AFTER save_as_mainfile, after an edit, and after a
    # second save_mainfile. It never clears. So refusing on it in background mode refuses ALWAYS,
    # and this op - whose whole purpose is spawning a background render - was unusable from a
    # headless bridge, which is a legitimate way to run this addon.
    #
    # In a GUI session the flag works and the guard is worth having, because rendering a stale file
    # silently produces the wrong frames. So it is enforced there and DOWNGRADED TO A WARNING here,
    # with the risk stated rather than an unverifiable claim of safety. Suppressing the refusal
    # while saying nothing would be the worse of the three options.
    dirty_checkable = not bpy.app.background
    if dirty_checkable and bpy.data.is_dirty:
        raise MifOpError("the session has unsaved changes, so the file on disk is NOT the scene you "
                         "are looking at - rendering it would silently produce the wrong frames. "
                         "Save first with save_file (repointSession:true). NOTHING was started.")
    if not os.path.isfile(blend):
        raise MifOpError("bpy.data.filepath is '%s' but no file is there - it was moved or deleted "
                         "since the last save. NOTHING was started." % blend)
    if sc.camera is None:
        raise MifOpError("scene '%s' has no camera, so every frame would fail. Set one with "
                         "set_camera. NOTHING was started." % sc.name)

    start = take_int(params, "frameStart", "start", default=sc.frame_start)
    end = take_int(params, "frameEnd", "end", default=sc.frame_end)
    step = take_int(params, "frameStep", "step", default=sc.frame_step)
    if step < 1:
        raise MifOpError("frameStep must be at least 1, got %d. NOTHING was started." % step)
    if end < start:
        raise MifOpError("frameEnd %d is before frameStart %d - that range renders nothing. "
                         "NOTHING was started." % (end, start))

    # THE OUTPUT DIRECTORY IS CHECKED BY WRITING TO IT, not by asking os.access. On Windows access()
    # reports the DACL and not the effective permission, so it says yes for directories a write then
    # fails on. Minutes of render are the thing being protected here; one probe file is cheap.
    frames = list(range(start, end + 1, step))
    container = r.image_settings.file_format in _CONTAINER_FORMATS
    if container:
        paths = [bpy.path.abspath(r.frame_path(frame=start))]
    else:
        paths = [bpy.path.abspath(r.frame_path(frame=f)) for f in frames]
    # check_writable_dir, shared with export_scene - see it for why the probe's name is no longer fixed.
    outdir = os.path.dirname(paths[0])
    try:
        check_writable_dir(outdir)
    except Exception as exc:                     # noqa: BLE001
        raise MifOpError("the output directory '%s' cannot be written (%s). The render would run "
                         "for minutes and produce nothing. NOTHING was started." % (outdir, exc))

    # WHAT WAS ALREADY THERE, recorded before a single frame is drawn. Without this a progress
    # number counts files somebody else made.
    before = {}
    for p in paths:
        try:
            before[p] = os.path.getmtime(p)
        except OSError:
            before[p] = None

    exe = bpy.app.binary_path
    if not exe or not os.path.isfile(exe):
        raise MifOpError("bpy.app.binary_path is '%s', which is not a file - this Blender cannot "
                         "spawn a copy of itself to render with. NOTHING was started." % exe)

    _JOB_SEQ[0] += 1
    job_id = "ranim%d_%d" % (os.getpid(), _JOB_SEQ[0])
    log_path = os.path.join(outdir, "%s.log" % job_id)
    # -S BEFORE -a, and only when a scene was named. Blender applies these left to right and -a
    # starts the render, so anything after it is ignored.
    argv = [exe, "-b", blend]
    if scene_name:
        argv += ["-S", sc.name]
    gpu = params.get("gpu") in (True, 1, "true", "True", "1")
    gpu_script = None
    if gpu:
        gpu_script = os.path.join(outdir, "%s_gpu.py" % job_id)
        with open(gpu_script, "w", encoding="utf-8") as fh:
            fh.write(_GPU_SCRIPT.replace("@@BACKENDS@@", repr((gpu_backend,) if gpu_backend else _GPU_BACKENDS)))
        argv += ["--python", gpu_script]            # runs after the file loads, before the frames
    region = _parse_region(params, sc, "NOTHING was started.")
    if region is not None:
        region_script = os.path.join(outdir, "%s_region.py" % job_id)
        with open(region_script, "w", encoding="utf-8") as fh:
            fh.write(_REGION_SCRIPT.replace("@@BORDER@@", repr(tuple(region["border"])))
                     .replace("@@CROP@@", repr(region["crop"])))
        argv += ["--python", region_script]
    argv += ["-s", str(start), "-e", str(end), "-j", str(step), "-a"]

    started = time.time()
    sid, held = childproc.acquire("blender render", os.path.basename(blend))
    if sid is None:
        raise MifOpError(childproc.busy_message("render_animation", held) + " NOTHING was started.")
    try:
        log_fh = open(log_path, "wb")
        popen = subprocess.Popen(argv, **childproc.hidden(dict(stdout=log_fh, stderr=subprocess.STDOUT,
                                                               stdin=subprocess.DEVNULL, cwd=outdir)))
    except Exception as exc:                     # noqa: BLE001
        childproc.release(sid)
        raise MifOpError("could not start '%s' (%s). NOTHING was started." % (exe, exc))
    watch = childproc.watch(popen, sid, None if mem_gb is None else childproc.memory_limit_bytes(float(mem_gb)),
                            "blender render")

    _JOBS[job_id] = {
        "popen": popen, "watch": watch, "logHandle": log_fh, "log": log_path, "paths": paths,
        "frames": frames, "before": before, "started": started, "blend": blend,
        "container": container, "argv": argv,
    }

    # render_info's OTHER diagnoses are carried as warnings rather than refusals. A black render is
    # not the same failure as a render that cannot run, and refusing on "no world" would block the
    # legitimate case of a scene lit entirely by emissive materials.
    warnings = [b for b in op_render_info({})["blockers"] if "no scene camera" not in b]
    # AND THE SIMULATIONS THIS RENDER WILL NOT ANIMATE - see _out_of_process_sim_rows. A warning, not a refusal, for the
    # reason the blockers above are: a caller may render a sim that is meant to hold still.
    sims = _out_of_process_sim_rows(sc, start)
    if sims:
        warnings.append("%d simulation(s) will not animate in this render of the saved file: %s. bake_physics, then "
                        "save_file, then render_animation." % (len(sims), ", ".join("%s (%s)" % (s["cache"], s["renders"])
                                                                                 for s in sims)))

    return {
        "ok": True,
        "jobId": job_id,
        "pid": popen.pid,
        "memoryLimitBytes": watch.limit or None,
        "gpuBackendRequested": gpu_backend,
        "blendFile": blend,
        "scene": sc.name,
        "sceneWasNamed": bool(scene_name),
        # SAID, NOT ASSUMED. In background mode the staleness of the file on disk is UNKNOWN to
        # this op, and a caller relying on it deserves to be told rather than reassured.
        "unsavedChangesChecked": dirty_checkable,
        "unsavedChangesWarning": (None if dirty_checkable else
                                  "bpy.data.is_dirty is always True under blender -b and never "
                                  "clears, so this op CANNOT tell whether the file on disk matches "
                                  "this session. If anything changed since the last save, the child "
                                  "is rendering the older scene."),
        "frameStart": start, "frameEnd": end, "frameStep": step,
        "framesExpected": len(frames),
        "outputDir": outdir,
        # A SAMPLE, AND THE NAME SAYS SO. This was outputPaths plus an outputPathsTruncated
        # boolean, and the boolean was a claim about the response that no suite could check -
        # reaching it needs a live Blender, past every refusal and a spawn. Rather than ship an
        # unverifiable flag, the truth moved into the field name: compare its length against
        # framesExpected. One fewer thing that can be wrong without anybody noticing.
        "outputPathsSample": paths if len(paths) <= 8 else paths[:8],
        "fileFormat": r.image_settings.file_format,
        "containerFormat": container,
        # STATED, not implied. For a movie container there is one file and the frames inside it
        # cannot be counted from outside, so render_status will never report per-frame progress.
        "framesVerifiable": not container,
        "existingFramesOverwritten": sorted(p for p, m in before.items() if m is not None),
        "log": log_path,
        "unbakedSimulations": sims or None,
        "region": (dict(region["pixels"], crop=region["crop"], border=[round(v, 6) for v in region["border"]],
                        setIn="the child process only") if region is not None else None),
        "warnings": warnings,
        "note": ("started out of process on the SAVED file - this session's unsaved state is not "
                 "part of it. Poll with render_status(jobId). The job is process-local and is "
                 "forgotten if Blender restarts."),
    }


def op_render_status(params):
    """How far has an out-of-process render actually got - measured on disk, not asked of anyone.

    THREE ANSWERS THAT MUST STAY DISTINCT, because collapsing any two of them strands a caller:

      unknown job    this Blender has never heard of that id. It was never started here, or Blender
                     restarted and the table went with it. NOT "still running" - a caller told that
                     waits forever for a process nobody is running.
      running        the child is alive. framesRendered is what is ON DISK with an mtime at or after
                     the job start, never what the child claims.
      finished       the child exited. exitCode is reported RAW, and finished does NOT mean complete:
                     a render killed halfway exits non-zero with real frames on disk, and both facts
                     are reported rather than one being turned into a verdict.

    Called with no jobId it lists the jobs this Blender knows about.

    params:
      jobId / job / id (str)  which job. Omit to list all known jobs.
      logLines (int)          how much log tail to return, 1 to 2000. Default 20.
    """
    reject_unknown(params, _STATUS_KEYS, "render_status")
    job_id = take(params, "jobId", "job", "id", kind=str)
    lines = take_int(params, "logLines", default=20)
    # RANGE-CHECKED: the tail is text.splitlines()[-lines:], so 0 answered every line of the 64 KB window and a negative
    # number dropped lines from the front (found 2026-10-25).
    if not 1 <= lines <= 2000:
        raise MifOpError("logLines is how many lines of the log's tail to return, 1 to 2000 - got %d." % lines)

    if not job_id:
        return {
            "ok": True,
            "jobs": [{"jobId": k, "pid": j["popen"].pid, "running": j["popen"].poll() is None,
                      "framesExpected": len(j["frames"]), "blendFile": j["blend"]}
                     for k, j in sorted(_JOBS.items())],
            "note": ("jobs are process-local. An id started before a Blender restart is not here "
                     "and reports unknownJob, which is not the same as unfinished."),
        }

    job = _JOBS.get(job_id)
    if job is None:
        return {
            "ok": True,
            "unknownJob": True,
            "jobId": job_id,
            "knownJobs": sorted(_JOBS),
            "error": ("no job '%s' in this Blender. It was never started here, or Blender has "
                      "restarted since - the table does not survive that. This is NOT a report "
                      "that the render is unfinished." % job_id),
        }

    code = job["popen"].poll()
    running = code is None

    # THE MEASUREMENT. A frame counts only if it exists AND its mtime is at or after the job start.
    # Existence alone counts leftovers from a previous run, which is how a progress bar reads 100%
    # before anything has been drawn - the same rule render_still needed and for the same reason.
    rendered, stale = [], []
    for p in job["paths"]:
        try:
            mtime = os.path.getmtime(p)
        except OSError:
            continue
        (rendered if mtime >= job["started"] else stale).append(p)

    out = {
        "ok": True,
        "jobId": job_id,
        "pid": job["popen"].pid,
        "running": running,
        "exitCode": code,
        "elapsedSeconds": round(time.time() - job["started"], 1),
        "framesExpected": len(job["frames"]),
        "blendFile": job["blend"],
        "outputDir": os.path.dirname(job["paths"][0]),
        "logTail": _log_tail(job["log"], lines),
        "log": job["log"],
        "gpuNote": _gpu_log_note("\n".join(_log_tail(job["log"], 400))),
    }
    if job.get("watch") is not None:
        out["job"] = job["watch"].report()       # the tree's peak memory, the ceiling, and a kill past it
    if job["container"]:
        # ONE FILE, and the frames inside it are not countable from here. Saying so is the honest
        # answer; a count derived from the frame range would be a restatement of the request.
        out["framesVerifiable"] = False
        out["containerWritten"] = bool(rendered)
        out["containerPath"] = job["paths"][0]
        out["note"] = ("a movie container was requested, so there is one output file and the frame "
                       "count inside it cannot be verified from the filesystem. containerWritten "
                       "says only that the file was touched by this job.")
    else:
        out["framesVerifiable"] = True
        out["framesRendered"] = len(rendered)
        out["framesRemaining"] = len(job["frames"]) - len(rendered)
        out["staleFramesIgnored"] = len(stale)
        if stale:
            out["staleNote"] = ("%d expected path(s) exist but predate this job and are NOT counted "
                                "- they are leftovers from an earlier render" % len(stale))
    if not running:
        _close_log(job)
        out["complete"] = (code == 0 and (out.get("containerWritten")
                                          if job["container"] else not out["framesRemaining"]))
        if code != 0:
            out["error"] = ("the render process exited with code %s. Frames already on disk are "
                            "still reported above - a killed render leaves real output." % code)
        killed = (out.get("job") or {}).get("killedForMemory")
        if killed:
            out["error"] = ("the render's process tree was ENDED at %.1f GB, past its memory ceiling of %.1f GB "
                            "(memoryLimitGB / MIF_CHILD_JOB_MEMORY_GB). Frames already on disk are reported above."
                            % (killed["bytes"] / float(1 << 30), killed["limit"] / float(1 << 30)))
    return out


def _close_log(job):
    """Release the log handle once the child is gone, so the file is not held open forever."""
    fh = job.pop("logHandle", None)
    if fh is not None:
        try:
            fh.close()
        except Exception:                        # noqa: BLE001
            pass


_CM_KEYS = {"viewTransform", "look", "exposure", "gamma", "displayDevice",
            "sequencerColorspace", "useCurveMapping", "workingSpace", "convertColors"}


def enum_ids(holder, prop):
    """The identifiers an enum ACTUALLY offers on this holder, or None if there is no such property.

    READ FROM THE INSTANCE, NOT THE TYPE, and that is the whole point for colour management.
    Everywhere else in this addon an enum is validated against bpy.types.X.bl_rna - light type,
    camera type, constraint type - because those sets are fixed by the build. Colour management is
    not: the OCIO config populates view_transform, look and display_device at runtime, so
    bpy.types.ColorManagedViewSettings.bl_rna gives the DEFAULT set while the scene in front of you
    may offer something else entirely. A studio config renames all of them, and Blender's own set
    changed under us - Filmic through 3.x, AgX from 4.0, with Khronos PBR Neutral added later.

    So a remembered list would be wrong on any config that is not stock, and wrong on stock Blender
    of a different version. The instance knows.
    """
    try:
        return {i.identifier for i in holder.bl_rna.properties[prop].enum_items}
    except (KeyError, AttributeError, TypeError):
        return None


def enum_is_trustworthy(holder, prop, ids):
    """Does this enum's item list actually describe what the property accepts?

    THE DOCSTRING ABOVE WAS WRONG ABOUT THE ONE PROPERTY IT WAS WRITTEN FOR. It reasons that reading
    enum_items off the INSTANCE gives what the loaded OCIO config offers. It does not: OCIO
    populates view_transform through a runtime callback, and the RNA item list is a static
    placeholder. Measured on 4.4.0, stock config:

        enum_items                     ['NONE']
        enum_items_static              ['NONE']
        scene.view_settings.view_transform   'AgX'
        setting it to 'Standard'       works

    So every real transform was refused with "Available: NONE", the value the scene was ALREADY
    using was absent from its own available list, and 'NONE' - the one identifier that IS listed -
    passed validation and then raised a raw bpy TypeError through to the caller.

    THE TELL IS SELF-EVIDENT AND NEEDS NO VERSION TABLE: a list that does not contain the value the
    property CURRENTLY HOLDS is not a list of what that property allows. That is the check, and it
    stays correct on a studio config, on a future Blender, and on the properties where the list IS
    honest - display_device and sequencer_colorspace behave the same way, which is why this is a
    helper rather than a special case for view_transform.
    """
    if not ids:
        return False
    try:
        current = getattr(holder, prop, None)
    except Exception:                                              # noqa: BLE001
        return False
    return current is None or current in ids


def _pick_enum(holder, prop, value, label, verb):
    ids = enum_ids(holder, prop)
    if ids is None:
        raise MifOpError("this Blender has no '%s' property for %s, so it cannot be set here. "
                         "NOTHING was changed." % (prop, label))
    # ONLY VALIDATE AGAINST A LIST THAT DESCRIBES REALITY. See enum_is_trustworthy: for the OCIO
    # properties the RNA list is a static ['NONE'] placeholder, so pre-validating against it
    # refused every real value and admitted the one value that cannot be set.
    if enum_is_trustworthy(holder, prop, ids):
        if value not in ids:
            raise MifOpError("'%s' is not a %s this configuration offers. Available: %s. This list "
                             "comes from the OCIO config actually loaded, not a remembered one - a "
                             "studio config renames all of them. NOTHING was changed."
                             % (value, label, ", ".join(sorted(ids))))
        return value

    # UNTRUSTWORTHY LIST: let BLENDER decide, and turn its TypeError into a refusal in this addon's
    # own voice. Blender's message names the values that actually work, which is more than the RNA
    # list could tell us - so the caller ends up better informed than the old check left them.
    try:
        setattr(holder, prop, value)
    except Exception as exc:                                       # noqa: BLE001
        raise MifOpError("'%s' is not a %s this configuration offers - Blender refused it and named "
                         "what it does offer: %s. (The RNA item list for this property is a static "
                         "placeholder, so it cannot be checked in advance - see enum_is_trustworthy.) "
                         "NOTHING was changed." % (value, label, exc))
    return value


def op_set_color_management(params):
    """View transform, look, exposure and gamma - the settings that change every pixel silently.

    THE LAST OF THE FIVE Tier 1 gaps, and the one behind "why does this look washed out". Colour
    management is applied to every render and every saved image, it is invisible in the scene data,
    and its default has CHANGED under this addon's supported range: Filmic through 3.x, AgX from
    4.0, with more added since. Nothing here could set it and only render_info could read it.

    EVERY ENUM IS VALIDATED AGAINST WHAT THIS BUILD WILL ACTUALLY ACCEPT - which is NOT the same as
    reading the RNA item list, and that distinction cost this op its entire usefulness for a while.
    Corrected 2026-09-07: for the OCIO-driven properties the RNA list is a static placeholder.
    Measured on 4.4.0 stock, enum_items for view_transform is ['NONE'] while the scene is USING
    'AgX' - a list that does not contain the value the property currently holds. So "Standard" was
    refused with "Available: NONE" and 'NONE', the one listed identifier, passed validation and
    raised a raw bpy TypeError. The check is now that self-evident test (enum_is_trustworthy), and
    where the list fails it Blender decides and its refusal is translated - which tells a caller MORE
    than the RNA ever could, because Blender's message names the real values.

    THE ORIGINAL REASONING BELOW IS STILL WHY THE LIST IS NOT HARD-CODED HERE, and it was right
    about that: a studio OCIO config renames every view transform and
    look, so a hard-coded set would refuse the only values that work and accept none of them.

    THE LOOK IS NAMESPACED BY THE VIEW TRANSFORM, which is why the order here matters. In 4.x a
    look reads as "AgX - Punchy", so a look valid under one transform is not offered under another,
    and setting both at once against the OLD transform's list would refuse a legal combination. So
    the transform is applied FIRST and the look is then validated against what the NEW transform
    offers - the same rule set_light already follows for per-type keys, validating against the type
    the light will BE rather than the one it was.

    params:
      viewTransform (str)        e.g. AgX, Filmic, Standard - whatever this config offers
      look (str)                 validated against the view transform in force AFTER this call
      exposure (float)           stops, applied before the transform
      gamma (float)
      displayDevice (str)        sRGB, Rec.1886, ... from the config
      sequencerColorspace (str)
      useCurveMapping (bool)
      workingSpace (str)         5.0+: the scene-linear working space (Linear Rec.709, Linear Rec.2020, ACEScg),
                                 through wm.set_working_color_space; refused by name on older builds (2026-10-06)
      convertColors (bool)       with workingSpace: convert the file's colours into the new space (default true,
                                 Blender's own default) or leave the numbers as they are
    """
    reject_unknown(params, _CM_KEYS, "set_color_management", {
        "ocio": "the OCIO config is chosen by the OCIO environment variable before Blender starts, "
                "not through the API. This op sets what the LOADED config offers, and a refusal "
                "lists those - which is also how you find out which config you have.",
        "ocioConfig": "see 'ocio'.",
        "lut": "there is no LUT slot in colour management. A `look` is the LUT-shaped control here, "
               "and it is validated against the view transform in force AFTER this call, because "
               "Blender resets the look when the transform changes under it.",
        "colorspace": "spelt sequencerColorspace for the strip one, or displayDevice for the "
                      "display. A bare 'colorspace' does not say which, and they are different "
                      "settings with different enums.",
    })
    sc = bpy.context.scene
    vs = getattr(sc, "view_settings", None)
    ds = getattr(sc, "display_settings", None)
    if vs is None:
        raise MifOpError("this scene has no view_settings, so colour management cannot be set. "
                         "NOTHING was changed.")

    before = {
        "viewTransform": getattr(vs, "view_transform", None),
        "look": getattr(vs, "look", None),
        "exposure": round(float(getattr(vs, "exposure", 0.0)), 6),
        "gamma": round(float(getattr(vs, "gamma", 1.0)), 6),
        "displayDevice": getattr(ds, "display_device", None) if ds else None,
    }

    vt = take(params, "viewTransform", kind=str)
    look = take(params, "look", kind=str)
    dev = take(params, "displayDevice", kind=str)
    seq = take(params, "sequencerColorspace", kind=str)
    exposure = take_float(params, "exposure", default=None)
    gamma = take_float(params, "gamma", default=None)
    curve = params.get("useCurveMapping")
    working = take(params, "workingSpace", kind=str)
    convert = take_bool(params, "convertColors", default=True) if params.get("convertColors") is not None else None
    if convert is not None and working is None:
        raise MifOpError("convertColors goes with workingSpace. NOTHING was changed.")
    if working is not None and getattr(getattr(bpy.data, "colorspace", None), "working_space", None) is None:
        raise MifOpError("workingSpace needs Blender 5.0 or later (bpy.data.colorspace.working_space); this is %s. "
                         "NOTHING was changed." % bpy.app.version_string)
    working_before = getattr(getattr(bpy.data, "colorspace", None), "working_space", None)
    if not any(v is not None for v in (vt, look, dev, seq, exposure, gamma, curve, working)):
        raise MifOpError("nothing to do - pass at least one of viewTransform, look, exposure, "
                         "gamma, displayDevice, sequencerColorspace, useCurveMapping or workingSpace. NOTHING "
                         "was changed.")

    # READ AND CHECKED BEFORE THE FIRST WRITE (2026-09-21): useCurveMapping and the sequencer colour space were
    # checked after exposure and gamma had been written - "Everything else requested was applied".
    want_curve = take_bool(params, "useCurveMapping", default=False) if curve is not None else None
    scs = getattr(sc, "sequencer_colorspace_settings", None)
    if seq is not None:
        if scs is None:
            raise MifOpError("this Blender has no sequencer_colorspace_settings, so sequencerColorspace cannot be "
                             "set. NOTHING was changed.")
        ids = enum_ids(scs, "name")
        # TRUST-CHECKED like the other OCIO-driven lists: this enum can expose a static placeholder list, and a check
        # against it refuses a real colour space (found 2026-10-25). Untrusted, the read-back below decides.
        if ids is not None and enum_is_trustworthy(scs, "name", ids) and str(seq) not in ids:
            raise MifOpError("sequencerColorspace '%s' is not a colour space this configuration offers. Available: "
                             "%s. NOTHING was changed." % (seq, ", ".join(sorted(ids))))

    # EVERYTHING THIS OP CAN WRITE, AND PUT BACK IN THIS ORDER on any refusal (2026-09-21). The device, then the
    # transform, then the LOOK: Blender resets the look whenever the transform changes, so the restore below used to
    # put the transform back and lose the caller's look all the same - {viewTransform: Standard, look: <none of
    # Standard's>} on an AgX scene with "AgX - Punchy" set came back AgX with look None.
    device_before = ds.display_device if ds is not None else None
    transform_before = vs.view_transform
    look_before = vs.look
    rest_before = [(vs, a, getattr(vs, a)) for a in ("exposure", "gamma", "use_curve_mapping") if hasattr(vs, a)]
    seq_before = scs.name if scs is not None else None

    order = []
    if ds is not None and device_before is not None:
        order.append((ds, "display_device", device_before))
    order += [(vs, "view_transform", transform_before), (vs, "look", look_before)] + rest_before
    if scs is not None and seq_before is not None:
        order.append((scs, "name", seq_before))

    def restore_snapshot():
        for holder, attr, value in order:
            try:
                setattr(holder, attr, value)
            except (TypeError, ValueError, AttributeError):   # noqa: PERF203 - put back what can be
                pass

    # THE DISPLAY DEVICE GOES FIRST because it can re-populate the view transform list: the
    # transforms on offer are those the config defines FOR THAT DISPLAY. Validating a transform
    # against the old device's set would refuse a value that is about to become legal.
    try:
        if dev is not None:
            if ds is None:
                raise MifOpError("this scene has no display_settings, so displayDevice cannot be "
                                 "set. NOTHING was changed.")
            ds.display_device = _pick_enum(ds, "display_device", str(dev), "display device",
                                           "set_color_management")
        if vt is not None:
            vs.view_transform = _pick_enum(vs, "view_transform", str(vt), "view transform",
                                           "set_color_management")
        if look is not None:
            # VALIDATED AFTER the transform was applied, against what it NOW offers - see the
            # docstring - which is exactly why the device has to be PUT BACK if it refuses.
            vs.look = _pick_enum(vs, "look", str(look), "look", "set_color_management")
    except MifOpError:
        # The ordering above is deliberate and cannot change: the device decides which transforms
        # exist, so it must be written before a transform can be validated. That left a refused call
        # having switched the display device while promising "NOTHING was changed" - true of the
        # transform, false of the scene. Undone here rather than by reordering, because the order is
        # the thing that makes the validation correct. The TRANSFORM is put back for the same
        # reason one step further down: `look` is validated against what the transform now offers,
        # so a refused look had already moved the transform - and the look goes back after it.
        restore_snapshot()
        raise
    if exposure is not None:
        vs.exposure = exposure
    if gamma is not None:
        vs.gamma = gamma
    if want_curve is not None and hasattr(vs, "use_curve_mapping"):
        vs.use_curve_mapping = want_curve
    if seq is not None:
        # GUARDED, now that an untrustworthy enum list is not checked against above: a name Blender refuses raises
        # TypeError here, and everything written so far goes back.
        try:
            scs.name = str(seq)
        except TypeError as exc:
            restore_snapshot()
            raise MifOpError("sequencerColorspace '%s' is not a colour space this configuration offers: %s. NOTHING "
                             "was changed." % (seq, str(exc).split(":", 1)[-1].strip()))
    if working is not None and working != working_before:
        # LAST, because the operator decides what it accepts (its enum is generated) and converts colours as it goes:
        # a name it refuses puts everything above back.
        try:
            bpy.ops.wm.set_working_color_space(working_space=str(working),
                                               convert_colors=True if convert is None else convert)
        except TypeError as exc:
            restore_snapshot()
            raise MifOpError("workingSpace %r is not one this config offers: %s. NOTHING was changed."
                             % (working, str(exc).split(":", 1)[-1].strip()))

    after = {
        "viewTransform": getattr(vs, "view_transform", None),
        "look": getattr(vs, "look", None),
        "exposure": round(float(getattr(vs, "exposure", 0.0)), 6),
        "gamma": round(float(getattr(vs, "gamma", 1.0)), 6),
        "displayDevice": getattr(ds, "display_device", None) if ds else None,
        "sequencerColorspace": getattr(getattr(sc, "sequencer_colorspace_settings", None),
                                       "name", None),
        "useCurveMapping": bool(getattr(vs, "use_curve_mapping", False)),
        "workingSpace": getattr(getattr(bpy.data, "colorspace", None), "working_space", None),
    }
    # EACH REQUESTED WRITE VERIFIED. Blender SILENTLY RESETS look to 'None' when the view transform
    # changes out from under it, so a call that set both could apply the transform, drop the look,
    # and report a success.
    #
    # WHAT THIS ADDS, MEASURED RATHER THAN ASSUMED. `after` is read from the scene, so the response
    # is honest with or without the check below - deleting it alone changed nothing in B117, which
    # was established by planting exactly that. What it adds is turning a silently WRONG outcome
    # into a loud one: with the ordering also broken, the op would return look:'None' to a caller
    # who asked for something else, ok:true and all. The two together are caught by B117; this
    # guard on its own is defence in depth, and saying so is more useful than implying a plant
    # proved it.
    wrong = {}
    _vt_ids = enum_ids(vs, "view_transform")
    _look_ids = enum_ids(vs, "look")
    _dd_ids = enum_ids(ds, "display_device") or []
    for key, want in (("viewTransform", vt), ("look", look), ("displayDevice", dev),
                      ("sequencerColorspace", seq)):
        if want is not None and after.get(key) != str(want):
            wrong[key] = (str(want), after.get(key))
    if wrong:
        restore_snapshot()
        # THE WORKING SPACE TOO, which restore_snapshot does not hold: it is file-wide and changed through an operator
        # that converts colours as it goes, so it is put back the same way, converting them back (found 2026-10-25).
        if working is not None and working_before is not None and after["workingSpace"] != working_before:
            try:
                bpy.ops.wm.set_working_color_space(working_space=str(working_before),
                                                   convert_colors=True if convert is None else convert)
            except (TypeError, RuntimeError):
                pass
        raise MifOpError("wrote %s but the scene reads back %s afterwards. Blender resets `look` "
                         "when the view transform changes, so a look must be one the NEW transform "
                         "offers. Everything this call wrote was put back."
                         % ({k: v[0] for k, v in wrong.items()},
                            {k: v[1] for k, v in wrong.items()}))
    return {
        "ok": True,
        "scene": sc.name,
        "before": before,
        # None RATHER THAN A WRONG LIST. Reporting ['NONE'] as the available transforms - which is
        # what the RNA gives - tells a caller something false in a field they would act on. An
        # absent answer is worse than a right one and far better than a confident wrong one.
        "availableViewTransforms": (sorted(_vt_ids) if enum_is_trustworthy(vs, "view_transform",
                                                                          _vt_ids) else None),
        "availableLooks": (sorted(_look_ids) if enum_is_trustworthy(vs, "look", _look_ids)
                           else None),
        "availableListsNote": (None if enum_is_trustworthy(vs, "view_transform", _vt_ids) else
                               "null means this Blender's RNA does not expose the real list for "
                               "these OCIO-driven properties - it reports a static placeholder that "
                               "does not even contain the value in use. Pass a value and Blender "
                               "will name the alternatives if it is wrong."),
        # THE SAME GUARD AS THE TWO ABOVE, and this field did not have it. enum_is_trustworthy's own
        # docstring names display_device as one of the properties that behaves this way - "which is
        # why this is a helper rather than a special case for view_transform" - and then this line
        # called enum_ids directly and published the placeholder.
        #
        # MEASURED 2026-09-15 on 4.2.17 LTS and 5.2.1 LTS: availableDisplayDevices came back
        # ["NONE"] on both, while Blender's own refusal for a bad value names
        # ('sRGB', 'Display P3', 'Rec.2020', ...) and the scene was sitting on 'sRGB' - a value
        # absent from its own available list, which is exactly the tell the helper tests for.
        # A caller picking "a device that is not the current one" from that list picks NONE, the one
        # identifier that is listed and the one value that is refused.
        "availableDisplayDevices": (sorted(_dd_ids) if enum_is_trustworthy(ds, "display_device",
                                                                          _dd_ids) else None),
        "note": ("these lists come from the OCIO config actually loaded, not from a remembered "
                 "set - the defaults changed from Filmic to AgX in 4.0 and a studio config renames "
                 "all of them."),
        **after,
    }

# ---------------------------------------------------------------------------------------------------------------
# compose_atlas - rendered frames tiled into one flipbook sheet
# ---------------------------------------------------------------------------------------------------------------
# WHY AN OP AND NOT A SCRIPT. A particle flipbook is N rendered frames in a columns x rows grid, read by Unreal's
# SubUV left-to-right, top-to-bottom; nothing in the add-on could put two images into one. The tiling needs the
# pixels, and Blender already holds an image reader, a PNG writer and numpy - so it happens here, on the frames
# render_animation wrote, with no dependency the caller has to install.
#
# ALPHA IS A DECISION, not a copy (alphaMode):
#   film       the rendered alpha - right for absorbing subjects on a transparent film (smoke, dust).
#   luminance  alpha = clamp(max(R,G,B) x alphaGain) of the LINEAR colour, and the colour divided by it - right for
#              emissive subjects rendered over black (fire, flashes, glows), whose film alpha is ~0 because emission
#              absorbs nothing. Dividing is what makes a translucent material's Emissive x Opacity give back exactly
#              the rendered emission, so the sprite reads as the render did.
#   hybrid     BOTH, from two renders of the same frames: `frames` over black with an opaque film (the light) and
#              `alphaFrames` on a transparent film (the absorbing smoke's coverage). alpha = max(the film's alpha, the
#              light's luminance alpha) and the colour is the light divided by it - so a fireball's flame AND the dark
#              smoke it turns into are one sheet. Luminance alone cannot hold dark smoke (it has no light, so no
#              alpha); film alone loses the flame (emission absorbs nothing). Measured on 5.2.1:
#              a burst sheet whose fire cooled into dark grey smoke visible on a grey sky.
# All arithmetic is in linear light (PNG values are sRGB-decoded first and encoded again on the way out).
#
# loopBlend K crossfades the last K cells with the K frames rendered before the first cell, so a looping flipbook
# has no pop at the wrap. It needs columns x rows + K frames.
#
# normalOutput writes a second sheet: a tangent-space normal (DirectX / Unreal green-down) derived per cell from the
# alpha's gradient - a soft-body normal for lit smoke, not a render pass.
#
# alphaFloor is the black point: alpha = (a - floor) / (1 - floor), clamped. A volume render is never exactly empty -
# EEVEE's sampling leaves a faint haze over the whole frame (measured on a 5.2.1 fire sheet: 3% of all pixels at
# alpha 0-0.01 in areas that hold nothing), which a sprite would draw as a dirty square. And a colour is written only
# where the stored 8-bit alpha is not 0: dividing by a tiny alpha had left saturated noise under alpha 0 (RGB up to
# 0.84 there), which block compression and mip maps would bleed into the flame's edge.
#
# motionOutput writes a third sheet: each cell's MOTION toward the next cell, as optical flow between the two frames
# (2026-10-11). A material that knows it can blend two cells by moving each toward the other instead of crossfading
# them, which is what makes a flipbook of few frames play smoothly. The flow is computed here, from the frames, and not
# taken from a render pass: a fluid domain is a static box, so a renderer's vector pass holds the box's motion (none),
# never the gas's - the pass is for moving surfaces. Pyramidal Lucas-Kanade on each cell's coverage (alpha), three
# levels, warped and refined; RG = 0.5 + flow / (2 x motionScale) in pixels of the cell, green DOWN (a uv's v grows
# downward, so the flow reads straight into Unreal's uv), B 0, A 1. The last cell's motion is toward the first when
# loopBlend makes the sheet loop, else 0. motionResidual says, per cell, how much of the difference to the next cell the
# flow explains: |next warped back - this| over |next - this|, under 1 when the motion is real.
_ATLAS_KEYS = {"frames", "columns", "rows", "output", "overwrite", "alphaMode", "alphaGain", "alphaFloor",
               "loopBlend", "normalOutput", "normalStrength", "alphaFrames", "motionOutput", "motionScale"}


def _blur(np, img, sigma):
    """Separable Gaussian blur of a 2-D array (edges clamped) - numpy only; Blender ships no scipy."""
    r = max(1, int(round(sigma * 2.5)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1, dtype=np.float32) / float(sigma)) ** 2)
    k /= k.sum()
    out = img
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (r, r)
        p = np.pad(out, pad, mode="edge")
        acc = np.zeros_like(out)
        n = out.shape[axis]
        for i, w in enumerate(k):
            acc += w * (p[i:i + n, :] if axis == 0 else p[:, i:i + n])
        out = acc
    return out


def _warp(np, img, fx, fy):
    """img sampled at (x + fx, y + fy), bilinear, edges clamped."""
    h, w = img.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    x = np.clip(xs + fx, 0.0, w - 1.001)
    y = np.clip(ys + fy, 0.0, h - 1.001)
    x0, y0 = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
    tx, ty = x - x0, y - y0
    a = img[y0, x0] * (1 - tx) + img[y0, x0 + 1] * tx
    b = img[y0 + 1, x0] * (1 - tx) + img[y0 + 1, x0 + 1] * tx
    return a * (1 - ty) + b * ty


def _optical_flow(np, a, b, levels=3, iters=4, window=2.0):
    """Per-pixel (fx, fy) in pixels with a(p) ~ b(p + f): pyramidal Lucas-Kanade, coarse to fine."""
    pa, pb = [a], [b]
    for _ in range(levels - 1):
        if min(pa[-1].shape) < 16:
            break
        pa.append(_blur(np, pa[-1], 1.0)[::2, ::2])
        pb.append(_blur(np, pb[-1], 1.0)[::2, ::2])
    fx = np.zeros_like(pa[-1])
    fy = np.zeros_like(pa[-1])
    for lvl in range(len(pa) - 1, -1, -1):
        A, B = pa[lvl], pb[lvl]
        if fx.shape != A.shape:
            fx = 2.0 * np.repeat(np.repeat(fx, 2, 0), 2, 1)[:A.shape[0], :A.shape[1]]
            fy = 2.0 * np.repeat(np.repeat(fy, 2, 0), 2, 1)[:A.shape[0], :A.shape[1]]
        for _ in range(iters):
            Bw = _warp(np, B, fx, fy)
            m = 0.5 * (A + Bw)
            ix = np.zeros_like(m)
            iy = np.zeros_like(m)
            ix[:, 1:-1] = (m[:, 2:] - m[:, :-2]) * 0.5
            iy[1:-1, :] = (m[2:, :] - m[:-2, :]) * 0.5
            it = Bw - A
            sxx, syy, sxy = _blur(np, ix * ix, window), _blur(np, iy * iy, window), _blur(np, ix * iy, window)
            sxt, syt = _blur(np, ix * it, window), _blur(np, iy * it, window)
            det = sxx * syy - sxy * sxy
            ok = det > 1e-7
            safe = np.where(ok, det, 1.0)
            fx = fx + np.where(ok, -(syy * sxt - sxy * syt) / safe, 0.0)
            fy = fy + np.where(ok, -(sxx * syt - sxy * sxt) / safe, 0.0)
    return fx, fy
_ATLAS_ALPHA_MODES = ("film", "luminance", "hybrid")
_ATLAS_MAX_EDGE = 8192
_ATLAS_MAX_CELLS = 256


def _srgb_to_linear(np, c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(np, c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)


def _load_frame(np, path):
    """(h, w, 4) float32, TOP row first, straight alpha as the file stores it - and the datablock gone again."""
    img = bpy.data.images.load(path, check_existing=False)
    try:
        w, h = int(img.size[0]), int(img.size[1])
        if w <= 0 or h <= 0:
            raise MifOpError("'%s' loaded with a zero dimension (%dx%d)" % (path, w, h))
        buf = np.empty(w * h * 4, dtype=np.float32)
        img.pixels.foreach_get(buf)
        return buf.reshape(h, w, 4)[::-1].copy()
    finally:
        try:
            bpy.data.images.remove(img)
        except Exception:                                           # noqa: BLE001
            pass


def _write_sheet(np, rgba_top_first, path, is_data):
    """Write (H, W, 4) values in [0, 1] as an 8-bit RGBA PNG, and measure the file rather than trust the save."""
    h, w = int(rgba_top_first.shape[0]), int(rgba_top_first.shape[1])
    path = abs_output_path(path)                    # Image.save() writes nothing to a relative path (abs_output_path)
    img = bpy.data.images.new(os.path.splitext(os.path.basename(path))[0], width=w, height=h, alpha=True,
                              float_buffer=False, is_data=bool(is_data))
    try:
        img.alpha_mode = "STRAIGHT"
        img.pixels.foreach_set(np.ascontiguousarray(rgba_top_first[::-1], dtype=np.float32).ravel())
        img.filepath_raw = path
        img.file_format = "PNG"
        before = os.path.getmtime(path) if os.path.isfile(path) else None
        img.save()
    finally:
        try:
            bpy.data.images.remove(img)
        except Exception:                                           # noqa: BLE001
            pass
    size = os.path.getsize(path) if os.path.isfile(path) else -1
    fresh = size > 0 and (before is None or os.path.getmtime(path) > before)
    return {"path": path, "wroteFile": fresh, "fileBytes": size, "width": w, "height": h}


def op_compose_atlas(params):
    """Tile rendered frames into one flipbook atlas (left-to-right, top-to-bottom), with an alpha rule, an optional
    loop crossfade and an optional derived normal sheet. Writes PNG files; adds nothing to the scene.

    params:
      frames (list of str, required)  image files in playback order - columns x rows + loopBlend of them
      columns, rows (int, required)   the grid; the atlas is columns x cellWidth by rows x cellHeight
      output (str, required)          the atlas PNG path
      overwrite (bool)                replace an existing output file (default false - refused)
      alphaMode (str)                 film | luminance | hybrid (default film) - see the block comment above
      alphaFrames (list of str)       hybrid only, and required there: the same frames rendered on a transparent
                                      film, one per frame, in the same order
      alphaGain (float)               luminance alpha = clamp(max(RGB) x gain); default 1.5
      alphaFloor (float)              black point below which alpha is 0; default 0.01, 0 to keep every value
      loopBlend (int)                 frames of crossfade at the wrap; default 0
      normalOutput (str)              also write a derived normal sheet here
      normalStrength (float)          slope of that normal; default 2.0
      motionOutput (str)              also write a motion-vector sheet here: each cell's optical flow to the next
      motionScale (float)             the flow in pixels that RG 0 / 1 encode; default: the sheet's own largest
                                      motion (99th percentile, at least 1) - the reply says which
    """
    reject_unknown(params, _ATLAS_KEYS, "compose_atlas", notes={
        "cellSize": "cells are the frames' own size - render at the cell size you want (set_render_settings "
                    "resolutionX/resolutionY); this op never resamples a frame",
        "directory": "pass `frames` as an explicit ordered list - a directory listing's order is not the playback "
                     "order, and a stray file in it would become a frame",
        "fps": "an atlas has no frame rate - the Niagara SubUV module or material decides playback speed"})
    frames = take(params, "frames", required=True, kind=list)
    cols = finite_int(take_float(params, "columns", required=True), "columns")
    rows = finite_int(take_float(params, "rows", required=True), "rows")
    out_raw = str(take(params, "output", required=True, kind=str))
    out_path = bpy.path.abspath(out_raw)
    overwrite = take_bool(params, "overwrite", default=False)
    mode = str(take(params, "alphaMode", default="film", kind=str)).strip().lower()
    alpha_frames = take(params, "alphaFrames", default=None, kind=list)
    gain = finite_float(take_float(params, "alphaGain", default=1.5), "alphaGain")
    floor = finite_float(take_float(params, "alphaFloor", default=0.01), "alphaFloor")
    blend = finite_int(take_float(params, "loopBlend", default=0), "loopBlend")
    n_raw = take(params, "normalOutput", default=None, kind=str)
    n_path = bpy.path.abspath(str(n_raw)) if n_raw else None
    n_strength = finite_float(take_float(params, "normalStrength", default=2.0), "normalStrength")
    m_raw = take(params, "motionOutput", default=None, kind=str)
    m_path = bpy.path.abspath(str(m_raw)) if m_raw else None
    m_scale = take_float(params, "motionScale", default=None)
    if m_scale is not None:
        m_scale = finite_float(m_scale, "motionScale")
        if not m_path:
            raise MifOpError("motionScale goes with motionOutput - it encodes the motion sheet. NOTHING was written.")
        if not (0.0 < m_scale <= 1024.0):
            raise MifOpError("motionScale must be above 0 and at most 1024 pixels, got %r. NOTHING was written."
                             % m_scale)

    if cols < 1 or rows < 1 or cols * rows > _ATLAS_MAX_CELLS:
        raise MifOpError("columns x rows must be between 1 and %d cells, got %d x %d. NOTHING was written."
                         % (_ATLAS_MAX_CELLS, cols, rows))
    if mode not in _ATLAS_ALPHA_MODES:
        raise MifOpError("alphaMode must be one of %s, got %r. NOTHING was written."
                         % (", ".join(_ATLAS_ALPHA_MODES), mode))
    if mode == "hybrid" and not alpha_frames:
        raise MifOpError("alphaMode hybrid needs alphaFrames - the same frames rendered on a transparent film, which "
                         "give the smoke its coverage. NOTHING was written.")
    if alpha_frames and mode != "hybrid":
        raise MifOpError("alphaFrames is read by alphaMode hybrid only; with %s it would be ignored. NOTHING was "
                         "written." % mode)
    if not (0.0 < gain <= 1000.0):
        raise MifOpError("alphaGain must be above 0 and at most 1000, got %r. NOTHING was written." % gain)
    if not (0.0 <= floor < 0.5):
        raise MifOpError("alphaFloor must be at least 0 and below 0.5, got %r. NOTHING was written." % floor)
    if blend < 0 or blend >= cols * rows:
        raise MifOpError("loopBlend must be between 0 and cells - 1 (%d), got %d. NOTHING was written."
                         % (cols * rows - 1, blend))
    if not (0.0 < n_strength <= 100.0):
        raise MifOpError("normalStrength must be above 0 and at most 100, got %r. NOTHING was written." % n_strength)
    want = cols * rows + blend
    if len(frames) != want:
        raise MifOpError("a %d x %d atlas with loopBlend %d needs exactly %d frames, got %d. NOTHING was written."
                         % (cols, rows, blend, want, len(frames)))
    paths = []
    for i, f in enumerate(frames):
        if not isinstance(f, str) or not f:
            raise MifOpError("frames[%d] must be a file path, got %r. NOTHING was written." % (i, f))
        paths.append(bpy.path.abspath(f))
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise MifOpError("%d of the %d frames do not exist, first %s - a render that has not finished, or a "
                         "different naming. NOTHING was written." % (len(missing), len(paths), missing[0]))
    alpha_paths = []
    if alpha_frames:
        if len(alpha_frames) != len(paths):
            raise MifOpError("alphaFrames needs one file per frame (%d), got %d. NOTHING was written."
                             % (len(paths), len(alpha_frames)))
        for i, f in enumerate(alpha_frames):
            if not isinstance(f, str) or not f:
                raise MifOpError("alphaFrames[%d] must be a file path, got %r. NOTHING was written." % (i, f))
            alpha_paths.append(bpy.path.abspath(f))
        missing = [p for p in alpha_paths if not os.path.isfile(p)]
        if missing:
            raise MifOpError("%d of the %d alphaFrames do not exist, first %s - the transparent-film pass has not "
                             "finished, or a different naming. NOTHING was written."
                             % (len(missing), len(alpha_paths), missing[0]))
        paths_all = paths + alpha_paths
    else:
        paths_all = paths
    for label, target, raw in (("output", out_path, out_raw), ("normalOutput", n_path, n_raw),
                               ("motionOutput", m_path, m_raw)):
        if target is None:
            continue
        check_output_path(raw, target, "written")
        if not target.lower().endswith(".png"):
            raise MifOpError("%s must be a .png path - the sheet is written as an 8-bit RGBA PNG, got '%s'. "
                             "NOTHING was written." % (label, target))
        if os.path.isfile(target) and not overwrite:
            raise MifOpError("%s '%s' already exists - pass overwrite:true to replace it. NOTHING was written."
                             % (label, target))
        if os.path.normcase(target) in {os.path.normcase(p) for p in paths_all}:
            raise MifOpError("%s '%s' is also one of the frames - writing it would destroy an input. NOTHING was "
                             "written." % (label, target))
    if n_path and os.path.normcase(n_path) == os.path.normcase(out_path):
        raise MifOpError("normalOutput and output are the same file. NOTHING was written.")
    if m_path and os.path.normcase(m_path) in {os.path.normcase(out_path), os.path.normcase(n_path or "")}:
        raise MifOpError("motionOutput is the same file as output or normalOutput. NOTHING was written.")
    for target in [out_path] + ([n_path] if n_path else []) + ([m_path] if m_path else []):
        parent = os.path.dirname(target)
        try:
            check_writable_dir(parent or ".")
        except OSError as exc:
            raise MifOpError("cannot write in '%s' (%s). NOTHING was written." % (parent, exc))
    try:
        import numpy as np
    except ImportError:
        raise MifOpError("this Blender's Python has no numpy, which the tiling needs. NOTHING was written.")

    t0 = time.time()
    first = _load_frame(np, paths[0])
    ch, cw = int(first.shape[0]), int(first.shape[1])
    if cw * cols > _ATLAS_MAX_EDGE or ch * rows > _ATLAS_MAX_EDGE:
        raise MifOpError("the atlas would be %d x %d pixels; the limit is %d on either edge - render smaller "
                         "cells. NOTHING was written." % (cw * cols, ch * rows, _ATLAS_MAX_EDGE))
    # PREMULTIPLIED LINEAR, so a crossfade blends light rather than straight-alpha colour.
    pre = np.empty((want, ch, cw, 4), dtype=np.float32)
    for i, p in enumerate(paths):
        px = first if i == 0 else _load_frame(np, p)
        if px.shape[0] != ch or px.shape[1] != cw:
            raise MifOpError("frame %d is %dx%d and frame 0 is %dx%d - every cell must be the same size. NOTHING "
                             "was written." % (i, px.shape[1], px.shape[0], cw, ch))
        lin = _srgb_to_linear(np, px[..., :3])
        film = px[..., 3:4]
        if mode == "film":
            a = film
            rgb = lin * a
        elif mode == "hybrid":
            # the opaque-black pass is the premultiplied light; the transparent pass says where the smoke covers
            fp = _load_frame(np, alpha_paths[i])
            if fp.shape[0] != ch or fp.shape[1] != cw:
                raise MifOpError("alphaFrames[%d] is %dx%d and the frames are %dx%d - the two passes must be the "
                                 "same size. NOTHING was written." % (i, fp.shape[1], fp.shape[0], cw, ch))
            a = np.maximum(fp[..., 3:4], np.clip(np.max(lin, axis=2, keepdims=True) * gain, 0.0, 1.0))
            rgb = lin
        else:
            # rendered over black, so the colour already IS the premultiplied emission
            a = np.clip(np.max(lin, axis=2, keepdims=True) * gain, 0.0, 1.0)
            rgb = lin
        pre[i, ..., :3] = rgb
        pre[i, ..., 3:4] = a
    cells = pre[blend:].copy()
    for j in range(blend):
        w = float(j + 1) / float(blend + 1)
        k = cols * rows - blend + j
        cells[k] = (1.0 - w) * pre[blend + k] + w * pre[j]
    atlas = np.zeros((ch * rows, cw * cols, 4), dtype=np.float32)
    normals = np.zeros_like(atlas) if n_path else None
    coverage = []
    alphas = []
    for idx in range(cols * rows):
        r, c = divmod(idx, cols)
        cell = cells[idx]
        raw = np.clip(cell[..., 3:4], 0.0, 1.0)
        a = np.clip((raw - floor) / (1.0 - floor), 0.0, 1.0)
        # the colour against the ORIGINAL alpha (it is what the premultiplied light was divided by), and none at all
        # where the stored 8-bit alpha will be 0
        straight = np.where(a >= 0.5 / 255.0, cell[..., :3] / np.maximum(raw, 1e-4), 0.0)
        atlas[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw, :3] = _linear_to_srgb(np, straight)
        atlas[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw, 3:4] = a
        coverage.append(round(float(np.mean(a > 0.02)), 4))
        if m_path:
            alphas.append(a[..., 0].astype(np.float32))
        if normals is not None:
            h2 = a[..., 0]
            gx = np.zeros_like(h2)
            gy = np.zeros_like(h2)
            gx[:, 1:-1] = (h2[:, 2:] - h2[:, :-2]) * 0.5
            gy[1:-1, :] = (h2[2:, :] - h2[:-2, :]) * 0.5
            n = np.stack([-gx * n_strength * cw / 64.0, -gy * n_strength * ch / 64.0, np.ones_like(h2)], axis=2)
            n /= np.linalg.norm(n, axis=2, keepdims=True)
            normals[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw, :3] = n * 0.5 + 0.5
            normals[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw, 3] = 1.0
    motion = None
    if m_path:
        n_cells = cols * rows
        flows, residual = [], []
        for idx in range(n_cells):
            nxt = idx + 1 if idx + 1 < n_cells else (0 if blend > 0 else None)
            if nxt is None or (alphas[idx].max() < 0.02 and alphas[nxt].max() < 0.02):
                flows.append((np.zeros((ch, cw), np.float32), np.zeros((ch, cw), np.float32)))
                residual.append(None)
                continue
            fx, fy = _optical_flow(np, alphas[idx], alphas[nxt])
            # only where there is something to move: outside both frames' coverage the flow is noise
            mask = _blur(np, np.maximum(alphas[idx], alphas[nxt]), 1.5) > 0.02
            fx, fy = np.where(mask, fx, 0.0), np.where(mask, fy, 0.0)
            # no motion larger than a quarter of the cell - LK past that is a wrong match, not a motion
            lim = 0.25 * max(cw, ch)
            mag = np.sqrt(fx * fx + fy * fy)
            k = np.where(mag > lim, lim / np.maximum(mag, 1e-6), 1.0)
            fx, fy = (fx * k).astype(np.float32), (fy * k).astype(np.float32)
            flows.append((fx, fy))
            before = float(np.mean(np.abs(alphas[nxt] - alphas[idx])))
            after = float(np.mean(np.abs(_warp(np, alphas[nxt], fx, fy) - alphas[idx])))
            residual.append(round(after / before, 3) if before > 1e-6 else None)
        mags = np.concatenate([np.sqrt(fx * fx + fy * fy)[np.sqrt(fx * fx + fy * fy) > 0].ravel() for fx, fy in flows]
                              or [np.zeros(1, np.float32)])
        auto = m_scale is None
        if auto:
            m_scale = max(1.0, float(np.ceil(np.percentile(mags, 99)))) if mags.size else 1.0
        msheet = np.zeros_like(atlas)
        for idx, (fx, fy) in enumerate(flows):
            r, c = divmod(idx, cols)
            msheet[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw, 0] = np.clip(0.5 + fx / (2.0 * m_scale), 0.0, 1.0)
            msheet[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw, 1] = np.clip(0.5 + fy / (2.0 * m_scale), 0.0, 1.0)
            msheet[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw, 3] = 1.0
        motion = _write_sheet(np, msheet, m_path, is_data=True)
        meas = [x for x in residual if x is not None]
        motion.update({
            "motionScalePx": m_scale, "motionScaleAuto": auto,
            # what a material multiplies the decoded (RG x 2 - 1) by to move the ATLAS uv
            "motionStrengthUV": [round(m_scale / float(cw * cols), 6), round(m_scale / float(ch * rows), 6)],
            "maxMotionPx": round(float(mags.max()), 3) if mags.size else 0.0,
            "meanMotionPx": round(float(mags.mean()), 3) if mags.size else 0.0,
            "residual": residual,
            "meanResidual": round(sum(meas) / len(meas), 3) if meas else None,
            "lastCell": "toward the first cell (the sheet loops)" if blend > 0 else "0 - a one-shot does not wrap",
            "encoding": "RG = 0.5 + flow / (2 x motionScalePx), in pixels of the cell, green down (uv's v); B 0, A 1"})
    sheet = _write_sheet(np, atlas, out_path, is_data=False)
    normal_sheet = _write_sheet(np, normals, n_path, is_data=True) if n_path else None
    empty = [i for i, v in enumerate(coverage) if v < 0.001]
    # A CELL IDENTICAL TO THE ONE BEFORE IT is a frame the render did not advance. Measured on 5.0.1: EEVEE's
    # out-of-process animation render drew a baked fluid frozen at its first frame, 16 cells pixel-identical, while
    # the same file rendered in Cycles moved - a sheet that looks fine until it plays.
    repeated = [i for i in range(1, cols * rows) if np.array_equal(cells[i], cells[i - 1])]
    out = {
        "atlas": sheet["path"], "wroteFile": sheet["wroteFile"], "fileBytes": sheet["fileBytes"],
        "width": sheet["width"], "height": sheet["height"], "columns": cols, "rows": rows,
        "cellWidth": cw, "cellHeight": ch, "frames": cols * rows, "framesRead": want, "loopBlend": blend,
        "alphaMode": mode, "alphaGain": gain if mode != "film" else None, "alphaFloor": floor,
        "coverage": coverage, "emptyCells": empty, "repeatedCells": repeated,
        "meanCoverage": round(sum(coverage) / float(len(coverage)), 4),
        "frameOrder": "left-to-right, top-to-bottom - the order Unreal's SubUV reads a Sub Image Size grid",
        "elapsedSeconds": round(time.time() - t0, 3),
        "sizeNote": "wroteFile and fileBytes are stat'd off disk after the save, not taken from it.",
    }
    if alpha_paths:
        out["alphaFramesRead"] = len(alpha_paths)      # the transparent-film pass a hybrid sheet took its coverage from
    if normal_sheet:
        out["normalAtlas"] = normal_sheet
        out["normalNote"] = ("tangent-space normal derived from each cell's alpha gradient, green-down (DirectX, "
                             "Unreal's convention); import it with sRGB off and normal-map compression")
    if motion:
        out["motionAtlas"] = motion
        out["motionNote"] = ("optical flow of each cell toward the next; import it with sRGB off and no mipmaps, and "
                             "give a material motionStrengthUV (create_particle_material motion_flipbook)")
    if empty:
        out["emptyCellsNote"] = ("%d cell(s) hold nothing (coverage under 0.1%%) - an effect that had not started "
                                 "or had left the frame; a SubUV sprite shows a blank frame there" % len(empty))
    if cols * rows > 1 and len(repeated) == cols * rows - 1:
        out["frozenNote"] = ("every cell is the same image - the frames did not change, so the sheet will not "
                             "animate. For a simulation this is a render that did not advance it (an unbaked or "
                             "unread cache, or an engine that did not refresh the volume between frames)")
    elif repeated:
        out["repeatedCellsNote"] = ("%d cell(s) are pixel-identical to the cell before them - a held frame, or a "
                                    "render that did not advance there" % len(repeated))
    return out


_PREVIEW_KEYS = {"object", "name", "textures", "hdri", "hdriStrength", "views", "elevation", "resolution", "samples",
                 "outputDir", "prefix", "backfaceCulling", "filmTransparent"}
_PREVIEW_VIEWS = {"FRONT": 0.0, "RIGHT": 90.0, "BACK": 180.0, "LEFT": 270.0}
_PREVIEW_HDRIS = ("city", "courtyard", "forest", "interior", "night", "studio", "sunrise", "sunset")


def _preview_hdri(raw):
    """(path, label) for a bundled studio-light world name or an image file, else a refusal."""
    word = str(raw).strip()
    if word.lower() in _PREVIEW_HDRIS:
        base = bpy.utils.system_resource("DATAFILES", path="studiolights/world")
        path = os.path.join(base or "", word.lower() + ".exr")
        if not os.path.isfile(path):
            raise MifOpError("this Blender (%s) ships no %s.exr in %s. NOTHING was rendered."
                             % (bpy.app.version_string, word.lower(), base))
        return path, word.lower()
    if not os.path.isfile(word):
        raise MifOpError("hdri must be a bundled world (%s) or an image file; %r is neither. NOTHING was rendered."
                         % (", ".join(_PREVIEW_HDRIS), raw))
    return word, os.path.basename(word)


def _preview_views(raw):
    """[(label, azimuth degrees)] from names FRONT/RIGHT/BACK/LEFT or numbers (0 = from -Y, 90 = from +X)."""
    if raw is None:
        raw = ["FRONT", "BACK"]
    if not isinstance(raw, list) or not raw or len(raw) > 8:
        raise MifOpError("views must be a list of 1-8 entries, each FRONT, RIGHT, BACK, LEFT or an azimuth in degrees. "
                         "NOTHING was rendered.")
    out = []
    for v in raw:
        if isinstance(v, str) and v.upper() in _PREVIEW_VIEWS:
            out.append((v.upper(), _PREVIEW_VIEWS[v.upper()]))
        elif isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
            out.append(("AZ%03d" % (int(round(v)) % 360), float(v)))
        else:
            raise MifOpError("views entry %r is not FRONT, RIGHT, BACK, LEFT or an azimuth in degrees. NOTHING was "
                             "rendered." % (v,))
    azs = [round(az % 360.0, 3) for _, az in out]
    if len(set(azs)) != len(azs):
        raise MifOpError("views names one azimuth twice (%s; FRONT is 0, RIGHT 90, BACK 180, LEFT 270). NOTHING was "
                         "rendered." % ", ".join(str(v) for v in raw))
    return out


def _preview_texture(textures, key, existing):
    """An image for textures[key] - an image already in the file by name, or a file to load. (image, loaded?)."""
    raw = textures.get(key)
    if raw is None:
        return None, False
    if not isinstance(raw, str) or not raw:
        raise MifOpError("textures.%s must be an image name or a file path. NOTHING was rendered." % key)
    if raw in existing:
        return bpy.data.images[raw], False
    if not os.path.isfile(raw):
        raise MifOpError("textures.%s %r is neither an image in this file nor a file on disk. NOTHING was rendered."
                         % (key, raw))
    return None, True


def _preview_material(name, imgs, uv_name):
    """Principled with base color x ORM red (AO), ORM green to roughness, ORM blue to metallic, normal through a
    Normal Map. Returns (material, wiring rows). The images' color spaces are the caller's business."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    wiring = []

    def tex(img, y):
        n = nt.nodes.new("ShaderNodeTexImage")
        n.image = img
        n.location = (-700, y)
        if uv_name:
            uv = nt.nodes.new("ShaderNodeUVMap")
            uv.uv_map = uv_name
            uv.location = (-900, y)
            nt.links.new(uv.outputs[0], n.inputs[0])
        return n

    bc = tex(imgs["baseColor"], 300) if imgs.get("baseColor") else None
    orm = tex(imgs["orm"], 0) if imgs.get("orm") else None
    sep = None
    if orm is not None:
        sep = nt.nodes.new("ShaderNodeSeparateColor")
        sep.location = (-450, 0)
        nt.links.new(orm.outputs["Color"], sep.inputs[0])
        nt.links.new(sep.outputs["Green"], bsdf.inputs["Roughness"])
        nt.links.new(sep.outputs["Blue"], bsdf.inputs["Metallic"])
        wiring += ["orm.G -> Roughness", "orm.B -> Metallic"]
    if bc is not None and sep is not None:
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        mix.blend_type = "MULTIPLY"
        mix.location = (-250, 300)
        mix.inputs["Factor"].default_value = 1.0
        a = next(s for s in mix.inputs if s.identifier == "A_Color")
        b = next(s for s in mix.inputs if s.identifier == "B_Color")
        nt.links.new(bc.outputs["Color"], a)
        nt.links.new(sep.outputs["Red"], b)
        nt.links.new(next(s for s in mix.outputs if s.identifier == "Result_Color"), bsdf.inputs["Base Color"])
        wiring.append("baseColor x orm.R (AO) -> Base Color")
    elif bc is not None:
        nt.links.new(bc.outputs["Color"], bsdf.inputs["Base Color"])
        wiring.append("baseColor -> Base Color")
    if imgs.get("normal"):
        nrm = tex(imgs["normal"], -300)
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nm.location = (-250, -300)
        if uv_name:
            nm.uv_map = uv_name
        nt.links.new(nrm.outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
        wiring.append("normal -> Normal Map -> Normal")
    return mat, wiring


def op_preview_garment(params):
    """Render a garment alone the way an asset viewer lights it: EEVEE, an HDRI world, backface culling, its textures.

    WHY (2026-10-06). Judging a texture change on a thumbnail with no sky light shipped a wrong hood lift. The check
    that worked was EEVEE, an HDRI world, backface culling, the garment alone, and a material that multiplies base
    color by the ORM's AO and wires ORM blue to metallic. Cycles is not a substitute: it ignores backface culling,
    so a garment's inside shows through where a game engine culls it.

    NOTHING IN THE FILE CHANGES. The garment is copied AS IT EVALUATES NOW (modifiers, shape keys, pose - and a cloth
    modifier's current state, without stepping any simulation) into a temporary scene with only a camera and the HDRI
    world; every datablock made for the preview is removed afterwards and leftBehind counts what was not.

    params:
      object (str, required)        the garment (a mesh, or anything that evaluates to one)
      textures {baseColor, normal, orm}  image names in the file or files; orm is R = AO, G = roughness, B = metallic.
                                    Without textures the object's own materials are rendered, copied, culling on.
      hdri (str)                    a bundled world - city, courtyard, forest, interior, night, studio (default),
                                    sunrise, sunset - or an image file; hdriStrength (default 1)
      views [..]                    FRONT, RIGHT, BACK, LEFT or azimuths in degrees (0 looks from -Y, 90 from +X);
                                    default [FRONT, BACK]; elevation degrees (default 10)
      resolution (int)              square, 64-2048, default 512; samples (EEVEE TAA, default 16, 1-256)
      outputDir (str, required), prefix (default the object name): files <prefix>_<view>.png
      backfaceCulling (default true), filmTransparent (default true; coverage is the alpha share)
    """
    reject_unknown(params, _PREVIEW_KEYS, "preview_garment", {
        "engine": "this preview is EEVEE by design - Cycles ignores backface culling, which is the point of it.",
        "camera": "the camera frames the object from each view; set views and elevation.",
    })
    tail = "NOTHING was rendered."
    src = get_object(take(params, "object", "name", required=True))
    if src.type not in ("MESH", "CURVE", "SURFACE", "META", "FONT", "CURVES"):
        raise MifOpError("'%s' is a %s and does not evaluate to a mesh. %s" % (src.name, src.type, tail))
    textures = params.get("textures")
    if textures is not None and (not isinstance(textures, dict) or not textures
                                 or set(textures) - {"baseColor", "normal", "orm"}):
        raise MifOpError("textures is {baseColor, normal, orm} (image names or files), got %r. %s" % (textures, tail))
    textures = textures or {}
    if textures and "baseColor" not in textures:
        raise MifOpError("textures needs baseColor - the material is built from it. %s" % tail)
    hdri_path, hdri_label = _preview_hdri(take(params, "hdri", default="studio", kind=str))
    strength = take_float(params, "hdriStrength", default=1.0)
    if not (math.isfinite(strength) and 0 < strength <= 100):
        raise MifOpError("hdriStrength must be above 0 and at most 100. %s" % tail)
    views = _preview_views(params.get("views"))
    elev = take_float(params, "elevation", default=10.0)
    if not (math.isfinite(elev) and -89 <= elev <= 89):
        raise MifOpError("elevation must be between -89 and 89 degrees. %s" % tail)
    res = take_int(params, "resolution", default=512)
    samples = take_int(params, "samples", default=16)
    if not 64 <= res <= 2048 or not 1 <= samples <= 256:
        raise MifOpError("resolution must be 64-2048 and samples 1-256. %s" % tail)
    outdir = take(params, "outputDir", default=None, kind=str)
    if not outdir:
        raise MifOpError("outputDir is required - the views are written there as <prefix>_<view>.png. %s" % tail)
    prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", take(params, "prefix", default=src.name, kind=str)) or "garment"
    cull = take_bool(params, "backfaceCulling", default=True)
    transparent = take_bool(params, "filmTransparent", default=True)
    existing_imgs = set(bpy.data.images.keys())
    imgs, to_load = {}, {}
    for key in ("baseColor", "normal", "orm"):
        img, load = _preview_texture(textures, key, existing_imgs)
        if img is not None:
            imgs[key] = img
        elif load:
            to_load[key] = textures[key]
    outdir = os.path.abspath(bpy.path.abspath(outdir))
    for label, _ in views:
        check_output_path(outdir, os.path.join(outdir, "%s_%s.png" % (prefix, label)), "rendered")
    # _engine_ids() assigns a bogus engine to read Blender's own list and puts the engine back - after every check
    # that can refuse.
    eevee = sorted(e for e in _engine_ids() if "EEVEE" in e)
    if not eevee:
        raise MifOpError("this Blender offers no EEVEE engine (%s). %s" % (sorted(_engine_ids()), tail))
    try:
        check_writable_dir(outdir)
    except OSError as exc:
        raise MifOpError("cannot write to outputDir %s: %s. %s" % (outdir, exc, tail))

    counts_before = {k: len(getattr(bpy.data, k)) for k in ("objects", "meshes", "materials", "images", "worlds",
                                                             "cameras", "scenes")}
    made = {"objects": [], "meshes": [], "materials": [], "images": [], "worlds": [], "cameras": [], "scenes": []}
    rows, warnings = [], []
    wiring = tex_info = view_transform = None
    t0 = time.perf_counter()
    try:
        for key, path in to_load.items():
            img = bpy.data.images.load(path, check_existing=False)
            made["images"].append(img)
            if key != "baseColor":
                try:
                    img.colorspace_settings.name = "Non-Color"
                except TypeError:
                    pass
            imgs[key] = img
        # AN IMAGE ALREADY IN THE FILE KEEPS ITS COLOUR SPACE: setting it on a generated or painted image reloads the
        # buffer and throws the pixels away (measured 2026-10-06 on 5.2.1: a painted ORM lost its AO half). A data map
        # read as sRGB is said instead.
        for key in ("normal", "orm"):
            img = imgs.get(key)
            if img is not None and img not in made["images"] and img.colorspace_settings.name != "Non-Color":
                warnings.append("textures.%s '%s' is in color space %s, so its values are read as color, not data; "
                                "set it to Non-Color to judge AO, roughness and metallic as the engine will."
                                % (key, img.name, img.colorspace_settings.name))
        tex_info = {k: (v.filepath or v.name) for k, v in imgs.items()} if imgs else None
        deps = bpy.context.evaluated_depsgraph_get()
        mesh = bpy.data.meshes.new_from_object(src.evaluated_get(deps), preserve_all_data_layers=True,
                                               depsgraph=deps)
        made["meshes"].append(mesh)
        mesh.transform(src.matrix_world)
        scene = bpy.data.scenes.new("MifGarmentPreview")
        made["scenes"].append(scene)
        obj = bpy.data.objects.new(prefix + "_preview", mesh)
        made["objects"].append(obj)
        scene.collection.objects.link(obj)
        if imgs:
            uv = mesh.uv_layers.active.name if mesh.uv_layers.active else None
            mat, wiring = _preview_material("MifGarmentPreview", imgs, uv)
            made["materials"].append(mat)
            mesh.materials.clear()
            mesh.materials.append(mat)
            for p in mesh.polygons:
                p.material_index = 0
        else:
            for i, m in enumerate(mesh.materials):
                if m is not None:
                    cp = m.copy()
                    made["materials"].append(cp)
                    mesh.materials[i] = cp
        for m in made["materials"]:
            m.use_backface_culling = cull
        world = bpy.data.worlds.new("MifGarmentPreview")
        made["worlds"].append(world)
        world.use_nodes = True
        wn = world.node_tree
        bg = next(n for n in wn.nodes if n.type == "BACKGROUND")
        env = wn.nodes.new("ShaderNodeTexEnvironment")
        env_img = bpy.data.images.load(hdri_path, check_existing=False)
        made["images"].append(env_img)
        env.image = env_img
        wn.links.new(env.outputs["Color"], bg.inputs["Color"])
        bg.inputs["Strength"].default_value = strength
        scene.world = world
        r = scene.render
        r.engine = eevee[0]
        r.resolution_x = r.resolution_y = res
        r.resolution_percentage = 100
        r.film_transparent = transparent
        r.image_settings.file_format = "PNG"
        r.image_settings.color_mode = "RGBA"
        scene.eevee.taa_render_samples = samples
        view_transform = scene.view_settings.view_transform
        cam_data = bpy.data.cameras.new("MifGarmentPreview")
        made["cameras"].append(cam_data)
        cam = bpy.data.objects.new("MifGarmentPreviewCam", cam_data)
        made["objects"].append(cam)
        scene.collection.objects.link(cam)
        scene.camera = cam
        pts = [Vector(v.co) for v in mesh.vertices]
        if not pts:
            raise MifOpError("'%s' evaluates to a mesh with no vertices. %s" % (src.name, tail))
        lo = Vector([min(p[i] for p in pts) for i in range(3)])
        hi = Vector([max(p[i] for p in pts) for i in range(3)])
        center, radius = (lo + hi) / 2, max((hi - lo).length / 2, 1e-3)
        dist = radius / math.sin(cam_data.angle / 2) * 1.05
        cam_data.clip_start = max(dist - radius * 2, 1e-3) * 0.5
        cam_data.clip_end = dist + radius * 2
        e = math.radians(elev)
        for label, az in views:
            a = math.radians(az)
            pos = center + Vector((math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e))) * dist
            cam.location = pos
            cam.rotation_euler = (center - pos).to_track_quat("-Z", "Y").to_euler()
            path = os.path.join(outdir, "%s_%s.png" % (prefix, label))
            r.filepath = path
            bpy.ops.render.render(write_still=True, scene=scene.name)
            if not os.path.isfile(path):
                raise MifOpError("the %s view rendered no file at %s." % (label, path))
            shot = bpy.data.images.load(path, check_existing=False)
            try:
                px = list(shot.pixels)
            finally:
                bpy.data.images.remove(shot)
            n = len(px) // 4
            cov = [i for i in range(n) if px[i * 4 + 3] > 0.5]
            lum = (sum(0.2126 * px[i * 4] + 0.7152 * px[i * 4 + 1] + 0.0722 * px[i * 4 + 2] for i in cov) / len(cov)
                   if cov else None)
            rows.append({"view": label, "azimuth": az, "file": path, "coverage": round(len(cov) / float(n), 5),
                         "meanLuminance": round(lum, 5) if lum is not None else None})
    finally:
        for kind in ("scenes", "objects", "cameras", "meshes", "materials", "worlds", "images"):
            coll = getattr(bpy.data, kind)
            for idb in made[kind]:
                try:
                    coll.remove(idb)
                except (ReferenceError, RuntimeError):
                    pass
    left = {k: len(getattr(bpy.data, k)) - v for k, v in counts_before.items() if len(getattr(bpy.data, k)) != v}
    return {"object": src.name, "engine": eevee[0], "hdri": hdri_label, "hdriStrength": strength,
            "backfaceCulling": cull, "material": wiring if wiring is not None else "the object's own materials, copied",
            "textures": tex_info,
            "views": rows, "viewTransform": view_transform, "resolution": res, "samples": samples, "seconds": round(time.perf_counter() - t0, 3),
            "leftBehind": left, "warnings": warnings}


_STEREO_KEYS = ("enabled", "viewsFormat", "output", "displayMode", "camera", "convergenceMode", "interocularDistance",
                "convergenceDistance", "pivot", "views")


def _enum_of(holder, prop):
    return [i.identifier for i in holder.bl_rna.properties[prop].enum_items]


def _stereo_state(sc, cam):
    r = sc.render
    im = r.image_settings
    out = {"enabled": bool(r.use_multiview), "viewsFormat": r.views_format,
           "views": [{"name": v.name, "use": bool(v.use), "suffix": v.file_suffix} for v in r.views],
           "output": im.views_format, "displayMode": im.stereo_3d_format.display_mode}
    if cam is not None:
        st = cam.data.stereo
        out["camera"] = {"name": cam.name, "convergenceMode": st.convergence_mode,
                         "interocularDistance": round(st.interocular_distance, 6),
                         "convergenceDistance": round(st.convergence_distance, 6), "pivot": st.pivot}
    return out


def op_set_stereo(params):
    """Stereoscopic and multiview rendering: on or off, the views format, how the views are written, and the camera's
    stereo convergence, interocular distance and pivot - read back.

    WHY (second 1.3 audit, 2026-10-25): a VR 180 / 360 deliverable needs two eyes, and set_camera_panorama made only the
    equirectangular camera. RenderSettings.use_multiview / views_format and Camera.stereo (5.2
    bl_ui/properties_output.py:340, :687) are plain data, so nothing here needs an editor.

    params:
      enabled (bool)              RenderSettings.use_multiview
      viewsFormat                 STEREO_3D (left and right) or MULTIVIEW (the views list)
      views {name: bool}          which render views are on (MULTIVIEW; STEREO_3D uses left and right)
      output                      INDIVIDUAL (a file per view) or STEREO_3D (one file, displayMode)
      displayMode                 ANAGLYPH, INTERLACE, SIDEBYSIDE or TOPBOTTOM - this build's list
      camera (str)                the camera whose stereo is set (default the scene camera)
      convergenceMode             OFFAXIS, PARALLEL or TOE; interocularDistance (m), convergenceDistance (m)
      pivot                       LEFT, RIGHT or CENTER
    With nothing to set it reads. Every enum is checked against this build's own list before anything is written.
    """
    reject_unknown(params, _STEREO_KEYS, "set_stereo")
    tail = "NOTHING was changed."
    sc = bpy.context.scene
    cam_name = take(params, "camera", kind=str)
    cam = get_object(cam_name) if cam_name else sc.camera
    if cam is not None and cam.type != "CAMERA":
        raise MifOpError("'%s' is a %s, not a camera. %s" % (cam.name, cam.type, tail))
    r = sc.render
    im = r.image_settings
    plan = []
    for key, holder, prop in (("viewsFormat", r, "views_format"), ("output", im, "views_format"),
                              ("displayMode", im.stereo_3d_format, "display_mode")):
        if params.get(key) is not None:
            val = str(params[key]).upper()
            ok = _enum_of(holder, prop)
            if val not in ok:
                raise MifOpError("%s is one of %s on this build, got %r. %s" % (key, ", ".join(ok), params[key], tail))
            plan.append((holder, prop, val))
    cam_keys = [k for k in ("convergenceMode", "interocularDistance", "convergenceDistance", "pivot")
                if params.get(k) is not None]
    if cam_keys and cam is None:
        raise MifOpError("%s set a camera's stereo, and the scene has no camera - pass camera. %s"
                         % (", ".join(cam_keys), tail))
    if cam is not None:
        st = cam.data.stereo
        for key, prop in (("convergenceMode", "convergence_mode"), ("pivot", "pivot")):
            if params.get(key) is not None:
                val = str(params[key]).upper()
                ok = _enum_of(st, prop)
                if val not in ok:
                    raise MifOpError("%s is one of %s, got %r. %s" % (key, ", ".join(ok), params[key], tail))
                plan.append((st, prop, val))
        for key, prop in (("interocularDistance", "interocular_distance"), ("convergenceDistance", "convergence_distance")):
            if params.get(key) is not None:
                v = finite_float(params[key], key)
                if v < 0:
                    raise MifOpError("%s is 0 or more metres, got %g. %s" % (key, v, tail))
                plan.append((st, prop, v))
    views = params.get("views")
    if views is not None:
        if not isinstance(views, dict) or not views or not all(isinstance(v, bool) for v in views.values()):
            raise MifOpError("views is {view name: true | false}. %s" % tail)
        missing = [n for n in views if r.views.get(n) is None]
        if missing:
            raise MifOpError("no render view %s (the scene has %s). %s"
                             % (missing, [v.name for v in r.views], tail))
    enabled = take_bool(params, "enabled") if params.get("enabled") is not None else None
    before = _stereo_state(sc, cam)
    if enabled is not None:
        r.use_multiview = enabled
    for holder, prop, val in plan:
        setattr(holder, prop, val)
    for n, use in (views or {}).items():
        r.views[n].use = use
    after = _stereo_state(sc, cam)
    out = {"before": before, "after": after, "wrote": len(plan) > 0 or views is not None or enabled is not None}
    if after["enabled"] and after["viewsFormat"] == "STEREO_3D" and not all(
            v["use"] for v in after["views"] if v["name"] in ("left", "right")):
        out["warning"] = "STEREO_3D renders the left and right views, and one of them is off"
    return out


OPS = {
    "set_stereo": op_set_stereo,
    "preview_garment": op_preview_garment,
    "set_render_settings": op_set_render_settings,
    "render_still": op_render_still,
    "compare_to_reference": op_compare_to_reference,
    "render_info": op_render_info,
    "render_animation": op_render_animation,
    "render_status": op_render_status,
    "set_color_management": op_set_color_management,
    "compose_atlas": op_compose_atlas,
}
