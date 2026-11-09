"""MifBlender ops: image datablocks as a canvas - make a blank one, write one to disk, derive a map from one.

THE GAP (spec, "Blender: images, texture density and painting" and "materials, skin, eyes and textures", 2026-10-04):
bpy.data.images.new appeared twice in the add-on and neither was a call a user could make - bake_texture's own bake image
and the atlas writer's temporary sheet. So a painting canvas, a makeup layer at alpha 0, a white roughness layer and
the save after every paint session had no typed call, and nothing turned one colour image into the height, normal and
specular maps an eye or a skin needed.

HOW AN IMAGE REACHES THE DISK, MEASURED 2026-10-04 on 3.6.23, 4.4.0 and 5.2.1 (a 4x4 image of [0.5, 0.25, 0.75, 1],
written and loaded back):
  * Image.save() writes the image's OWN buffer in its file_format: an 8-bit PNG from a byte image (0.502 back) and a
    16-bit PNG from a float image (0.5 exact); JPEG takes `quality` (a 64 px noise image: 1,065 / 2,119 / 5,108 bytes
    at quality 10 / 50 / 95); OPEN_EXR is always FULL float (pixel type 2), use_half_precision or not.
  * save(filepath=X) writes the file and leaves the datablock's filepath EMPTY with its source FILE - an image that
    cannot reload. Setting filepath_raw first and calling save() re-points it: source FILE, the path set, not dirty, and
    a reload gives the same pixels. So `repoint` sets the path first, and repoint false writes from a copy.
  * Image.save_render() goes through the scene's colour management: a BYTE sRGB image saved as a 16-bit PNG through a
    Standard view came back 0.216 for 0.5 (converted twice), and the default view (Filmic on 3.6, AgX from 4.0) changed
    every value. A FLOAT Non-Color image came back exact as a 16-bit PNG and as a half (pixel type 1) or full EXR. A
    float sRGB image saved as EXR stayed 0.5 on 3.6.23 and became 0.214 on 5.2.1. So save_render is used for ONE case
    only - a half-float EXR of a float data image - and everything else is Image.save; the combinations neither route
    writes exactly are refused with that reason.
  * A datablock with no users and no fake user is deleted by the next save of the .blend, which is why create_image
    gives its canvas a fake user by default.
"""
from __future__ import annotations

import math
import os

import bpy

from .ops_common import (abs_output_path, remove_objects, MifOpError, assign_id_name, check_output_path, file_before, finite_floats, finite_int,
                         reject_unknown, replaced_fields, take, take_bool, take_float, take_int)

# formats a still image can be written in on this build: the Image.file_format enum without the movie containers
_MOVIE_FORMATS = {"AVI_JPEG", "AVI_RAW", "FFMPEG"}


def _image_formats():
    return [i.identifier for i in bpy.types.Image.bl_rna.properties["file_format"].enum_items
            if i.identifier not in _MOVIE_FORMATS]


def _get_image(name, tail="NOTHING was changed."):
    img = bpy.data.images.get(str(name)) if name else None
    if img is None:
        have = sorted(i.name for i in bpy.data.images if i.source not in ("VIEWER",))[:25]
        raise MifOpError("no image named '%s'. This file has: %s. %s" % (name, ", ".join(have) or "(none)", tail))
    if img.source == "VIEWER":
        raise MifOpError("'%s' is a viewer image (the render result or the compositor's viewer) - it has no pixels of "
                         "its own to work on. %s" % (img.name, tail))
    return img


def _colorspace_or_refuse(img, name, tail):
    try:
        img.colorspace_settings.name = name
    except TypeError:
        valid = [i.identifier for i in img.colorspace_settings.bl_rna.properties["name"].enum_items]
        raise MifOpError("'%s' is not a colour space on this Blender. Available: %s. %s"
                         % (name, ", ".join(valid), tail))


# --------------------------------------------------------------------------------------------- create_image

_CREATE_KEYS = ("name", "width", "height", "color", "alpha", "float", "generatedType", "colorspace", "fakeUser",
                "material", "input")
_CREATE_NOTES = {
    "size": "width and height, in pixels.",
    "file": "create_image makes a canvas in memory; save_image writes it to disk, and set_material_texture loads a file.",
    "fill": "the key is 'color': [r, g, b] or [r, g, b, a], 0-1.",
}


def op_create_image(params):
    """Make a blank image datablock - a canvas to paint on or bake into, filled with one colour.

    params:
      name (str)               required
      width, height (int)      pixels (Blender's own limits, read off images.new)
      color [r, g, b(, a)]     the fill, 0-1 for an 8-bit image (default black, alpha 1)
      alpha (bool)             an alpha channel (default false: an opaque base colour needs none)
      float (bool)             a 32-bit float buffer (default false) - what a 16-bit PNG or an EXR needs
      generatedType (str)      BLANK (default) | UV_GRID | COLOR_GRID, read from the build's own list
      colorspace (str)         default sRGB; Non-Color for a data map (roughness, height, a mask)
      fakeUser (bool)          default true: an image nothing uses is DELETED by the next save of the .blend
      material + input         wire it into that material (set_material_texture's route); refused naming the inputs
    The canvas is not on disk until save_image writes it.
    """
    reject_unknown(params, _CREATE_KEYS, "create_image", notes=_CREATE_NOTES)
    tail = "NOTHING was created."
    name = take(params, "name", required=True, kind=str)
    fn = bpy.data.images.bl_rna.functions["new"]
    lims = {p.identifier: (p.hard_min, p.hard_max) for p in fn.parameters if p.identifier in ("width", "height")}
    size = []
    for key in ("width", "height"):
        val = take_int(params, key, required=True)
        lo, hi = lims.get(key, (1, 65536))
        hi = min(hi, 16384)
        if not max(lo, 1) <= val <= hi:
            raise MifOpError("%s %d is outside 1-%d. %s" % (key, val, hi, tail))
        size.append(val)
    want_float = take_bool(params, "float", default=False)
    want_alpha = take_bool(params, "alpha", default=False)
    color = params.get("color")
    if color is not None:
        if not isinstance(color, (list, tuple)) or len(color) not in (3, 4) or any(isinstance(c, bool) for c in color):
            raise MifOpError("color takes [r, g, b] or [r, g, b, a], got %r. %s" % (color, tail))
        color = finite_floats(list(color), "color")
        if len(color) == 3:
            color.append(1.0)
        if not want_float and any(not 0.0 <= c <= 1.0 for c in color):
            raise MifOpError("color %r has a value outside 0-1, which an 8-bit image clamps - pass float: true for "
                             "values above 1. %s" % (color, tail))
        if color[3] < 1.0 and not want_alpha:
            raise MifOpError("color has alpha %s and alpha is off, so the image would store no transparency - pass "
                             "alpha: true for a layer that starts transparent. %s" % (color[3], tail))
    gtypes = [i.identifier for i in bpy.types.Image.bl_rna.properties["generated_type"].enum_items]
    gtype = str(take(params, "generatedType", default="BLANK", kind=str)).upper()
    if gtype not in gtypes:
        raise MifOpError("generatedType '%s' is not one this Blender has. Valid: %s. %s" % (gtype, ", ".join(gtypes),
                                                                                           tail))
    if gtype != "BLANK" and color is not None:
        raise MifOpError("color fills a BLANK image; %s draws its own pattern. %s" % (gtype, tail))
    cs = take(params, "colorspace", default=None, kind=str)
    fake = take_bool(params, "fakeUser", default=True)
    mat_name = take(params, "material", default=None, kind=str)
    which = take(params, "input", default=None, kind=str)
    if (mat_name is None) != (which is None):
        raise MifOpError("material and input go together - which material, and which of its inputs. %s" % tail)
    if mat_name is not None and bpy.data.materials.get(mat_name) is None:
        have = sorted(m.name for m in bpy.data.materials)[:25]
        raise MifOpError("no material named '%s'. Present: %s. %s" % (mat_name, ", ".join(have) or "(none)", tail))

    img = bpy.data.images.new(str(name), size[0], size[1], alpha=want_alpha, float_buffer=want_float)
    try:
        img.generated_type = gtype
        if color is not None:
            img.generated_color = color
        if cs:
            _colorspace_or_refuse(img, str(cs), tail)
        img.use_fake_user = fake
        wired = None
        if mat_name is not None:
            from .ops_material import op_set_material_texture   # noqa: PLC0415 - the one wiring route
            wired = op_set_material_texture({"material": mat_name, "image": img.name, "input": which})
    except Exception:
        bpy.data.images.remove(img)
        raise
    return {
        "image": img.name,
        "requestedName": name,
        "nameWasSuffixed": img.name != name,
        "size": list(img.size),
        "float": bool(img.is_float),
        "alpha": bool(want_alpha),
        "generatedType": img.generated_type,
        "color": [round(c, 6) for c in img.generated_color],
        "colorSpace": img.colorspace_settings.name,
        "fakeUser": bool(img.use_fake_user),
        "users": img.users,
        "source": img.source,
        "wired": wired,
        "note": ("in memory only: save_image writes it to disk and points the datablock at the file. Until then "
                 "%s." % ("it is kept by its fake user, and any painting on it lives only in this session"
                          if fake else "an image nothing uses is deleted by the next save of the .blend")),
    }


# --------------------------------------------------------------------------------------------- save_image

_SAVE_KEYS = ("image", "name", "filepath", "format", "colorDepth", "quality", "overwrite", "repoint")
_EXT = {"PNG": ".png", "JPEG": ".jpg", "OPEN_EXR": ".exr", "OPEN_EXR_MULTILAYER": ".exr", "TARGA": ".tga",
        "TARGA_RAW": ".tga", "TIFF": ".tif", "BMP": ".bmp", "WEBP": ".webp", "HDR": ".hdr", "JPEG2000": ".jp2",
        "AVIF": ".avif"}


def _format_from_path(path):
    ext = os.path.splitext(path)[1].lower()
    for fmt, e in _EXT.items():
        if e == ext and fmt in _image_formats():
            return fmt
    return None


def op_save_image(params):
    """Write an image datablock to disk and, by default, point it at the file - a painted canvas survives a reopen.

    params:
      image / name (str)     required
      filepath (str)         required; '//' is relative to the .blend
      format (str)           the build's own still formats (PNG, JPEG, OPEN_EXR, TARGA, ...); default from the
                             extension, else PNG
      colorDepth (int)       8 | 16 for PNG (16 needs a float image), 16 | 32 for OPEN_EXR (16, half float, for a float
                             Non-Color image only); default the image's own
      quality (int)          0-100, JPEG only
      overwrite (bool)       default true; false refuses a path that exists. A replaced file is reported.
      repoint (bool)         default true: the datablock now reads the file (source FILE, not dirty); false writes a
                             copy and leaves it as it was
    """
    reject_unknown(params, _SAVE_KEYS, "save_image")
    tail = "NOTHING was written."
    img = _get_image(take(params, "image", "name", required=True, kind=str), tail)
    raw = take(params, "filepath", required=True, kind=str)
    resolved = os.path.abspath(bpy.path.abspath(str(raw)))
    check_output_path(raw, resolved, "written")
    formats = _image_formats()
    fmt = take(params, "format", default=None, kind=str)
    if fmt is not None:
        fmt = str(fmt).upper()
        if fmt not in formats:
            raise MifOpError("format '%s' is not a still-image format on Blender %s. Available: %s. %s"
                             % (fmt, bpy.app.version_string, ", ".join(formats), tail))
    else:
        fmt = _format_from_path(resolved) or "PNG"
    if img.size[0] == 0 or img.size[1] == 0:
        raise MifOpError("'%s' has no pixels (0x0) - nothing to write. %s" % (img.name, tail))
    depth = take_int(params, "colorDepth", default=None)
    quality = take_int(params, "quality", default=None)
    overwrite = take_bool(params, "overwrite", default=True)
    repoint = take_bool(params, "repoint", default=True)
    if quality is not None:
        if fmt != "JPEG":
            raise MifOpError("quality is JPEG's; %s ignores it. %s" % (fmt, tail))
        if not 0 <= quality <= 100:
            raise MifOpError("quality %d is outside 0-100. %s" % (quality, tail))
    is_float = bool(img.is_float)
    data = img.colorspace_settings.name == "Non-Color" or getattr(img.colorspace_settings, "is_data", False)
    route = "save"
    if fmt in ("OPEN_EXR", "OPEN_EXR_MULTILAYER"):
        if not is_float:
            raise MifOpError("'%s' is an 8-bit image and %s stores floats - the values would be converted, not kept. "
                             "Make the canvas with create_image float: true, or save it as PNG. %s"
                             % (img.name, fmt, tail))
        if depth is None:
            depth = 32
        if depth not in (16, 32):
            raise MifOpError("an EXR is 16 (half) or 32 (full) bits per channel, got %r. %s" % (depth, tail))
        if depth == 16:
            if not data:
                raise MifOpError("a half-float EXR is written through the scene's colour management, and for a colour "
                                 "(%s) image that changes the values on some builds (measured: 0.5 stays 0.5 on 3.6.23 "
                                 "and becomes 0.214 on 5.2.1). Only a Non-Color data image is written exactly - set "
                                 "its colour space, or use colorDepth 32. %s" % (img.colorspace_settings.name, tail))
            route = "save_render"
    elif fmt == "PNG":
        own = 16 if is_float else 8
        if depth is None:
            depth = own
        if depth != own:
            raise MifOpError("a PNG is written at the image's own depth - %d bits for this %s image - so colorDepth %d "
                             "cannot be honoured: %s. %s"
                             % (own, "float" if is_float else "8-bit", depth,
                                "create the canvas with float: true for 16 bits" if depth == 16 else
                                "an 8-bit PNG of a float image would round every value", tail))
    elif depth is not None:
        raise MifOpError("colorDepth applies to PNG and OPEN_EXR; %s is written at its own depth. %s" % (fmt, tail))
    before = file_before(resolved)
    if before is not None and not overwrite:
        raise MifOpError("'%s' exists and overwrite is false. %s" % (resolved, tail))
    folder = os.path.dirname(resolved)
    if folder and not os.path.isdir(folder):
        raise MifOpError("the folder '%s' does not exist. %s" % (folder, tail))

    # A COPY IS NOT A WAY TO SAVE WITHOUT RE-POINTING. Measured 2026-10-04 on 5.2.1: Image.copy() of a generated image
    # ignored the format and wrote a PNG into a .exr path, and a copy of a FILE image had no pixels ("does not have any
    # image data"). save(save_copy=True) leaves the datablock as it was (4.4.0 and 5.0.1 have it; 3.6.23 and 4.2.17 LTS
    # do not, measured 2026-10-10 - hence the RNA read below, not a version); without save_copy its
    # save(filepath=) turns a generated image into a FILE image with no path - putting the source back regenerates it
    # from its fill colour and the painting is gone. So repoint false is refused there for a generated image.
    has_copy = "save_copy" in [p.identifier for p in bpy.types.Image.bl_rna.functions["save"].parameters]
    if not repoint and route == "save" and img.source == "GENERATED" and not has_copy:
        raise MifOpError("repoint false needs Image.save(save_copy), which Blender %s does not have: saving a generated "
                         "image here turns it into a file image with no path, and turning it back regenerates it from "
                         "its fill colour (the painting is lost). Save with repoint true. %s"
                         % (bpy.app.version_string, tail))
    state = (img.filepath_raw, img.file_format, img.source)
    try:
        if route == "save":
            # THE FORMAT ONLY STICKS ON A BUFFER THAT EXISTS. Measured 2026-10-04 on 3.6.23, 4.4.0 and 5.2.1: after a
            # generated image's colour space changes, file_format reads back PNG whatever is written until the buffer
            # is made again - and the file came out a PNG named .exr. update() makes it; the format is then read back.
            img.update()
            img.file_format = fmt
            if img.file_format != fmt:
                raise TypeError("the image kept file_format %s after %s was written" % (img.file_format, fmt))
            kwargs = {"quality": quality} if quality is not None else {}
            if repoint:
                img.filepath_raw = resolved
                img.save(**kwargs)
            elif has_copy:
                img.save(filepath=resolved, save_copy=True, **kwargs)
                img.file_format = state[1]
            else:
                # a FILE image on 3.6: save(filepath=) writes there and keeps its own path (measured)
                img.save(filepath=resolved, **kwargs)
                img.file_format = state[1]
        else:
            sc = bpy.data.scenes.new("MifSaveImage")
            try:
                ims = sc.render.image_settings
                ims.file_format = fmt
                ims.color_depth = str(depth)
                ims.color_mode = "RGBA" if img.depth in (32, 128) else "RGB"
                sc.view_settings.view_transform = "Standard"
                sc.view_settings.look = "None"
                img.save_render(filepath=resolved, scene=sc)
            finally:
                bpy.data.scenes.remove(sc)
            if repoint:
                img.filepath_raw = resolved
                img.file_format = fmt
                img.source = "FILE"
    except (RuntimeError, TypeError) as exc:
        img.filepath_raw = state[0]
        try:
            img.file_format = state[1]
        except TypeError:
            pass
        raise MifOpError("Blender could not write '%s': %s. %s" % (resolved, exc, tail))
    if not os.path.isfile(resolved):
        # THE PATH PUT BACK, as the except above does: on the repoint route filepath_raw already pointed at the missing
        # file, and nothing restored it (found 2026-10-25). The source too where this op set it (the render route); on
        # the save route Blender set it, and putting GENERATED back would regenerate the pixels from the fill colour.
        if repoint:
            img.filepath_raw = state[0]
            try:
                img.file_format = state[1]
            except TypeError:
                pass
            if route != "save" and img.source != state[2]:
                img.source = state[2]
        raise MifOpError("Blender reported the save done and no file is at '%s'. Nothing was written%s."
                         % (resolved, "; the image's path was put back" if repoint else ""))
    out = {
        "image": img.name,
        "file": resolved,
        "bytes": os.path.getsize(resolved),
        "format": fmt,
        "colorDepth": depth,
        "quality": quality,
        "route": route,
        "repointed": repoint,
        "source": img.source,
        "filepath": img.filepath if repoint else None,
        "dirty": bool(img.is_dirty),
        "colorSpace": img.colorspace_settings.name,
    }
    out.update(replaced_fields(before))
    if not repoint:
        out["note"] = "a copy was written; the datablock still reads %s" % (state[0] or "its generated pixels")
    return out


# --------------------------------------------------------------------------------------------- derive_image_map

_DERIVE_KEYS = ("image", "kind", "name", "strength", "blur", "contrast", "invert", "flipGreen", "range", "filepath",
                "colorspace")
_KINDS = ("HEIGHT", "NORMAL", "SPECULAR")


def _box_blur(np, a, radius):
    """A separable box blur, three passes (close to a Gaussian), edges clamped. radius in pixels."""
    if radius <= 0:
        return a
    r = int(radius)
    for _ in range(3):
        for axis in (0, 1):
            pad = [(0, 0), (0, 0)]
            pad[axis] = (r + 1, r)
            p = np.pad(a, pad, mode="edge")
            c = np.cumsum(p, axis=axis, dtype=np.float64)
            if axis == 0:
                a = (c[2 * r + 1:, :] - c[:-2 * r - 1, :]) / (2 * r + 1)
            else:
                a = (c[:, 2 * r + 1:] - c[:, :-2 * r - 1]) / (2 * r + 1)
    return a.astype(np.float32)


def op_derive_image_map(params):
    """Derive a HEIGHT, NORMAL or SPECULAR map from one colour image - the missing maps of an iris or a skin.

    params:
      image (str)              required: the colour image
      kind (str)               HEIGHT | NORMAL | SPECULAR
      name (str)               the new image's name (default '<image>_<kind>')
      strength (float)         NORMAL: the slope scale (default 1); others: unused and refused
      blur (float)             pixels of blur on the height first (default 0)
      contrast (float)         about mid-grey, applied to the height (default 1)
      invert (bool)            dark is high (default false: light is high)
      flipGreen (bool)         NORMAL: green down, the DirectX convention (default false: Blender's, green up)
      range [lo, hi]           SPECULAR: the height mapped onto lo..hi (default [0, 0.5]; 0.5 is the Principled
                               default specular level, so a mid-grey iris lands near it)
      filepath (str)           also write it (PNG 16-bit, through save_image)
      colorspace (str)         default Non-Color: these are data maps
    The height is the image's luminance (0.2126 R + 0.7152 G + 0.0722 B of its stored values). The new image is a
    float image with a fake user; the reply gives the value range of what was made.
    """
    reject_unknown(params, _DERIVE_KEYS, "derive_image_map")
    tail = "NOTHING was created."
    import numpy as np   # noqa: PLC0415 - bundled with every Blender; only this op needs it
    src = _get_image(take(params, "image", required=True, kind=str), tail)
    kind = str(take(params, "kind", required=True, kind=str)).upper()
    if kind not in _KINDS:
        raise MifOpError("kind '%s' is not one of %s. %s" % (kind, ", ".join(_KINDS), tail))
    strength = take_float(params, "strength", default=None)
    if strength is not None and kind != "NORMAL":
        raise MifOpError("strength scales a NORMAL map's slopes; drop it for %s. %s" % (kind, tail))
    strength = 1.0 if strength is None else strength
    blur = take_float(params, "blur", default=0.0)
    if blur < 0 or blur > 256:
        raise MifOpError("blur %s is outside 0-256 pixels. %s" % (blur, tail))
    contrast = take_float(params, "contrast", default=1.0)
    invert = take_bool(params, "invert", default=False)
    flip = take_bool(params, "flipGreen", default=False)
    if params.get("flipGreen") is not None and kind != "NORMAL":
        raise MifOpError("flipGreen is a NORMAL map's convention; drop it for %s. %s" % (kind, tail))
    rng = params.get("range")
    if rng is not None:
        if kind != "SPECULAR":
            raise MifOpError("range is SPECULAR's; drop it for %s. %s" % (kind, tail))
        if not isinstance(rng, (list, tuple)) or len(rng) != 2:
            raise MifOpError("range takes [lo, hi], got %r. %s" % (rng, tail))
        rng = finite_floats(list(rng), "range")
    else:
        rng = [0.0, 0.5]
    name = str(take(params, "name", default="%s_%s" % (src.name.rsplit(".", 1)[0], kind.lower()), kind=str))
    raw_path = take(params, "filepath", default=None, kind=str)
    if raw_path is not None:
        check_output_path(raw_path, os.path.abspath(bpy.path.abspath(str(raw_path))), "created")
    cs = take(params, "colorspace", default="Non-Color", kind=str)
    w, h = src.size[0], src.size[1]
    if w < 2 or h < 2:
        raise MifOpError("'%s' is %dx%d - a map needs at least 2x2 pixels. %s" % (src.name, w, h, tail))

    px = np.empty(w * h * 4, dtype=np.float32)
    src.pixels.foreach_get(px)
    px = px.reshape(h, w, 4)
    height = 0.2126 * px[:, :, 0] + 0.7152 * px[:, :, 1] + 0.0722 * px[:, :, 2]
    height = _box_blur(np, height.astype(np.float32), blur)
    height = np.clip((height - 0.5) * contrast + 0.5, 0.0, 1.0)
    if invert:
        height = 1.0 - height
    out = np.ones((h, w, 4), dtype=np.float32)
    if kind == "HEIGHT":
        for c in range(3):
            out[:, :, c] = height
    elif kind == "SPECULAR":
        spec = rng[0] + (rng[1] - rng[0]) * height
        for c in range(3):
            out[:, :, c] = spec
    else:
        # CENTRAL DIFFERENCES in pixels, edges CLAMPED (a wrap would put a cliff at the border of an image that does
        # not tile); Blender's tangent space is +X right, +Y up, and pixel rows run bottom to top
        p = np.pad(height, 1, mode="edge")
        dx = (p[1:-1, 2:] - p[1:-1, :-2]) * 0.5
        dy = (p[2:, 1:-1] - p[:-2, 1:-1]) * 0.5
        nx, ny, nz = -dx * strength * w / 256.0, -dy * strength * h / 256.0, np.ones_like(height)
        if flip:
            ny = -ny
        length = np.sqrt(nx * nx + ny * ny + nz * nz)
        out[:, :, 0] = nx / length * 0.5 + 0.5
        out[:, :, 1] = ny / length * 0.5 + 0.5
        out[:, :, 2] = nz / length * 0.5 + 0.5
    img = bpy.data.images.new(name, w, h, alpha=False, float_buffer=True)
    try:
        _colorspace_or_refuse(img, str(cs), tail)
        img.pixels.foreach_set(out.ravel())
        img.update()
        img.use_fake_user = True
        saved = None
        if raw_path is not None:
            saved = op_save_image({"image": img.name, "filepath": raw_path, "format": "PNG"})
    except Exception:
        bpy.data.images.remove(img)
        raise
    rgb = out[:, :, :3]
    return {
        "image": img.name,
        "requestedName": name,
        "nameWasSuffixed": img.name != name,
        "source": src.name,
        "kind": kind,
        "size": [w, h],
        "valueRange": [round(float(rgb.min()), 6), round(float(rgb.max()), 6)],
        "mean": [round(float(rgb[:, :, c].mean()), 6) for c in range(3)],
        "settings": {"strength": strength if kind == "NORMAL" else None, "blur": blur, "contrast": contrast,
                     "invert": invert, "flipGreen": flip if kind == "NORMAL" else None,
                     "range": rng if kind == "SPECULAR" else None},
        "colorSpace": img.colorspace_settings.name,
        "saved": saved,
        "note": ("green up (Blender's convention)" if kind == "NORMAL" and not flip else
                 "green down (DirectX)" if kind == "NORMAL" else None),
    }


# ---------------------------------------------------------------------------------------------------------------------
# bake_hidden_face_map (2026-10-06, spec "A resizable MetaHuman outfit ... hidden-face maps", merged "Make a body
# hidden-face map from a garment")
# ---------------------------------------------------------------------------------------------------------------------

_HFM_KEYS = ("body", "garments", "uvMap", "resolution", "maxDistance", "grayBand", "margin", "keep", "outputPath",
             "overwrite", "maxCullValue", "minKeepValue")


def _hfm_parse(params, tail):
    from .ops_common import get_object   # noqa: PLC0415
    body = get_object(take(params, "body", required=True, kind=str), want_mesh=True)
    raw = params.get("garments")
    names = [raw] if isinstance(raw, str) else raw
    if not isinstance(names, (list, tuple)) or not names or not all(isinstance(n, str) for n in names):
        raise MifOpError("garments is a mesh name or a list of them. %s" % tail)
    garments = [get_object(n, want_mesh=True) for n in dict.fromkeys(names)]
    if body in garments:
        raise MifOpError("'%s' is the body; a garment is what covers it. %s" % (body.name, tail))
    uv_name = take(params, "uvMap", default=None, kind=str)
    layers = body.data.uv_layers
    if not layers:
        raise MifOpError("'%s' has no UV map - a hidden-face map is written in the body's UV layout. %s"
                         % (body.name, tail))
    layer = layers.get(uv_name) if uv_name else (layers.active or layers[0])
    if layer is None:
        raise MifOpError("'%s' has no UV map %r (it has %s). %s" % (body.name, uv_name, [u.name for u in layers], tail))
    res = take_int(params, "resolution", default=512)
    if not 16 <= res <= 4096:
        raise MifOpError("resolution must be 16-4096, got %d. %s" % (res, tail))
    reach = take_float(params, "maxDistance", default=0.1)
    if not reach > 0:
        raise MifOpError("maxDistance must be above 0, got %g. %s" % (reach, tail))
    band = take_int(params, "grayBand", default=4)
    margin = take_int(params, "margin", default=4)
    if band < 0 or margin < 0:
        raise MifOpError("grayBand and margin are texel counts of 0 or more. %s" % tail)
    cull = take_float(params, "maxCullValue", default=0.1)
    keep_v = take_float(params, "minKeepValue", default=0.9)
    if not 0 <= cull < keep_v <= 1:
        raise MifOpError("need 0 <= maxCullValue < minKeepValue <= 1, got %g and %g. %s" % (cull, keep_v, tail))
    boxes = []
    for b in params.get("keep") or []:
        if not isinstance(b, dict) or set(b) != {"min", "max"}:
            raise MifOpError("keep is a list of {min: [x, y, z], max: [x, y, z]} world boxes kept white. %s" % tail)
        lo, hi = finite_floats(list(b["min"]), "keep.min"), finite_floats(list(b["max"]), "keep.max")
        if len(lo) != 3 or len(hi) != 3 or any(a > c for a, c in zip(lo, hi)):
            raise MifOpError("a keep box needs min <= max, three numbers each. %s" % tail)
        boxes.append((lo, hi))
    out_path = take(params, "outputPath", default=None, kind=str)
    if out_path:
        check_output_path(out_path, bpy.path.abspath(out_path), "baked")
        out_path = abs_output_path(out_path)        # Image.save() writes nothing to a relative path (abs_output_path)
    return body, garments, layer, res, reach, band, margin, cull, keep_v, boxes, out_path


def op_bake_hidden_face_map(params):
    """Bake a body hidden-face map from garments: black where a garment covers the body, a gray band at the edges, white
    elsewhere, in the body's UV layout - the mask a MetaHuman outfit hides the body with.

    THE GAP (2026-10-04): an outfit hides the body under it through a texture in the body's UV space, and nothing made
    one - a suit's mask was the engine's body mask painted black by hand. THE ENGINE'S RULE (5.7 / 5.8 geometry
    removal): a body triangle is removed only when every pixel it covers is at or below MaxCullValue (0.1); vertices on
    pixels between MaxCullValue and MinKeepValue (0.9) shrink along minus their normal by MaxShrinkDistance; the value
    is the highest of R, G, B.

    Each texel of the body's UV layout (uvMap, default the active one; resolution square, default 512) inside a
    triangle is a point on the evaluated body with its interpolated normal; a ray from it along the normal that meets
    a garment within maxDistance (default 0.1 m) marks it covered. Covered texels more than grayBand (default 4) texels
    from an uncovered one are 0, the rest of the covered ones 0.5, uncovered ones 1; keep [{min, max}] world boxes are
    forced to 1 (wrists and ankles left showing). margin (default 4) bleeds the values that many texels past the UV
    islands' edges. outputPath writes an 8-bit PNG (overwrite, default true); the image stays in the file either way.

    Reply: the texel counts at 0, 0.5 and 1 and the covered share, and the engine's rule applied to this map:
    trianglesRemoved (every texel at or under maxCullValue) and verticesShrunk (on a texel between the two values), of
    the body's triangles and vertices.
    """
    tail = "NOTHING was made."
    reject_unknown(params, _HFM_KEYS, "bake_hidden_face_map")
    body, garments, layer, res, reach, band, margin, cull, keep_v, boxes, out_path = _hfm_parse(params, tail)
    # IMPORTED AFTER THE PARSE, so a refusal never depends on numpy (the offline refusal sweep has none)
    import numpy as np   # noqa: PLC0415
    from mathutils import Vector   # noqa: PLC0415
    from mathutils.bvhtree import BVHTree   # noqa: PLC0415
    overwrite = take_bool(params, "overwrite", default=True)
    if out_path:
        full = bpy.path.abspath(out_path)
        check_output_path(out_path, full, "written")
        if not full.lower().endswith(".png"):
            raise MifOpError("outputPath must end in .png. %s" % tail)
        if os.path.exists(full) and not overwrite:
            raise MifOpError("%s exists and overwrite is false. %s" % (out_path, tail))
    dg = bpy.context.evaluated_depsgraph_get()
    gv, gt = [], []
    for g in garments:
        ev = g.evaluated_get(dg)
        me = ev.to_mesh()
        me.calc_loop_triangles()
        base = len(gv)
        gv.extend(ev.matrix_world @ v.co for v in me.vertices)
        gt.extend(tuple(base + i for i in t.vertices) for t in me.loop_triangles)
        ev.to_mesh_clear()
    if not gt:
        raise MifOpError("the garments have no faces. %s" % tail)
    bvh = BVHTree.FromPolygons(gv, gt, all_triangles=True)
    evb = body.evaluated_get(dg)
    me = evb.to_mesh()
    try:
        me.calc_loop_triangles()
        uv = me.uv_layers.get(layer.name) or me.uv_layers.active
        mw = evb.matrix_world
        nm = mw.to_3x3().inverted_safe().transposed()
        co = [mw @ v.co for v in me.vertices]
        try:
            vn = [(nm @ v.normal).normalized() for v in me.vertices]
        except AttributeError:
            vn = [(nm @ Vector(n.vector)).normalized() for n in me.vertex_normals]
        tri_rows = []
        for t in me.loop_triangles:
            tri_rows.append((tuple(t.vertices), [tuple(uv.data[li].uv) for li in t.loops]))
        nverts = len(me.vertices)
    finally:
        evb.to_mesh_clear()
    if not tri_rows:
        raise MifOpError("'%s' has no faces. %s" % (body.name, tail))
    value = np.full((res, res), -1.0)          # -1: no triangle covers the texel
    owner = {}
    covered_px = np.zeros((res, res), dtype=bool)
    tri_px = []
    for ti, (vids, uvs) in enumerate(tri_rows):
        p = np.array(uvs) * res - 0.5
        x0, x1 = int(max(0, np.floor(p[:, 0].min()))), int(min(res - 1, np.ceil(p[:, 0].max())))
        y0, y1 = int(max(0, np.floor(p[:, 1].min()))), int(min(res - 1, np.ceil(p[:, 1].max())))
        if x1 < x0 or y1 < y0:
            tri_px.append([])
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        (ax, ay), (bx, by), (cx, cy) = p
        den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(den) < 1e-12:
            tri_px.append([])
            continue
        w0 = ((by - cy) * (xs - cx) + (cx - bx) * (ys - cy)) / den
        w1 = ((cy - ay) * (xs - cx) + (ax - cx) * (ys - cy)) / den
        w2 = 1 - w0 - w1
        inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
        pix = []
        for yy, xx, a, b, c in zip(ys[inside], xs[inside], w0[inside], w1[inside], w2[inside]):
            pt = co[vids[0]] * a + co[vids[1]] * b + co[vids[2]] * c
            n = (vn[vids[0]] * a + vn[vids[1]] * b + vn[vids[2]] * c)
            if n.length < 1e-12:
                continue
            n.normalize()
            if (yy, xx) not in owner:
                owner[(yy, xx)] = ti
                hit = bvh.ray_cast(pt + n * 1e-5, n, reach)[0] is not None
                if hit and boxes:
                    hit = not any(all(lo[k] <= pt[k] <= hi[k] for k in range(3)) for lo, hi in boxes)
                covered_px[yy, xx] = hit
                value[yy, xx] = 0.0
            pix.append((yy, xx))
        tri_px.append(pix)
    filled = value >= 0
    if not filled.any():
        raise MifOpError("no texel of the %dx%d map falls inside a UV triangle of '%s' - the UV map is empty or "
                         "outside 0-1. %s" % (res, res, body.name, tail))
    # distance in texels from an uncovered filled texel, by repeated 8-neighbour growth (band steps at most)
    near_open = filled & ~covered_px
    reach_mask = near_open.copy()
    for _ in range(band):
        grown = reach_mask.copy()
        grown[1:, :] |= reach_mask[:-1, :]
        grown[:-1, :] |= reach_mask[1:, :]
        grown[:, 1:] |= reach_mask[:, :-1]
        grown[:, :-1] |= reach_mask[:, 1:]
        grown[1:, 1:] |= reach_mask[:-1, :-1]
        grown[:-1, :-1] |= reach_mask[1:, 1:]
        grown[1:, :-1] |= reach_mask[:-1, 1:]
        grown[:-1, 1:] |= reach_mask[1:, :-1]
        reach_mask = grown
    value[filled] = 1.0
    value[filled & covered_px] = 0.0
    value[filled & covered_px & reach_mask] = 0.5 if band else 0.0
    n0, ngray, n1 = int((value[filled] == 0).sum()), int((value[filled] == 0.5).sum()), int((value[filled] == 1).sum())
    # THE ENGINE'S RULE on this map, before any bleed
    removed = sum(1 for pix in tri_px if pix and all(value[yy, xx] <= cull for yy, xx in pix))
    shrunk = set()
    for (vids, _uv), pix in zip(tri_rows, tri_px):
        if pix and any(cull < value[yy, xx] < keep_v for yy, xx in pix):
            shrunk.update(vids)
    # BLEED past the islands
    img = value.copy()
    for _ in range(margin):
        empty = img < 0
        if not empty.any():
            break
        acc = np.zeros_like(img)
        cnt = np.zeros_like(img)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            sh = np.roll(np.roll(img, dy, axis=0), dx, axis=1)
            ok = sh >= 0
            acc += np.where(ok, sh, 0)
            cnt += ok
        grow = empty & (cnt > 0)
        img[grow] = acc[grow] / cnt[grow]
    img[img < 0] = 1.0
    name = "%s_HiddenFaceMap" % body.name
    image = bpy.data.images.get(name)
    if image is None or tuple(image.size) != (res, res):
        if image is not None:
            bpy.data.images.remove(image)
        image = bpy.data.images.new(name, width=res, height=res, alpha=False)
    rgba = np.ones((res, res, 4))
    rgba[..., 0] = rgba[..., 1] = rgba[..., 2] = img
    image.pixels[:] = rgba.ravel().tolist()
    image.update()
    saved = None
    if out_path:
        image.filepath_raw = out_path
        image.file_format = "PNG"
        image.save()
        saved = out_path
    return {"body": body.name, "garments": [g.name for g in garments], "uvMap": layer.name, "image": image.name,
            "resolution": res, "savedTo": saved, "maxDistance": reach, "grayBand": band, "margin": margin,
            "texels": {"black": n0, "gray": ngray, "white": n1, "outsideUv": int((~filled).sum())},
            "coveredShare": round((n0 + ngray) / float(max(1, n0 + ngray + n1)), 6),
            "engineCheck": {"maxCullValue": cull, "minKeepValue": keep_v, "trianglesRemoved": removed,
                            "triangles": len(tri_rows), "verticesShrunk": len(shrunk), "vertices": nverts,
                            "note": "the 5.7 / 5.8 rule: a triangle goes when every texel it covers is at or under "
                                    "maxCullValue; a vertex on a texel between the two values shrinks by "
                                    "MaxShrinkDistance (0 by default, so nothing moves until it is set)"}}

# --------------------------------------------------------------------------------------------- resize_image

_RESIZE_KEYS = ("image", "name", "width", "height", "filepath", "format", "colorspace", "overwrite", "pack")
_RESIZE_NOTES = {
    "scale": "give the exact width and height in pixels; a factor is not taken, so the size written is the one asked",
    "colorDepth": "the image's own depth is written (save_image's rule); save_image takes colorDepth",
}


def op_resize_image(params):
    """Resize an image read from a file to an exact width and height, and re-encode it - a texture re-exported at 4K.

    THE GAP (spec, "Resize or re-encode an image to an exact size and format", 2026-10-06): Image.scale appeared only in
    preview thumbnails and bake_texture only makes an image at a size, so a painted colour map re-exported at 4K and its
    normal at 8K had no call.

    params:
      image / name (str)     required: an image READ FROM A FILE with no unsaved edits. A generated canvas or one with
                             edits in memory is refused - save it with save_image first - so a failed write can always
                             be undone by reading the file again. A packed image is refused (its packed bytes would
                             stay at the old size); a tiled (UDIM) image is refused (scale reaches one tile).
      width, height (int)    pixels, 1-16384
      filepath (str)         write the resized image there and point the datablock at it (save_image's route); omitted,
                             the resize lives in memory only and the note says so
      format (str)           PNG | TARGA | OPEN_EXR | JPEG | any still format of the build; default from the extension.
                             Needs filepath. An EXR of an 8-bit image is refused (save_image's rule).
      colorspace (str)       the datablock's colour space, set before the resize
      overwrite (bool)       default true; false refuses a path that exists
      pack (bool)            pack the written file into the .blend (needs filepath)
    Replies sizeBefore, sizeAfter and readBack: the written file loaded again on its own and measured.
    """
    reject_unknown(params, _RESIZE_KEYS, "resize_image", notes=_RESIZE_NOTES)
    tail = "NOTHING was changed."
    img = _get_image(take(params, "image", "name", required=True, kind=str), tail)
    size = []
    for key in ("width", "height"):
        val = take_int(params, key, required=True)
        if not 1 <= val <= 16384:
            raise MifOpError("%s %d is outside 1-16384. %s" % (key, val, tail))
        size.append(val)
    raw = take(params, "filepath", default=None, kind=str)
    fmt = take(params, "format", default=None, kind=str)
    cs = take(params, "colorspace", default=None, kind=str)
    overwrite = take_bool(params, "overwrite", default=True)
    pack = take_bool(params, "pack", default=False)
    if raw is None and (fmt is not None or pack):
        raise MifOpError("%s needs filepath - the resized image is written there first. %s"
                         % ("format" if fmt is not None else "pack", tail))
    if getattr(img, "packed_file", None):
        raise MifOpError("'%s' is PACKED: scaling it changes the pixels in memory and the packed copy stays at the old "
                         "size, so a reload or a reopen undoes it. Unpack it first. %s" % (img.name, tail))
    if img.source == "TILED":
        raise MifOpError("'%s' is a tiled (UDIM) image and a resize reaches its first tile only - resize each tile's "
                         "file, then create_udim_image loads them again. %s" % (img.name, tail))
    path = bpy.path.abspath(img.filepath) if img.filepath else ""
    if img.source != "FILE" or not path:
        raise MifOpError("'%s' is %s and not read from a file - an unsaved canvas. save_image writes it first, so a "
                         "failed write here can be undone by reading the file again. %s" % (img.name, img.source, tail))
    if not os.path.isfile(path):
        raise MifOpError("'%s' points at %s, which does not exist. %s" % (img.name, path, tail))
    if img.is_dirty:
        raise MifOpError("'%s' has UNSAVED edits in memory - save it with save_image first, or the edits would be "
                         "resampled and the full-size original exist nowhere. %s" % (img.name, tail))
    before = [int(img.size[0]), int(img.size[1])]
    if before == [0, 0]:
        raise MifOpError("'%s' read no pixels from %s. %s" % (img.name, path, tail))
    if cs is not None:
        valid = [i.identifier for i in img.colorspace_settings.bl_rna.properties["name"].enum_items]
        if cs not in valid:
            raise MifOpError("'%s' is not a colour space on this Blender. Available: %s. %s"
                             % (cs, ", ".join(valid), tail))
    resolved = None
    if raw is not None:
        resolved = os.path.abspath(bpy.path.abspath(str(raw)))
        check_output_path(raw, resolved, "written")
        if fmt is not None:
            fmt = str(fmt).upper()
            if fmt not in _image_formats():
                raise MifOpError("format '%s' is not a still-image format on Blender %s. Available: %s. %s"
                                 % (fmt, bpy.app.version_string, ", ".join(_image_formats()), tail))
        else:
            fmt = _format_from_path(resolved) or "PNG"
        if fmt.startswith("OPEN_EXR") and not img.is_float:
            raise MifOpError("'%s' is an 8-bit image and %s stores floats - the values would be converted, not kept. "
                             "Write PNG, TARGA or JPEG. %s" % (img.name, fmt, tail))
        if file_before(resolved) is not None and not overwrite:
            raise MifOpError("'%s' exists and overwrite is false. %s" % (resolved, tail))
        folder = os.path.dirname(resolved)
        if folder and not os.path.isdir(folder):
            raise MifOpError("the folder '%s' does not exist. %s" % (folder, tail))

    old_cs = img.colorspace_settings.name
    if cs is not None:
        img.colorspace_settings.name = cs
    img.scale(size[0], size[1])
    saved = None
    if resolved is not None:
        try:
            saved = op_save_image({"image": img.name, "filepath": resolved, "format": fmt, "overwrite": overwrite})
        except MifOpError as exc:
            img.colorspace_settings.name = old_cs
            img.reload()          # it was read from a file with no edits, so the file puts it back
            raise MifOpError("%s The resize was undone: '%s' reads %s again." % (exc, img.name, path))
    if pack:
        img.pack()
    out = {"image": img.name, "sizeBefore": before, "sizeAfter": [int(img.size[0]), int(img.size[1])],
           "colorSpace": img.colorspace_settings.name, "packed": bool(getattr(img, "packed_file", None)),
           "file": resolved, "format": fmt, "readBack": None}
    if saved is not None:
        out.update({k: saved[k] for k in ("bytes", "overwrote", "replacedBytes", "replacedModified") if k in saved})
        probe = bpy.data.images.load(resolved, check_existing=False)
        try:
            out["readBack"] = {"size": [int(probe.size[0]), int(probe.size[1])], "file": resolved}
        finally:
            bpy.data.images.remove(probe)
    else:
        out["note"] = ("resized in memory only: '%s' still reads %s at %dx%d on a reload or reopen - pass filepath, or "
                       "save_image, to keep it" % (img.name, path, before[0], before[1]))
    return out


# --------------------------------------------------------------------------------------------- create_udim_image

_UDIM_KEYS = ("name", "tiles", "filepath", "overwrite", "colorspace", "fakeUser", "material", "input")
_UDIM_NOTES = {
    "width": "a tile is read at its file's own size; resize_image changes a file's size first",
    "color": "the tiles are read from files; create_image makes a one-colour canvas",
}


def _udim_pattern_of(tiles):
    """The '<UDIM>' path every tile file already fits, or None: each file's name holds its own tile number."""
    pats = set()
    for num, path in tiles.items():
        folder, base = os.path.split(path)
        at = base.rfind(str(num))
        if at < 0:
            return None
        pats.add(os.path.join(folder, base[:at] + "<UDIM>" + base[at + 4:]))
    return pats.pop() if len(pats) == 1 else None


def _udim_tiles_on_disk(pattern):
    """Tile numbers whose file fits the pattern in its folder - every one of them is loaded by Blender."""
    import re  # noqa: PLC0415
    folder, base = os.path.split(pattern)
    head, rest = base.split("<UDIM>", 1)
    rx = re.compile(re.escape(head) + r"(\d{4})" + re.escape(rest) + "$")
    found = set()
    if os.path.isdir(folder):
        for f in os.listdir(folder):
            m = rx.match(f)
            if m:
                found.add(int(m.group(1)))
    return found


def op_create_udim_image(params):
    """Load tile files as ONE tiled (UDIM) image - a face on tile 1001 and a body on 1002 read through one texture.

    THE GAP (spec, "... and make UDIM-tiled images", 2026-10-06): nothing made a tiled image, so two textures could not
    load as one image over two UV tiles.

    params:
      name (str)                 required
      tiles {"1001": path, ...}  required: tile number (1001-2000) to an image file. Each tile keeps its file's size.
      filepath (str)             a pattern holding <UDIM> ('//tex/body.<UDIM>.png'): each file is COPIED to its tile's
                                 name there (same extension - resize_image re-encodes). Omitted, the files must already
                                 be named by one pattern (body.1001.png, body.1002.png).
      overwrite (bool)           default true; false refuses a copy over a file that exists
      colorspace (str)           default the build's own for the file; Non-Color for a normal or data map
      fakeUser (bool)            default true
      material + input           wire it into that material (set_material_texture's route)
    REFUSED before anything is copied or loaded: a missing file, a tile outside 1001-2000, names that fit no pattern and
    no filepath, a pattern without <UDIM>, a mismatched extension, and a folder holding OTHER tiles of the pattern -
    Blender loads every tile it finds there.
    """
    reject_unknown(params, _UDIM_KEYS, "create_udim_image", notes=_UDIM_NOTES)
    import shutil  # noqa: PLC0415
    tail = "NOTHING was created."
    name = take(params, "name", required=True, kind=str)
    raw_tiles = params.get("tiles")
    if not isinstance(raw_tiles, dict) or not raw_tiles:
        raise MifOpError("tiles takes a map of tile number to file, {\"1001\": \"face.png\", \"1002\": \"body.png\"}, "
                         "got %r. %s" % (raw_tiles, tail))
    tiles = {}
    for key, path in raw_tiles.items():
        try:
            num = int(str(key))
        except ValueError:
            num = None
        if num is None or not 1001 <= num <= 2000:
            raise MifOpError("tile %r is not a UDIM tile number (1001-2000). %s" % (key, tail))
        if not isinstance(path, str) or not path:
            raise MifOpError("tile %d's file must be a path, got %r. %s" % (num, path, tail))
        full = os.path.abspath(bpy.path.abspath(path))
        if not os.path.isfile(full):
            raise MifOpError("tile %d's file %s does not exist. %s" % (num, full, tail))
        tiles[num] = full
    raw = take(params, "filepath", default=None, kind=str)
    overwrite = take_bool(params, "overwrite", default=True)
    cs = take(params, "colorspace", default=None, kind=str)
    fake = take_bool(params, "fakeUser", default=True)
    mat_name = take(params, "material", default=None, kind=str)
    which = take(params, "input", default=None, kind=str)
    if (mat_name is None) != (which is None):
        raise MifOpError("material and input go together - which material, and which of its inputs. %s" % tail)
    if mat_name is not None and bpy.data.materials.get(mat_name) is None:
        have = sorted(m.name for m in bpy.data.materials)[:25]
        raise MifOpError("no material named '%s'. Present: %s. %s" % (mat_name, ", ".join(have) or "(none)", tail))
    copies = []
    if raw is None:
        pattern = _udim_pattern_of(tiles)
        if pattern is None:
            raise MifOpError("the files are not named by one tile pattern (body.1001.png, body.1002.png in one folder): "
                             "%s. Pass filepath with <UDIM> to copy them into one. %s"
                             % (", ".join("%d: %s" % kv for kv in sorted(tiles.items())), tail))
    else:
        pattern = os.path.abspath(bpy.path.abspath(str(raw)))
        if "<UDIM>" not in os.path.basename(pattern):
            raise MifOpError("filepath '%s' holds no <UDIM> in its file name - the tile number goes there "
                             "('//tex/body.<UDIM>.png'). %s" % (raw, tail))
        check_output_path(raw, pattern.replace("<UDIM>", "1001"), "written")
        ext = os.path.splitext(pattern)[1].lower()
        wrong = ["%d: %s" % (n, p) for n, p in sorted(tiles.items()) if os.path.splitext(p)[1].lower() != ext]
        if wrong:
            raise MifOpError("a tile file is COPIED, not re-encoded, so each must already be %s: %s - resize_image "
                             "writes another format. %s" % (ext or "(no extension)", ", ".join(wrong), tail))
        folder = os.path.dirname(pattern)
        if not os.path.isdir(folder):
            raise MifOpError("the folder '%s' does not exist. %s" % (folder, tail))
        for num, src in sorted(tiles.items()):
            dst = pattern.replace("<UDIM>", str(num))
            if os.path.normcase(dst) == os.path.normcase(src):
                continue
            if os.path.isfile(dst) and not overwrite:
                raise MifOpError("'%s' exists and overwrite is false. %s" % (dst, tail))
            copies.append((num, src, dst))
    others = sorted(_udim_tiles_on_disk(pattern) - set(tiles))
    if others:
        raise MifOpError("the folder of %s also holds tile(s) %s, and Blender loads every tile it finds there - copy "
                         "these into their own folder with filepath. %s"
                         % (pattern, ", ".join(str(n) for n in others), tail))

    copied = []
    for num, src, dst in copies:
        before = file_before(dst)
        shutil.copyfile(src, dst)
        row = {"tile": num, "from": src, "to": dst}
        row.update(replaced_fields(before))
        copied.append(row)
    first = pattern.replace("<UDIM>", str(min(tiles)))
    img = bpy.data.images.load(first, check_existing=False)
    try:
        assign_id_name(img, str(name), bpy.data.images)
        img.source = "TILED"
        img.filepath = pattern
        img.reload()
        got = {t.number: [int(t.size[0]), int(t.size[1])] for t in img.tiles}
        if set(got) != set(tiles) or any(s == [0, 0] for s in got.values()):
            raise MifOpError("Blender read tiles %s from %s, not %s. NOTHING was created%s."
                             % (got, pattern, sorted(tiles), " (the copies were left)" if copied else ""))
        if cs:
            _colorspace_or_refuse(img, str(cs), tail)
        img.use_fake_user = fake
        wired = None
        if mat_name is not None:
            from .ops_material import op_set_material_texture   # noqa: PLC0415 - the one wiring route
            wired = op_set_material_texture({"material": mat_name, "image": img.name, "input": which})
    except Exception:
        bpy.data.images.remove(img)
        raise
    return {
        "image": img.name,
        "requestedName": name,
        "nameWasSuffixed": img.name != name,
        "source": img.source,
        "filepath": img.filepath,
        "tiles": [{"number": t.number, "size": [int(t.size[0]), int(t.size[1])],
                   "file": pattern.replace("<UDIM>", str(t.number))} for t in img.tiles],
        "copied": copied,
        "colorSpace": img.colorspace_settings.name,
        "fakeUser": bool(img.use_fake_user),
        "wired": wired,
        "note": "a UV map reaches tile 1002 at u 1-2, 1011 at v 1-2; a face whose UVs stay in 0-1 reads tile 1001 only",
    }

# --------------------------------------------------------------------------------------------- paint_texture

_PAINT_KEYS = ("image", "material", "input", "object", "uvMap", "layers", "symmetry", "palette", "base")
_PAINT_LAYER_KEYS = ("fill", "mask", "opacity", "blend", "tiles", "name")
_PAINT_MASK_KINDS = ("faces", "vertexGroup", "uvBox", "strokes", "image")
_PAINT_BLENDS = ("MIX", "MULTIPLY", "ADD", "ERASE_ALPHA")
_PAINT_NOTES = {
    "brush": "a stroke is a layer's mask: mask {strokes: [{points [[u, v], ...], radiusPx, strength, falloff}]}",
    "color": "a layer's fill: {color: [r, g, b(, a)]}, {color: '<palette name>'} or {image: name}",
    "filepath": "paint_texture paints in memory; save_image writes the image to disk",
}


def _paint_image_of(params, tail):
    """The image named, or the one an Image Texture feeds into the material's input (through a Normal Map node)."""
    name = take(params, "image", default=None, kind=str)
    mat_name = take(params, "material", default=None, kind=str)
    which = take(params, "input", default=None, kind=str)
    if (mat_name is None) != (which is None):
        raise MifOpError("material and input go together - which material, and which of its inputs. %s" % tail)
    if (name is None) == (mat_name is None):
        raise MifOpError("pass image, or material + input (the image wired into that input) - one of them. %s" % tail)
    if name is not None:
        return _get_image(name, tail)
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        have = sorted(m.name for m in bpy.data.materials)[:25]
        raise MifOpError("no material named '%s'. Present: %s. %s" % (mat_name, ", ".join(have) or "(none)", tail))
    from .ops_material import principled_socket   # noqa: PLC0415
    tree = mat.node_tree if mat.use_nodes else None
    bsdf = next((n for n in tree.nodes if n.type == "BSDF_PRINCIPLED"), None) if tree else None
    sock = principled_socket(bsdf, which) if bsdf is not None else None
    if sock is None:
        raise MifOpError("material '%s' has no Principled input '%s'. %s" % (mat_name, which, tail))
    todo, seen = [sock], 0
    while todo and seen < 6:
        seen += 1
        cur = todo.pop(0)
        for link in cur.links if cur.is_linked else ():
            node = link.from_node
            if node.type == "TEX_IMAGE" and node.image is not None:
                return node.image
            todo.extend(i for i in node.inputs if i.is_linked)
    raise MifOpError("no Image Texture feeds '%s' of material '%s' - set_material_texture or create_image with "
                     "material + input wires one. %s" % (which, mat_name, tail))


def _paint_color(raw, palette, where, tail):
    if isinstance(raw, str):
        if raw not in palette:
            raise MifOpError("%s names palette colour '%s' and the palette holds: %s. %s"
                             % (where, raw, ", ".join(sorted(palette)) or "(none - pass palette)", tail))
        raw = palette[raw]
    if not isinstance(raw, (list, tuple)) or len(raw) not in (3, 4) or any(isinstance(c, bool) for c in raw):
        raise MifOpError("%s takes [r, g, b] or [r, g, b, a], got %r. %s" % (where, raw, tail))
    col = finite_floats(list(raw), where)
    return col + [1.0] if len(col) == 3 else col


def _paint_parse_layer(i, lay, ctx, tail):
    """One layer checked whole - every refusal here fires before a pixel is written."""
    where = "layers[%d]" % i
    if not isinstance(lay, dict):
        raise MifOpError("%s must be an object {fill, mask, opacity, blend, tiles}, got %r. %s" % (where, lay, tail))
    bad = sorted(set(lay) - set(_PAINT_LAYER_KEYS))
    if bad:
        raise MifOpError("%s: unknown key(s) %s. Accepted: %s. %s" % (where, ", ".join(bad), ", ".join(_PAINT_LAYER_KEYS),
                                                                      tail))
    fill = lay.get("fill")
    erase = str(lay.get("blend", "MIX")).upper() == "ERASE_ALPHA"
    if erase and fill is not None:
        raise MifOpError("%s: ERASE_ALPHA takes no fill - it lowers the alpha under the mask. %s" % (where, tail))
    if erase:
        fill = {"color": [0.0, 0.0, 0.0, 1.0]}
    if not isinstance(fill, dict) or len(fill) != 1 or next(iter(fill)) not in ("color", "image"):
        raise MifOpError("%s.fill takes exactly one of {color: [r, g, b(, a)] | palette name} or {image: name}, got %r. "
                         "%s" % (where, fill, tail))
    # A STRING OR NOTHING: a mapping or a list went into the reply's layer name as it came (2026-10-28, the version
    # matrix's corrupted-value pass).
    if lay.get("name") is not None and not isinstance(lay["name"], str):
        raise MifOpError("%s.name is text, got %r. %s" % (where, lay["name"], tail))
    out = {"name": lay.get("name") or "layer %d" % i}
    if "color" in fill:
        out["color"] = _paint_color(fill["color"], ctx["palette"], where + ".fill.color", tail)
    else:
        src = _get_image(fill["image"], tail)
        if tuple(src.size) != ctx["size"]:
            raise MifOpError("%s.fill.image '%s' is %dx%d and the canvas is %dx%d - resize_image makes them match. %s"
                             % (where, src.name, src.size[0], src.size[1], ctx["size"][0], ctx["size"][1], tail))
        out["fillImage"] = src
    op = lay.get("opacity", 1.0)
    if isinstance(op, bool) or not isinstance(op, (int, float)) or not 0.0 <= float(op) <= 1.0:
        raise MifOpError("%s.opacity takes 0-1, got %r. %s" % (where, op, tail))
    out["opacity"] = float(op)
    blend = str(lay.get("blend", "MIX")).upper()
    if blend not in _PAINT_BLENDS:
        raise MifOpError("%s.blend '%s' is not one of %s. %s" % (where, blend, ", ".join(_PAINT_BLENDS), tail))
    out["blend"] = blend
    mask = lay.get("mask")
    if mask is None:
        out["mask"] = None
    else:
        if not isinstance(mask, dict) or len(mask) != 1 or next(iter(mask)) not in _PAINT_MASK_KINDS:
            raise MifOpError("%s.mask takes exactly one of %s (none paints the whole image), got %r. %s"
                             % (where, ", ".join(_PAINT_MASK_KINDS), mask, tail))
        kind, val = next(iter(mask.items()))
        out["mask"] = (kind, _paint_parse_mask(where, kind, val, ctx, tail))
    tiles = lay.get("tiles")
    if tiles is not None:
        if out["mask"] is None or out["mask"][0] not in ("faces", "vertexGroup"):
            raise MifOpError("%s.tiles limits a mesh mask (faces or vertexGroup) to the faces whose UVs sit in those "
                             "tiles; a %s mask is drawn in the image's own 0-1 square. %s"
                             % (where, out["mask"][0] if out["mask"] else "missing", tail))
        if not isinstance(tiles, list) or not tiles or any(isinstance(t, bool) or not isinstance(t, int)
                                                           or not 1001 <= t <= 2000 for t in tiles):
            raise MifOpError("%s.tiles takes a list of UDIM tile numbers (1001-2000), got %r. %s" % (where, tiles, tail))
        out["tiles"] = sorted(set(tiles))
    return out


def _paint_parse_mask(where, kind, val, ctx, tail):
    if kind in ("faces", "vertexGroup"):
        obj = ctx.get("object")
        if obj is None:
            raise MifOpError("%s.mask.%s reads a mesh - pass object (and uvMap, default the active one). %s"
                             % (where, kind, tail))
        if kind == "faces":
            n = len(obj.data.polygons)
            if val == "all":
                return list(range(n))
            if not isinstance(val, list) or not val or any(isinstance(f, bool) or not isinstance(f, int)
                                                           or not 0 <= f < n for f in val):
                raise MifOpError("%s.mask.faces takes 'all' or face indices 0-%d, got %r. %s" % (where, n - 1, val, tail))
            return sorted(set(val))
        vg = obj.vertex_groups.get(str(val)) if isinstance(val, str) else None
        if vg is None:
            raise MifOpError("%s.mask.vertexGroup: '%s' has no vertex group %r. It has: %s. %s"
                             % (where, obj.name, val, ", ".join(g.name for g in obj.vertex_groups) or "(none)", tail))
        return vg.index
    if kind == "uvBox":
        if (not isinstance(val, list) or len(val) != 4 or any(isinstance(c, bool) for c in val)):
            raise MifOpError("%s.mask.uvBox takes [u0, v0, u1, v1], got %r. %s" % (where, val, tail))
        box = finite_floats(list(val), where + ".mask.uvBox")
        if not (box[0] < box[2] and box[1] < box[3]):
            raise MifOpError("%s.mask.uvBox needs u0 < u1 and v0 < v1, got %r. %s" % (where, box, tail))
        return box
    if kind == "image":
        src = _get_image(val, tail)
        if tuple(src.size) != ctx["size"]:
            raise MifOpError("%s.mask.image '%s' is %dx%d and the canvas is %dx%d. %s"
                             % (where, src.name, src.size[0], src.size[1], ctx["size"][0], ctx["size"][1], tail))
        return src
    if not isinstance(val, list) or not val:
        raise MifOpError("%s.mask.strokes takes a list of strokes [{points [[u, v], ...], radiusPx, strength, falloff}]. "
                         "%s" % (where, tail))
    strokes = []
    for j, st in enumerate(val):
        sw = "%s.mask.strokes[%d]" % (where, j)
        if isinstance(st, list):
            st = {"points": st}
        if not isinstance(st, dict) or sorted(set(st) - {"points", "radiusPx", "strength", "falloff"}):
            raise MifOpError("%s takes {points, radiusPx, strength, falloff}, got %r. %s" % (sw, st, tail))
        pts = st.get("points")
        if (not isinstance(pts, list) or not pts or any(not isinstance(q, list) or len(q) != 2
                                                        or any(isinstance(c, bool) for c in q) for q in pts)):
            raise MifOpError("%s.points takes [[u, v], ...] (one point is a dab), got %r. %s" % (sw, pts, tail))
        pts = [finite_floats(list(q), sw + ".points") for q in pts]
        rad = st.get("radiusPx", 4)
        stren = st.get("strength", 1.0)
        if isinstance(rad, bool) or not isinstance(rad, (int, float)) or not 0.5 <= float(rad) <= 4096:
            raise MifOpError("%s.radiusPx takes 0.5-4096, got %r. %s" % (sw, rad, tail))
        if isinstance(stren, bool) or not isinstance(stren, (int, float)) or not 0.0 < float(stren) <= 1.0:
            raise MifOpError("%s.strength takes above 0 up to 1, got %r. %s" % (sw, stren, tail))
        fall = str(st.get("falloff", "SMOOTH")).upper()
        if fall not in ("SMOOTH", "CONSTANT", "LINEAR"):
            raise MifOpError("%s.falloff takes SMOOTH, LINEAR or CONSTANT, got %r. %s" % (sw, fall, tail))
        strokes.append({"points": pts, "radius": float(rad), "strength": float(stren), "falloff": fall})
    return strokes


def _paint_tris(np, obj, uv_name, face_ids, tiles):
    """UV triangles of the faces, per corner ([n,3,2] uv, [n,3] vertex index), keeping faces in `tiles`."""
    me = obj.data
    uvl = me.uv_layers[uv_name].data
    uvs, vids = [], []
    for fi in face_ids:
        poly = me.polygons[fi]
        lo = list(poly.loop_indices)
        cu = [tuple(uvl[li].uv) for li in lo]
        if tiles is not None:
            cuu = sum(c[0] for c in cu) / len(cu)
            cvv = sum(c[1] for c in cu) / len(cu)
            tile = 1001 + int(np.floor(cuu)) + 10 * int(np.floor(cvv))
            if tile not in tiles:
                continue
        for k in range(1, len(lo) - 1):
            uvs.append([cu[0], cu[k], cu[k + 1]])
            vids.append([me.loops[lo[0]].vertex_index, me.loops[lo[k]].vertex_index, me.loops[lo[k + 1]].vertex_index])
    return np.array(uvs, dtype=np.float64).reshape(-1, 3, 2), np.array(vids, dtype=np.int64).reshape(-1, 3)


def _paint_raster(np, w, h, uvs, values):
    """Rasterise UV triangles (wrapped into the 0-1 square) at pixel centres; values [n,3] per corner, interpolated."""
    out = np.zeros((h, w), dtype=np.float32)
    for tri, val in zip(uvs, values):
        off = np.floor(tri.mean(axis=0))
        p = (tri - off) * [w, h]
        x0, x1 = int(max(0, np.floor(p[:, 0].min()))), int(min(w - 1, np.ceil(p[:, 0].max())))
        y0, y1 = int(max(0, np.floor(p[:, 1].min()))), int(min(h - 1, np.ceil(p[:, 1].max())))
        if x1 < x0 or y1 < y0:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        (ax, ay), (bx, by), (cx, cy) = p
        den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(den) < 1e-12:
            continue
        l1 = ((by - cy) * (xs - cx) + (cx - bx) * (ys - cy)) / den
        l2 = ((cy - ay) * (xs - cx) + (ax - cx) * (ys - cy)) / den
        l3 = 1.0 - l1 - l2
        eps = -1e-6
        inside = (l1 >= eps) & (l2 >= eps) & (l3 >= eps)
        v = l1 * val[0] + l2 * val[1] + l3 * val[2]
        sub = out[y0:y1 + 1, x0:x1 + 1]
        np.maximum(sub, np.where(inside, v, 0.0), out=sub)
    return out


def _paint_strokes(np, w, h, strokes, mirror):
    out = np.zeros((h, w), dtype=np.float32)
    for st in strokes:
        sets = [st["points"]]
        if mirror:
            sets.append([[np.floor(u) + 1.0 - (u - np.floor(u)), v] for u, v in st["points"]])
        r = st["radius"]
        for pts in sets:
            px = [(u * w, v * h) for u, v in pts]
            segs = list(zip(px, px[1:])) or [(px[0], px[0])]
            for (ax, ay), (bx, by) in segs:
                x0, x1 = int(max(0, np.floor(min(ax, bx) - r))), int(min(w - 1, np.ceil(max(ax, bx) + r)))
                y0, y1 = int(max(0, np.floor(min(ay, by) - r))), int(min(h - 1, np.ceil(max(ay, by) + r)))
                if x1 < x0 or y1 < y0:
                    continue
                xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
                dx, dy = bx - ax, by - ay
                ll = dx * dx + dy * dy
                t = np.clip(((xs - ax) * dx + (ys - ay) * dy) / ll, 0.0, 1.0) if ll > 0 else 0.0
                d = np.hypot(xs - (ax + t * dx), ys - (ay + t * dy)) / r
                if st["falloff"] == "CONSTANT":
                    f = (d <= 1.0).astype(np.float32)
                elif st["falloff"] == "LINEAR":
                    f = np.clip(1.0 - d, 0.0, 1.0)
                else:
                    f = np.clip(1.0 - d * d, 0.0, 1.0) ** 2
                sub = out[y0:y1 + 1, x0:x1 + 1]
                np.maximum(sub, f * st["strength"], out=sub)
    return out


def _paint_mirror_faces(obj, faces, axis):
    """Faces whose centre mirrors a listed face's centre across the object's local axis (1e-4 of the mesh's size)."""
    from mathutils import kdtree   # noqa: PLC0415
    me = obj.data
    tree = kdtree.KDTree(len(me.polygons))
    for p in me.polygons:
        tree.insert(p.center, p.index)
    tree.balance()
    tol = max(1e-6, 1e-4 * max(obj.dimensions) if max(obj.dimensions) > 0 else 1e-4)
    found = set()
    k = "XYZ".index(axis)
    for fi in faces:
        c = me.polygons[fi].center.copy()
        c[k] = -c[k]
        co, idx, dist = tree.find(c)
        if idx is not None and dist <= tol:
            found.add(idx)
    return sorted(found - set(faces))


def _pixels_of(np, img):
    a = np.empty(img.size[0] * img.size[1] * 4, dtype=np.float32)
    img.pixels.foreach_get(a)
    return a.reshape(img.size[1], img.size[0], 4)


def op_paint_texture(params):
    """Paint an image in layers - fills through masks of faces, a vertex group, a UV box, brush strokes or an image.

    THE GAP (spec, "Paint a texture through Blender's own tools, in layers", 2026-10-06): nothing painted an image -
    dynamic paint is a physics wetmap and bake_texture only bakes. A skin edit is a stack: a fill over the whole mesh,
    a black mask revealed by painting white along a pattern, a fill for one body part, pen lines, symmetry, a layer
    limited to one UV tile, and the variation maps rebuilt by swapping the base underneath.

    BLENDER'S OWN BRUSH NEEDS A WINDOW. Measured 2026-10-06 on 5.2.2 headless: with the object in TEXTURE_PAINT mode
    bpy.ops.paint.image_paint.poll() is False ("context is incorrect" - it needs a 3D viewport or image editor), so the
    layers are composited with numpy straight into the image's pixels on every build, with or without a window.

    params:
      image (str) | material + input   the canvas, or the image an Image Texture feeds into that Principled input
      object (str), uvMap (str)        the mesh a faces or vertexGroup mask reads (default the active UV map)
      base (str)                       start from this image's pixels (same size) - the variation swap
      palette {name: [r, g, b, a]}     colours a fill may name
      symmetry (str)                   X (or Y, Z): a faces mask also takes the faces mirrored across the object's local
                                       axis; strokes and uvBox mirror across the middle of the UV tile (a symmetric
                                       unwrap)
      layers [{fill {color | image}, mask {faces [i] | 'all' | vertexGroup | uvBox [u0, v0, u1, v1] |
              strokes [{points [[u, v]], radiusPx 4, strength 1, falloff SMOOTH|LINEAR|CONSTANT}] | image},
              opacity 0-1, blend MIX|MULTIPLY|ADD|ERASE_ALPHA, tiles [1001], name}]   bottom first; no mask paints the
                                       whole image; a vertexGroup mask is the weight; tiles keeps a mesh mask to faces
                                       whose UVs sit in those tiles
    Colours are written as the image stores them: for an 8-bit sRGB image, the value a colour picker shows. A tiled
    image is refused (Python reaches its first tile only). The image is painted in memory; save_image writes it.
    """
    reject_unknown(params, _PAINT_KEYS, "paint_texture", notes=_PAINT_NOTES)
    tail = "NOTHING was painted."
    img = _paint_image_of(params, tail)
    if img.source == "TILED":
        raise MifOpError("'%s' is a tiled (UDIM) image and Python reaches its first tile's pixels only - paint each "
                         "tile's image, then create_udim_image loads them as one. %s" % (img.name, tail))
    w, h = int(img.size[0]), int(img.size[1])
    if w == 0 or h == 0:
        raise MifOpError("'%s' has no pixels (0x0 - a missing file?). %s" % (img.name, tail))
    obj_name = take(params, "object", default=None, kind=str)
    obj = None
    if obj_name is not None:
        obj = bpy.data.objects.get(obj_name)
        if obj is None or obj.type != "MESH":
            raise MifOpError("no mesh object named '%s'. %s" % (obj_name, tail))
    uv_name = take(params, "uvMap", default=None, kind=str)
    if obj is not None:
        if not obj.data.uv_layers:
            raise MifOpError("'%s' has no UV map, so its faces have no place in the image - uv_unwrap first. %s"
                             % (obj.name, tail))
        if uv_name is None:
            uv_name = obj.data.uv_layers.active.name
        elif obj.data.uv_layers.get(uv_name) is None:
            raise MifOpError("'%s' has no UV map '%s'. It has: %s. %s"
                             % (obj.name, uv_name, ", ".join(u.name for u in obj.data.uv_layers), tail))
    elif uv_name is not None:
        raise MifOpError("uvMap goes with object. %s" % tail)
    sym = take(params, "symmetry", default=None, kind=str)
    if sym is not None:
        sym = str(sym).upper()
        if sym not in ("X", "Y", "Z"):
            raise MifOpError("symmetry takes X, Y or Z (the object's local axis), got %r. %s" % (sym, tail))
    palette = params.get("palette") or {}
    if not isinstance(palette, dict):
        raise MifOpError("palette takes {name: [r, g, b, a]}, got %r. %s" % (palette, tail))
    palette = {str(k): _paint_color(v, {}, "palette.%s" % k, tail) for k, v in palette.items()}
    base_name = take(params, "base", default=None, kind=str)
    base = None
    if base_name is not None:
        base = _get_image(base_name, tail)
        if tuple(base.size) != (w, h):
            raise MifOpError("base '%s' is %dx%d and the canvas is %dx%d - resize_image makes them match. %s"
                             % (base.name, base.size[0], base.size[1], w, h, tail))
    layers = params.get("layers")
    if not isinstance(layers, list) or not layers:
        raise MifOpError("layers takes a non-empty list [{fill, mask, opacity, blend, tiles}], bottom first. %s" % tail)
    ctx = {"object": obj, "palette": palette, "size": (w, h)}
    parsed = [_paint_parse_layer(i, lay, ctx, tail) for i, lay in enumerate(layers)]

    import numpy as np   # noqa: PLC0415 - after every refusal
    canvas = _pixels_of(np, base if base is not None else img).copy()
    clamp = not img.is_float
    rows = []
    for lay in parsed:
        mirrored = 0
        if lay["mask"] is None:
            m = np.ones((h, w), dtype=np.float32)
        else:
            kind, val = lay["mask"]
            if kind == "faces":
                faces = list(val)
                if sym:
                    extra = _paint_mirror_faces(obj, faces, sym)
                    mirrored = len(extra)
                    faces += extra
                uvs, vids = _paint_tris(np, obj, uv_name, faces, lay.get("tiles"))
                m = _paint_raster(np, w, h, uvs, np.ones(vids.shape, dtype=np.float32))
            elif kind == "vertexGroup":
                wts = np.zeros(len(obj.data.vertices), dtype=np.float32)
                for v in obj.data.vertices:
                    for g in v.groups:
                        if g.group == val:
                            wts[v.index] = g.weight
                uvs, vids = _paint_tris(np, obj, uv_name, range(len(obj.data.polygons)), lay.get("tiles"))
                m = _paint_raster(np, w, h, uvs, wts[vids] if len(vids) else np.zeros((0, 3)))
            elif kind == "uvBox":
                boxes = [val]
                if sym:
                    boxes.append([1.0 - val[2], val[1], 1.0 - val[0], val[3]])
                m = np.zeros((h, w), dtype=np.float32)
                xs = (np.arange(w) + 0.5) / w
                ys = (np.arange(h) + 0.5) / h
                for b in boxes:
                    m = np.maximum(m, np.outer((ys >= b[1]) & (ys <= b[3]), (xs >= b[0]) & (xs <= b[2])))
            elif kind == "strokes":
                m = _paint_strokes(np, w, h, val, bool(sym))
            else:
                mp = _pixels_of(np, val)
                m = (0.2126 * mp[..., 0] + 0.7152 * mp[..., 1] + 0.0722 * mp[..., 2]).astype(np.float32)
        src = (np.broadcast_to(np.array(lay["color"], dtype=np.float32), (h, w, 4)) if "color" in lay
               else _pixels_of(np, lay["fillImage"]))
        a = np.clip(m, 0.0, 1.0) * lay["opacity"] * src[..., 3]
        a3 = a[..., None]
        blend = lay["blend"]
        if blend == "MIX":
            canvas[..., :3] = canvas[..., :3] * (1.0 - a3) + src[..., :3] * a3
            canvas[..., 3] = a + canvas[..., 3] * (1.0 - a)
        elif blend == "MULTIPLY":
            canvas[..., :3] = canvas[..., :3] * (1.0 - a3) + canvas[..., :3] * src[..., :3] * a3
        elif blend == "ADD":
            canvas[..., :3] = canvas[..., :3] + src[..., :3] * a3
        else:
            canvas[..., 3] = canvas[..., 3] * (1.0 - np.clip(m, 0.0, 1.0) * lay["opacity"])
        if clamp:
            np.clip(canvas, 0.0, 1.0, out=canvas)
        covered = int((m > 0).sum())
        rows.append({"name": lay["name"], "blend": blend, "opacity": lay["opacity"],
                     "mask": lay["mask"][0] if lay["mask"] else None, "coveragePx": covered,
                     "coverageShare": round(covered / float(w * h), 6), "meanStrength": round(float(m.mean()), 6),
                     "mirroredFaces": mirrored, "tiles": lay.get("tiles")})
    img.pixels.foreach_set(canvas.ravel())
    img.update()
    out = {"image": img.name, "size": [w, h], "layers": rows, "base": base.name if base is not None else None,
           "symmetry": sym, "uvMap": uv_name, "route": "numpy", "depth": int(img.depth),
           "dirty": bool(img.is_dirty),
           "note": "painted in memory with numpy (bpy.ops.paint.image_paint needs a window); save_image writes it"}
    # AN OPAQUE SAVE DROPS THE ALPHA PLANE FOR GOOD. Measured 2026-10-06 on 5.2.2: a 32-bit canvas saved while every
    # pixel was opaque reads depth 24 afterwards, and a later save writes RGB even after alpha below 1 is painted -
    # no RNA property puts the plane back. The alpha is in memory; it is said here rather than lost at the save.
    if int(img.depth) in (24, 96) and float(canvas[..., 3].min()) < 1.0:
        out["alphaNotSaved"] = True
        out["warning"] = ("'%s' is an RGB buffer (depth %d - an earlier save found it opaque and dropped the alpha "
                          "plane), so the alpha painted here stays in memory and a save writes RGB. Paint transparency "
                          "on a canvas made with create_image alpha: true that has not been saved opaque."
                          % (img.name, img.depth))
    return out

# --------------------------------------------------------------------------------------------- project_image_decal

_DECAL_KEYS = ("object", "image", "origin", "direction", "size", "rotation", "faces", "limit", "maxDepth",
               "wipeBackFaces", "uvMap", "maskAttribute", "material", "wire")
_DECAL_NOTES = {
    "scale": "size [w, h] is the decal's extent in world units on its plane",
    "mix": "wire (default true) mixes the decal over each covered material's Base Color; material picks one",
}


def _decal_vec(params, key, n, tail, required=True):
    raw = params.get(key)
    if raw is None and not required:
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) != n or any(isinstance(c, bool) for c in raw):
        raise MifOpError("%s takes %d numbers, got %r. %s" % (key, n, raw, tail))
    return finite_floats(list(raw), key)


def _decal_wire(mat, img, uv_name, mask_name):
    """Mix the decal over the material's Base Color: image colour by (image alpha x mask) - reused on a re-run."""
    tree = mat.node_tree
    bsdf = next(n for n in tree.nodes if n.type == "BSDF_PRINCIPLED")
    from .ops_material import principled_socket   # noqa: PLC0415
    base = principled_socket(bsdf, "baseColor")
    nodes = tree.nodes

    def node(kind, name, loc):
        nd = nodes.get(name)
        if nd is None or nd.bl_idname != kind:
            nd = nodes.new(kind)
            nd.name = nd.label = name
            nd.location = (bsdf.location.x + loc[0], bsdf.location.y + loc[1])
        return nd
    fresh = nodes.get("MifDecal Mix") is None   # a re-run keeps what the first run put underneath
    upstream = base.links[0].from_socket if fresh and base.is_linked else None
    default = list(base.default_value)
    uvn = node("ShaderNodeUVMap", "MifDecal UV", (-900, 300))
    uvn.uv_map = uv_name
    tex = node("ShaderNodeTexImage", "MifDecal Image", (-700, 300))
    tex.image = img
    tex.extension = "CLIP"
    att = node("ShaderNodeAttribute", "MifDecal Mask", (-700, 0))
    att.attribute_name = mask_name
    mul = node("ShaderNodeMath", "MifDecal Factor", (-450, 150))
    mul.operation = "MULTIPLY"
    mix = node("ShaderNodeMix", "MifDecal Mix", (-250, 250))
    mix.data_type = "RGBA"
    mix.blend_type = "MIX"
    ins = {s.identifier: s for s in mix.inputs}
    link = tree.links.new
    link(uvn.outputs["UV"], tex.inputs["Vector"])
    link(tex.outputs["Alpha"], mul.inputs[0])
    link(att.outputs["Fac"], mul.inputs[1])
    link(mul.outputs["Value"], ins["Factor_Float"])
    link(tex.outputs["Color"], ins["B_Color"])
    if upstream is not None:
        link(upstream, ins["A_Color"])
    elif fresh:
        ins["A_Color"].default_value = default
    link(next(o for o in mix.outputs if o.identifier == "Result_Color"), base)
    return {"material": mat.name, "mixNode": mix.name, "underneath": "linked" if ins["A_Color"].is_linked else "value"}


def op_project_image_decal(params):
    """Project an image onto the faces it hits from a plane - a logo sprayed onto a garment - and mix it over them.

    THE GAP (spec, "Project an image onto selected faces from a plane", 2026-10-06): uv_unwrap BOX writes one layer for
    the whole mesh, not the faces a decal plane hits; a logo placed, rotated and sprayed onto a garment, back faces
    wiped and mixed over the current material had no call.

    params:
      object (str), image (str)    the mesh and the decal image (already loaded)
      origin [x, y, z]             the decal plane's centre, world space
      direction [x, y, z]          where it projects (toward the mesh); the plane's right axis is direction x world Z
                                   (world Y when projecting along Z), up completes it - both are replied
      size [w, h]                  the decal's extent on its plane, world units
      rotation (deg)               about direction, right-handed (default 0)
      faces [i] | limit (int)      only these faces / at most N faces, nearest the plane first
      maxDepth (float)             only faces whose centre is within this distance in front of the plane
      wipeBackFaces (bool)         default true: faces turned away from the plane are left out
      uvMap (str)                  the UV layer written (default Decal; an existing one is overwritten, said so):
                                   covered faces get plane coordinates (0-1 inside the decal), other faces -1
      maskAttribute (str)          a face-corner float written 1 on covered faces, 0 elsewhere (default DecalMask)
      material (str), wire (bool)  wire true (default) mixes the image over Base Color (factor = image alpha x mask,
                                   image extension CLIP) in material, or in every material on the covered faces
    """
    reject_unknown(params, _DECAL_KEYS, "project_image_decal", notes=_DECAL_NOTES)
    tail = "NOTHING was changed."
    obj = bpy.data.objects.get(str(take(params, "object", required=True, kind=str)))
    if obj is None or obj.type != "MESH":
        raise MifOpError("no mesh object named '%s'. %s" % (params.get("object"), tail))
    img = _get_image(take(params, "image", required=True, kind=str), tail)
    origin = _decal_vec(params, "origin", 3, tail)
    direction = _decal_vec(params, "direction", 3, tail)
    size = _decal_vec(params, "size", 2, tail)
    rotation = take_float(params, "rotation", default=0.0)
    limit = take_int(params, "limit", default=None)
    max_depth = take_float(params, "maxDepth", default=None)
    wipe = take_bool(params, "wipeBackFaces", default=True)
    uv_name = str(take(params, "uvMap", default="Decal", kind=str))
    mask_name = str(take(params, "maskAttribute", default="DecalMask", kind=str))
    mat_name = take(params, "material", default=None, kind=str)
    wire = take_bool(params, "wire", default=True)
    faces = params.get("faces")
    if sum(c * c for c in direction) < 1e-12:
        raise MifOpError("direction is zero - it says where the decal projects. %s" % tail)
    if size[0] <= 0 or size[1] <= 0:
        raise MifOpError("size [w, h] must both be above 0, got %r. %s" % (size, tail))
    if limit is not None and limit < 1:
        raise MifOpError("limit must be 1 or more, got %d. %s" % (limit, tail))
    if max_depth is not None and not max_depth > 0:
        raise MifOpError("maxDepth must be above 0, got %r. %s" % (max_depth, tail))
    if faces is not None and limit is not None:
        raise MifOpError("pass faces or limit, not both. %s" % tail)
    me = obj.data
    nf = len(me.polygons)
    if faces is not None:
        if not isinstance(faces, list) or not faces or any(isinstance(f, bool) or not isinstance(f, int)
                                                           or not 0 <= f < nf for f in faces):
            raise MifOpError("faces takes face indices 0-%d, got %r. %s" % (nf - 1, faces, tail))
    if not uv_name or not mask_name:
        raise MifOpError("uvMap and maskAttribute need names. %s" % tail)
    att = me.attributes.get(mask_name)
    if att is not None and (att.domain != "CORNER" or att.data_type != "FLOAT"):
        raise MifOpError("'%s' already has an attribute '%s' (%s on %s) - name the mask another way with "
                         "maskAttribute. %s" % (obj.name, mask_name, att.data_type, att.domain, tail))
    if me.uv_layers.get(uv_name) is None and len(me.uv_layers) >= 8:
        raise MifOpError("'%s' has 8 UV maps, Blender's limit - name an existing one with uvMap. %s" % (obj.name, tail))
    if mat_name is not None:
        if not wire:
            raise MifOpError("material names where the decal is mixed, and wire is false. %s" % tail)
        if bpy.data.materials.get(mat_name) is None:
            have = sorted(m.name for m in bpy.data.materials)[:25]
            raise MifOpError("no material named '%s'. Present: %s. %s" % (mat_name, ", ".join(have) or "(none)", tail))

    import math  # noqa: PLC0415
    from mathutils import Matrix, Vector   # noqa: PLC0415 - after every argument refusal
    f = Vector(direction).normalized()
    up_ref = Vector((0, 0, 1)) if abs(f.z) < 0.99 else Vector((0, 1, 0))
    right = f.cross(up_ref).normalized()
    up = right.cross(f).normalized()
    if rotation:
        rot = Matrix.Rotation(math.radians(rotation), 3, f)
        right, up = rot @ right, rot @ up
    o = Vector(origin)
    mw = obj.matrix_world
    nmat = mw.to_3x3().inverted_safe().transposed()
    co = [mw @ v.co for v in me.vertices]
    rows, back = [], 0
    allowed = set(faces) if faces is not None else None
    for p in me.polygons:
        if allowed is not None and p.index not in allowed:
            continue
        depth = (mw @ p.center - o).dot(f)
        if depth < 0 or (max_depth is not None and depth > max_depth):
            continue
        uvs = [((co[vi] - o).dot(right) / size[0] + 0.5, (co[vi] - o).dot(up) / size[1] + 0.5) for vi in p.vertices]
        if (max(u for u, _ in uvs) <= 0 or min(u for u, _ in uvs) >= 1 or max(v for _, v in uvs) <= 0
                or min(v for _, v in uvs) >= 1):
            continue
        facing = (nmat @ p.normal).dot(-f)
        if abs(facing) < 1e-6:
            continue
        if facing < 0 and wipe:
            back += 1
            continue
        rows.append((depth, p.index))
    rows.sort()
    skipped = 0
    if limit is not None and len(rows) > limit:
        skipped = len(rows) - limit
        rows = rows[:limit]
    covered = sorted(i for _, i in rows)
    if not covered:
        raise MifOpError("the decal hits no face of '%s' from %s along %s at size %s (%d turned away%s). %s"
                         % (obj.name, origin, direction, size, back, " and wiped" if wipe else "", tail))
    cov = set(covered)
    mats = []
    if wire:
        names = [mat_name] if mat_name is not None else sorted({
            obj.material_slots[me.polygons[i].material_index].material.name for i in covered
            if me.polygons[i].material_index < len(obj.material_slots)
            and obj.material_slots[me.polygons[i].material_index].material is not None})
        if not names:
            raise MifOpError("the covered faces carry no material to mix the decal into - pass material, or wire "
                             "false for the UV map and mask alone. %s" % tail)
        for n in names:
            m = bpy.data.materials[n]
            if not m.use_nodes or m.node_tree is None or not any(x.type == "BSDF_PRINCIPLED" for x in m.node_tree.nodes):
                raise MifOpError("material '%s' has no Principled BSDF to mix the decal into. %s" % (n, tail))
            mats.append(m)

    mask = [0.0] * len(me.loops)
    flat = [-1.0] * (2 * len(me.loops))
    for p in me.polygons:
        if p.index in cov:
            for li in p.loop_indices:
                c = co[me.loops[li].vertex_index] - o
                flat[2 * li], flat[2 * li + 1] = c.dot(right) / size[0] + 0.5, c.dot(up) / size[1] + 0.5
                mask[li] = 1.0
    # EVERY LAYER IS MADE BEFORE ANY IS WRITTEN, EACH LOOKED UP AGAIN BY NAME AND WRITTEN WITH foreach_set. Measured
    # 2026-10-06: the UV layer's .data fetched, then the mask attribute added, then per-loop writes - Python read the
    # values back, and a Cycles bake on 4.2.23, 4.4.0 and 5.0.1 shaded as if neither layer existed (5.2.2 did not).
    # Adding a layer reallocates the face-corner data; in this order every build bakes the decal.
    replaced = me.uv_layers.get(uv_name) is not None
    if not replaced:
        me.uv_layers.new(name=uv_name, do_init=False)
    if me.attributes.get(mask_name) is None:
        me.attributes.new(mask_name, "FLOAT", "CORNER")
    layer = me.uv_layers[uv_name]
    layer.data.foreach_set("uv", flat)
    me.attributes[mask_name].data.foreach_set("value", mask)
    me.update()
    wired = [_decal_wire(m, img, layer.name, mask_name) for m in mats]
    return {"object": obj.name, "image": img.name, "uvMap": layer.name, "uvMapReplaced": replaced,
            "maskAttribute": mask_name, "facesCovered": len(covered), "faces": covered[:200],
            "backFacesWiped": back, "skippedByLimit": skipped, "rightAxis": [round(c, 6) for c in right],
            "upAxis": [round(c, 6) for c in up], "wired": wired,
            "note": "faces outside the decal carry UV (-1, -1) and mask 0 on the decal layer"}


# --------------------------------------------------------------------------------------------- spray_text_art

_SPRAY_KEYS = ("text", "font", "name", "size", "margin", "outlineWidth", "highlightWidth", "softness", "overspray",
               "overspraySpread", "seed", "colors", "passes", "strokeWidth", "nozzleSamples", "decalSize",
               "minReadableArea", "revealKeys", "colorPath", "revealPath", "overwrite", "drips", "runPath", "shadow",
               "nozzle")
_SPRAY_GRADIENT_KEYS = ("top", "bottom", "mottle", "mottleSize")
_SPRAY_LAYERS = ("outline", "fill", "highlight")
_SPRAY_COLORS = {"fill": [0.8, 0.1, 0.05], "outline": [0.02, 0.02, 0.02], "highlight": [0.95, 0.95, 0.95]}
_SPRAY_PASSES = {"outline": [0.0, 0.4], "fill": [0.4, 0.85], "highlight": [0.85, 1.0]}


def _dilate(np, m, r):
    """A boolean mask grown by a disc of radius r pixels: per row offset, a horizontal run of the disc's half width."""
    if r <= 0:
        return m.copy()
    h, w = m.shape
    out = np.zeros_like(m)
    for dy in range(-r, r + 1):
        hw = int(math.floor(math.sqrt(r * r - dy * dy)))
        sh = np.zeros_like(m)
        if dy >= 0:
            sh[dy:, :] = m[:h - dy, :]
        else:
            sh[:dy, :] = m[-dy:, :]
        c = np.cumsum(np.pad(sh.astype(np.int32), ((0, 0), (hw + 1, hw)), mode="constant"), axis=1)
        out |= (c[:, 2 * hw + 1:] - c[:, :-2 * hw - 1]) > 0
    return out


def _spray_times(np, mask, window, sw):
    """Per-pixel time of a pass: a serpentine nozzle over the mask's box in columns sw wide, left to right, up the
    even columns and down the odd; (times, box, columns), times 1.0 outside the mask."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    t = np.ones((h, w), dtype=np.float32)
    if not len(xs):
        return t, None, 0
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    ncol = max(1, int(math.ceil((x1 - x0) / float(sw))))
    gy, gx = np.mgrid[0:h, 0:w]
    col = np.clip((gx - x0) // sw, 0, ncol - 1)
    f = np.clip((gy - y0 + 0.5) / float(y1 - y0), 0.0, 1.0)
    f = np.where(col % 2 == 1, 1.0 - f, f)
    tt = window[0] + (window[1] - window[0]) * (col + f) / ncol
    t[mask] = tt[mask]
    return t, (x0, y0, x1, y1), ncol


def _write_png16(np, path, px):
    """A 16-bit RGBA PNG of float pixels [h, w, 4] in 0-1 (row 0 the bottom, as Blender's), channels as given."""
    import struct   # noqa: PLC0415
    import zlib   # noqa: PLC0415
    h, w = px.shape[0], px.shape[1]
    vals = np.round(np.clip(px[::-1], 0.0, 1.0) * 65535.0).astype(">u2").reshape(h, w * 4)
    raw = b"".join(b"\x00" + row.tobytes() for row in vals)

    def chunk(tag, body):
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)
    data = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 16, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
    with open(path, "wb") as fh:
        fh.write(data)


_NOZZLE_KEYS = ("space", "path", "radius", "radiusPx", "coneAngle", "decal", "offset", "timeRange", "passes",
                "unreached")
_NOZZLE_SPACES = ("uv", "decal", "world")
_NOZZLE_REVEAL_MAX = 0.9998   # the decal material shows a pass where its time is at or under Reveal x 0.9999


def _nozzle_vec(v, n, where, tail):
    """[x, y, z] or {x, y, z} ({x, y, z, w} for n 4) as n finite floats, or a refusal."""
    if isinstance(v, dict):
        ks = "xyzw"[:n]
        if set(v) != set(ks):
            raise MifOpError("%s is [%s] or {%s}, got %r. %s" % (where, ", ".join(ks), ", ".join(ks), v, tail))
        v = [v[k] for k in ks]
    try:
        c = finite_floats(list(v) if isinstance(v, (list, tuple)) else [None], where)
    except MifOpError as exc:
        raise MifOpError("%s %s" % (str(exc).rstrip("."), tail))
    if len(c) != n:
        raise MifOpError("%s takes %d numbers, got %r. %s" % (where, n, v, tail))
    return c


def _nozzle_frame(fr, tail):
    """The decal's plane from nozzle.decal: (origin, right, up, normal), each a unit 3-vector but the origin.

    {origin, right, up}: the axes as given (up made square to right). {origin, normal, up (default [0, 0, 1]),
    axes}: normal points out of the wall toward the viewer, and right is the viewer's right looking at the wall -
    normal x up in Unreal's left-handed axes, up x normal in Blender's right-handed ones, so `axes` is required."""
    def unit(v, where):
        n = math.sqrt(sum(c * c for c in v))
        if n < 1e-9:
            raise MifOpError("nozzle.decal.%s is zero. %s" % (where, tail))
        return [c / n for c in v]

    def cross(a, b):
        return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]

    def dot(a, b):
        return sum(x * y for x, y in zip(a, b))
    if not isinstance(fr, dict) or "origin" not in fr or set(fr) - {"origin", "right", "up", "normal", "axes"}:
        raise MifOpError("nozzle.decal is {origin, right, up} or {origin, normal, up, axes: unreal|blender} - the decal's "
                         "centre on the wall and its axes, world space - got %r. %s" % (fr, tail))
    origin = _nozzle_vec(fr["origin"], 3, "nozzle.decal.origin", tail)
    up = unit(_nozzle_vec(fr.get("up", [0.0, 0.0, 1.0]), 3, "nozzle.decal.up", tail), "up")
    if "right" in fr:
        if "normal" in fr or "axes" in fr:
            raise MifOpError("nozzle.decal takes right and up, or normal, up and axes - not both. %s" % tail)
        right = unit(_nozzle_vec(fr["right"], 3, "nozzle.decal.right", tail), "right")
        up = unit([u - dot(up, right) * r for u, r in zip(up, right)], "up (parallel to right)")
        return origin, right, up, unit(cross(right, up), "normal")
    if "normal" not in fr or fr.get("axes") not in ("unreal", "blender"):
        raise MifOpError("nozzle.decal needs right, or normal with axes 'unreal' (left-handed: right = normal x up) or "
                         "'blender' (right-handed: right = up x normal), got %r. %s" % (fr, tail))
    nrm = unit(_nozzle_vec(fr["normal"], 3, "nozzle.decal.normal", tail), "normal")
    up = unit([u - dot(up, nrm) * c for u, c in zip(up, nrm)], "up (parallel to normal)")
    right = cross(nrm, up) if fr["axes"] == "unreal" else cross(up, nrm)
    return origin, unit(right, "right"), up, nrm


def _quat_rotate(q, v):
    """v turned by the unit quaternion q [x, y, z, w] - FQuat::RotateVector's formula, the same in either handedness."""
    qx, qy, qz, qw = q
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    tx, ty, tz = 2 * (qy * v[2] - qz * v[1]), 2 * (qz * v[0] - qx * v[2]), 2 * (qx * v[1] - qy * v[0])
    return [v[0] + qw * tx + (qy * tz - qz * ty), v[1] + qw * ty + (qz * tx - qx * tz),
            v[2] + qw * tz + (qx * ty - qy * tx)]


def _nozzle_parse(nz, w, h, decal, tail):
    """nozzle {...} -> the path in pixel coordinates (x right, y up, 0 at the image's lower left), times, per-point
    spray radius in the metric (decal units with decalSize, else pixels) and the per-layer segments; refusals name the
    fault and nothing has been made yet."""
    if not isinstance(nz, dict) or set(nz) - set(_NOZZLE_KEYS):
        raise MifOpError("nozzle is {%s}, got keys %r. %s" % (", ".join(_NOZZLE_KEYS),
                                                             sorted(nz) if isinstance(nz, dict) else nz, tail))
    space = nz.get("space", "uv")
    if space not in _NOZZLE_SPACES:
        raise MifOpError("nozzle.space is uv, decal or world, got %r. %s" % (space, tail))
    if space in ("decal", "world") and decal is None:
        raise MifOpError("nozzle.space %s is in the decal's world units, so it needs decalSize. %s" % (space, tail))
    path = nz.get("path")
    if not isinstance(path, (list, tuple)) or not 2 <= len(path) <= 4096:
        raise MifOpError("nozzle.path is 2-4096 timed points, got %s. %s"
                         % (len(path) if isinstance(path, (list, tuple)) else repr(path), tail))
    frame = None
    if space == "world":
        if "decal" not in nz:
            raise MifOpError("nozzle.space world needs nozzle.decal {origin, normal, up, axes} or {origin, right, up}. "
                             "%s" % tail)
        frame = _nozzle_frame(nz["decal"], tail)
    elif "decal" in nz or "offset" in nz or "coneAngle" in nz:
        raise MifOpError("nozzle.decal, offset and coneAngle are for world-space points (space world). %s" % tail)
    offset = _nozzle_vec(nz["offset"], 3, "nozzle.offset", tail) if "offset" in nz else None
    ts, xs, ys, dist = [], [], [], []
    for i, p in enumerate(path):
        where = "nozzle.path[%d]" % i
        if space == "world" and isinstance(p, dict):
            if set(p) - {"t", "location", "quat"} or "t" not in p or "location" not in p:
                raise MifOpError("%s is [t, x, y, z] or {t, location, quat} (read_component_pose's world location and "
                                 "quat), got %r. %s" % (where, p, tail))
            t = _nozzle_vec([p["t"]], 1, where + ".t", tail)[0]
            loc = _nozzle_vec(p["location"], 3, where + ".location", tail)
            if offset is not None:
                if "quat" not in p:
                    raise MifOpError("%s has no quat, and nozzle.offset is along the socket's own axes. %s"
                                     % (where, tail))
                q = _nozzle_vec(p["quat"], 4, where + ".quat", tail)
                if sum(c * c for c in q) < 1e-12:
                    raise MifOpError("%s.quat is zero. %s" % (where, tail))
                loc = [a + b for a, b in zip(loc, _quat_rotate(q, offset))]
            vals = [t] + loc
        else:
            if offset is not None:
                raise MifOpError("%s: nozzle.offset turns with each point's quat, so the points are {t, location, quat}. "
                                 "%s" % (where, tail))
            vals = _nozzle_vec(p, 4 if space == "world" else 3, where, tail)
        ts.append(vals[0])
        if space == "uv":
            xs.append(vals[1] * w)
            ys.append(vals[2] * h)
            dist.append(0.0)
        elif space == "decal":
            xs.append((vals[1] / decal[0] + 0.5) * w)
            ys.append((vals[2] / decal[1] + 0.5) * h)
            dist.append(0.0)
        else:
            o, right, up, nrm = frame
            d = [vals[1 + k] - o[k] for k in range(3)]
            xs.append((sum(a * b for a, b in zip(d, right)) / decal[0] + 0.5) * w)
            ys.append((sum(a * b for a, b in zip(d, up)) / decal[1] + 0.5) * h)
            dist.append(abs(sum(a * b for a, b in zip(d, nrm))))
    if any(b <= a for a, b in zip(ts, ts[1:])):
        raise MifOpError("nozzle.path's times must rise point to point. %s" % tail)
    sx, sy = (decal[0] / w, decal[1] / h) if decal is not None else (1.0, 1.0)
    if ("radius" in nz) == ("radiusPx" in nz):
        raise MifOpError("nozzle takes radius (decal units, needs decalSize) or radiusPx (pixels) - one of them. %s"
                         % tail)
    if "radius" in nz:
        if decal is None:
            raise MifOpError("nozzle.radius is in the decal's world units, so it needs decalSize; radiusPx is pixels. "
                             "%s" % tail)
        r0 = _nozzle_vec([nz["radius"]], 1, "nozzle.radius", tail)[0]
    else:
        r0 = _nozzle_vec([nz["radiusPx"]], 1, "nozzle.radiusPx", tail)[0] * (sx + sy) / 2.0
    if not r0 > 0:
        raise MifOpError("the spray radius must be above 0, got %r. %s" % (nz.get("radius", nz.get("radiusPx")), tail))
    cone = 0.0
    if "coneAngle" in nz:
        cone = _nozzle_vec([nz["coneAngle"]], 1, "nozzle.coneAngle", tail)[0]
        if not 0.0 <= cone < 170.0:
            raise MifOpError("nozzle.coneAngle is the spray's full angle, 0-170 degrees, got %r. %s" % (cone, tail))
    k = math.tan(math.radians(cone) / 2.0)
    radii = [r0 + d * k for d in dist]
    tr = nz.get("timeRange", [ts[0], ts[-1]])
    tr = _nozzle_vec(tr, 2, "nozzle.timeRange", tail)
    if not tr[1] > tr[0]:
        raise MifOpError("nozzle.timeRange is [start, end] with start before end, got %r. %s" % (tr, tail))
    segs = {}
    raw = nz.get("passes") or {}
    if not isinstance(raw, dict) or set(raw) - set(_SPRAY_LAYERS):
        raise MifOpError("nozzle.passes is {outline, fill, highlight}, each [from, to] in the path's time, got %r. %s"
                         % (raw, tail))
    for layer in _SPRAY_LAYERS:
        if layer not in raw:
            segs[layer] = [ts[0], ts[-1]]
            continue
        a, b = _nozzle_vec(raw[layer], 2, "nozzle.passes.%s" % layer, tail)
        a, b = max(a, ts[0]), min(b, ts[-1])
        if not b > a:
            raise MifOpError("nozzle.passes.%s %r holds no part of the path (its times run %g to %g). %s"
                             % (layer, raw[layer], ts[0], ts[-1], tail))
        segs[layer] = [a, b]
    unreached = nz.get("unreached", "leave")
    if unreached not in ("leave", "nearest"):
        raise MifOpError("nozzle.unreached is 'leave' (unpainted) or 'nearest' (painted at the time the nozzle came "
                         "closest), got %r. %s" % (unreached, tail))
    return {"space": space, "t": ts, "x": xs, "y": ys, "r": radii, "sx": sx, "sy": sy, "radius": r0, "cone": cone,
            "timeRange": tr, "segments": segs, "unreached": unreached, "frame": frame,
            "units": "decal units" if decal is not None else "pixels"}


def _nozzle_clip(np, T, X, Y, R, a, b):
    """The path between times a and b, its ends interpolated: (T, X, Y, R) arrays in the metric."""
    inside = (T > a) & (T < b)

    def at(t):
        return [float(np.interp(t, T, arr)) for arr in (X, Y, R)]
    pa, pb = at(a), at(b)
    return (np.concatenate([[a], T[inside], [b]]), np.concatenate([[pa[0]], X[inside], [pb[0]]]),
            np.concatenate([[pa[1]], Y[inside], [pb[1]]]), np.concatenate([[pa[2]], R[inside], [pb[2]]]))


def _nozzle_times(np, mask, T, X, Y, R, sx, sy, unreached):
    """Per pixel of mask: the first path time the nozzle came within its spray radius - every segment's spray a disc
    whose radius runs linearly between its ends, the first s in [0, 1] where |P0 + s(P1 - P0) - q| <= r0 + s(r1 - r0),
    the smallest root of that quadratic once the pixel is outside at s 0. Returns (times, inf where never; reached
    mask; filled mask - the unreached pixels given the time of the nozzle's closest pass when unreached is nearest)."""
    h, w = mask.shape
    reach = np.full((h, w), np.inf)
    for i in range(len(T) - 1):
        x0, y0, x1, y1, r0, r1 = X[i], Y[i], X[i + 1], Y[i + 1], R[i], R[i + 1]
        rm = max(r0, r1)
        xa = max(0, int(math.floor((min(x0, x1) - rm) / sx - 1)))
        xb = min(w, int(math.ceil((max(x0, x1) + rm) / sx + 1)))
        ya = max(0, int(math.floor((min(y0, y1) - rm) / sy - 1)))
        yb = min(h, int(math.ceil((max(y0, y1) + rm) / sy + 1)))
        if xa >= xb or ya >= yb:
            continue
        sub = mask[ya:yb, xa:xb] & np.isinf(reach[ya:yb, xa:xb])
        if not sub.any():
            continue
        yy, xx = np.nonzero(sub)
        dx = x0 - (xx + xa + 0.5) * sx
        dy = y0 - (yy + ya + 0.5) * sy
        ex, ey, dr = x1 - x0, y1 - y0, r1 - r0
        qa = ex * ex + ey * ey - dr * dr
        qb = 2.0 * (dx * ex + dy * ey - r0 * dr)
        qc = dx * dx + dy * dy - r0 * r0
        s = np.full(len(dx), np.nan)
        if abs(qa) < 1e-12:
            with np.errstate(divide="ignore", invalid="ignore"):
                lin = -qc / qb
            ok = (qb < 0) & (lin >= 0) & (lin <= 1)
            s[ok] = lin[ok]
        else:
            disc = qb * qb - 4.0 * qa * qc
            sq = np.sqrt(np.maximum(disc, 0.0))
            s1, s2 = (-qb - sq) / (2.0 * qa), (-qb + sq) / (2.0 * qa)
            lo, hi = np.minimum(s1, s2), np.maximum(s1, s2)
            s = np.where((disc >= 0) & (lo >= 0) & (lo <= 1), lo,
                         np.where((disc >= 0) & (hi >= 0) & (hi <= 1), hi, np.nan))
        s[qc <= 0] = 0.0
        ok = ~np.isnan(s)
        reach[ya + yy[ok], xa + xx[ok]] = T[i] + s[ok] * (T[i + 1] - T[i])
    reached = mask & ~np.isinf(reach)
    filled = np.zeros_like(mask)
    if unreached == "nearest":
        uy, ux = np.nonzero(mask & ~reached)
        if len(ux):
            qx, qy = (ux + 0.5) * sx, (uy + 0.5) * sy
            best = np.full(len(ux), np.inf)
            bt = np.zeros(len(ux))
            for i in range(len(T) - 1):
                ex, ey = X[i + 1] - X[i], Y[i + 1] - Y[i]
                dx, dy = qx - X[i], qy - Y[i]
                ee = ex * ex + ey * ey
                s = np.clip((dx * ex + dy * ey) / ee, 0.0, 1.0) if ee > 0 else np.zeros(len(ux))
                d2 = (dx - s * ex) ** 2 + (dy - s * ey) ** 2
                better = d2 < best
                best[better] = d2[better]
                bt[better] = T[i] + s[better] * (T[i + 1] - T[i])
            reach[uy, ux] = bt
            filled[uy, ux] = True
    return reach, reached, filled


def _nozzle_front(np, times, T, X, Y, R, sx, sy):
    """Where the paint appears against where the nozzle is at that moment, over the pixels painted (times finite):
    distance, and lead (+, the paint ahead of the nozzle along its motion) / lag (-, behind it), in the metric."""
    py, px = np.nonzero(np.isfinite(times))
    if not len(px):
        return None
    tp = times[py, px]
    qx, qy = (px + 0.5) * sx, (py + 0.5) * sy
    nx, ny, nr = np.interp(tp, T, X), np.interp(tp, T, Y), np.interp(tp, T, R)
    i = np.clip(np.searchsorted(T, tp, side="right") - 1, 0, len(T) - 2)
    ex, ey = X[i + 1] - X[i], Y[i + 1] - Y[i]
    el = np.hypot(ex, ey)
    dx, dy = qx - nx, qy - ny
    dist = np.hypot(dx, dy)
    with np.errstate(divide="ignore", invalid="ignore"):
        along = np.where(el > 0, (dx * ex + dy * ey) / el, 0.0)
    return {"maxDistance": round(float(dist.max()), 6), "p95Distance": round(float(np.percentile(dist, 95)), 6),
            "meanDistance": round(float(dist.mean()), 6), "maxLead": round(float(max(0.0, along.max())), 6),
            "maxLag": round(float(max(0.0, -along.min())), 6),
            "maxBeyondRadius": round(float(max(0.0, (dist - nr).max())), 6)}


def _nozzle_paint(np, nzp, layers):
    """Each layer's reveal-time map from the nozzle path (1.0 where it never paints) and the report."""
    T = np.asarray(nzp["t"], dtype=np.float64)
    X = np.asarray(nzp["x"], dtype=np.float64) * nzp["sx"]
    Y = np.asarray(nzp["y"], dtype=np.float64) * nzp["sy"]
    R = np.asarray(nzp["r"], dtype=np.float64)
    t0, t1 = nzp["timeRange"]
    maps, rep, fronts = {}, {}, []
    union = np.zeros_like(layers["fill"])
    union_reached = np.zeros_like(union)
    for layer in _SPRAY_LAYERS:
        mask = layers[layer]
        a, b = nzp["segments"][layer]
        ct, cx, cy, cr = _nozzle_clip(np, T, X, Y, R, a, b)
        reach, reached, filled = _nozzle_times(np, mask, ct, cx, cy, cr, nzp["sx"], nzp["sy"], nzp["unreached"])
        rev = np.ones(mask.shape, dtype=np.float32)
        live = np.isfinite(reach)
        rev[live] = np.clip((reach[live] - t0) / (t1 - t0), 0.0, _NOZZLE_REVEAL_MAX)
        maps[layer] = rev
        n = int(np.count_nonzero(mask))
        nr_ = int(np.count_nonzero(reached))
        front = _nozzle_front(np, reach, ct, cx, cy, cr, nzp["sx"], nzp["sy"])
        if front:
            fronts.append(front)
        rep[layer] = {"segment": [round(a, 6), round(b, 6)], "pixels": n, "reached": nr_, "unreached": n - nr_,
                      "filledNearest": int(np.count_nonzero(filled)),
                      "coverage": round(100.0 * nr_ / n, 4) if n else None, "front": front}
        union |= mask
        union_reached |= reached
    miss = union & ~union_reached
    my, mx = np.nonzero(miss)
    front = None
    if fronts:
        front = {k: max(f[k] for f in fronts) for k in ("maxDistance", "p95Distance", "maxLead", "maxLag",
                                                        "maxBeyondRadius")}
    out = {"space": nzp["space"], "points": len(nzp["t"]), "units": nzp["units"], "radius": round(nzp["radius"], 6),
           "coneAngle": nzp["cone"], "timeRange": [round(t0, 6), round(t1, 6)], "unreachedMode": nzp["unreached"],
           "coverage": rep["fill"]["coverage"], "layers": rep,
           "unreachedPixels": int(len(mx)), "unreachedSample": [[int(x), int(y)] for x, y in zip(mx[:20], my[:20])],
           "front": front,
           "note": "coverage is the share (%%) of the letters' fill pixels the path reaches; front compares where each "
                   "pixel's paint appears with where the nozzle is at that moment, in %s - lead ahead of the nozzle's "
                   "motion, lag behind it" % nzp["units"]}
    if nzp["frame"] is not None:
        o, right, up, nrm = nzp["frame"]
        out["decalFrame"] = {"origin": o, "right": [round(c, 6) for c in right], "up": [round(c, 6) for c in up],
                             "normal": [round(c, 6) for c in nrm]}
    return maps, out


def op_spray_text_art(params):
    """Text in a font -> a spray-paint colour texture and a reveal map with the outline, fill and highlight passes
    trailing a nozzle, and the first reveal at which the paint covers a readable area.

    THE GAP (spec "A spray / decal-art tool", its Blender / offline half, with the merged "A visibility check for
    timed reveals has to count what reads"): text art for a decal - a colour texture and a position-based reveal map -
    was made in a one-off script, and a reveal's timing was read from the first frame any paint showed (11 cm2) while
    nothing read until about 60 cm2 on a dark wall at 4 m.

    text (1-200 characters, newlines allowed) in `font` (a .ttf / .otf / .pfb path; default Blender's built-in font,
    named in the reply) is fitted into size [w, h] pixels (default [1024, 512]) inside `margin` pixels (default
    outlineWidth + overspraySpread + 2). Layers, as masks: fill = the glyphs; outline = the fill grown by a disc of
    outlineWidth px (default 3% of h); highlight = the fill's pixels whose neighbour highlightWidth px (default 1.2% of
    h) up and left is not fill - its upper-left edge; overspray (0-1, default 0.25) speckles the band overspraySpread px
    (default outlineWidth) past the outline with that share halved, by `seed`. colors {fill, outline, highlight} are
    sRGB 0-1 as picked (written as is into the colour texture, an 8-bit RGBA image, alpha the paint, softened by
    `softness` px of blur). The reveal map (float, Non-Color, written as a 16-bit PNG) holds each pass's time in R
    (outline, speckles included), G (fill), B (highlight) and the first paint in A, 1.0 where nothing paints: a pass
    runs over its window `passes` {outline: [0, 0.4], fill: [0.4, 0.85], highlight: [0.85, 1]} as a nozzle sweeping
    its layer's box in columns strokeWidth px wide (default 2 x outlineWidth), left to right, up then down - a
    material shows a pass where its time is at or under the Reveal parameter. nozzlePath gives nozzleSamples (default
    32) points per pass, {layer, t, u, v} in the image's 0-1 space, for keying a moving nozzle.
    Drips (2026-10-07): drips {count, length, width (default half outlineWidth), seed} runs paint down from the fill's
    lower edge - count starts picked by seed, each length x 0.5-1 px long and width wide, in the fill colour in the
    colour texture - and runPath (.png, 16-bit) holds each drip pixel's progress in R, 0 where it leaves the paint to 1
    at its tip, 1.0 elsewhere: a material shows a drip where that is at or under its RunDown parameter.
    Gradient, shadow, drip colour (2026-10-08): colors.fill may be a gradient {top, bottom, mottle (0-0.5, default 0),
    mottleSize (px, default 1.2% of h)} - top to bottom over the fill's own rows, each pixel times (1 + mottle x a
    smooth unit-variance noise about mottleSize px across, by seed). shadow {color, offset [right, down] px, opacity
    (default 1)} paints the outline (speckles included) moved by offset, behind it, at that opacity, revealed with the
    outline pass: each shadow pixel holds, in R, the time of the outline pixel it falls from; the default margin grows
    to keep it in the image and the reply counts what fell outside (shadow.clipped). drips.color sets the drips' colour,
    which is otherwise the fill's (its top colour for a gradient).
    Nozzle path (2026-10-08): nozzle {path, radius | radiusPx, space, decal, offset, coneAngle, timeRange, passes,
    unreached} replaces the column sweep - each pixel's time is the FIRST moment the nozzle passed within the spray radius
    of it, so the paint appears where the nozzle points, stroke by stroke. path is 2-4096 points with rising times (any
    unit: 0-1, seconds, frames): space uv [t, u, v] (the image's 0-1, v up), decal [t, x, y] (decal units from its
    centre, x right, y up; needs decalSize) or world [t, x, y, z] | {t, location, quat} (needs decalSize and decal
    {origin, normal, up (default world Z), axes 'unreal' | 'blender'} or {origin, right, up}: the decal's centre and
    axes, normal out of the wall; right is the viewer's right). offset [x, y, z] moves each world point along its own
    quat - nozzle = socket location + quat x offset, for a hand socket and a can. radius is decal units (needs
    decalSize), radiusPx pixels; coneAngle (world, full angle in degrees) grows it by the distance off the wall x
    tan(angle / 2). timeRange [start, end] (default the path's first and last time) maps a time to the reveal value 0-1
    (clipped to 0.9998, under the decal material's 0.9999 cut). passes {outline, fill, highlight: [from, to]} gives a
    pass its own stretch of the path's time; a pass not named shares the whole path. unreached 'leave' (default) leaves
    a pixel the nozzle never reaches unpainted (1.0, still in the colour map) or 'nearest' paints it at the time of the
    nozzle's closest pass. The reply's nozzle {coverage (% of the fill the path reaches), layers {pixels, reached,
    unreached, filledNearest, coverage, front}, unreachedPixels, unreachedSample, front {maxDistance, p95Distance,
    maxLead, maxLag, maxBeyondRadius}} - front is where each pixel's paint appears against the nozzle at that moment
    (lead ahead of its motion, lag behind) - and nozzlePath echoes the path {t (reveal), time, u, v}. Refused with
    nozzle: passes, strokeWidth, nozzleSamples (the sweep's), a world point without decal, falling times.
    Readable: with decalSize [width, height] (world units of the placed decal) and minReadableArea, firstReveal is the
    reveal at which the painted pixels first cover that area (the N-th earliest first paint, N = ceil(area / pixel
    area)); revealKeys [[frame, value], ...] (value rising) put it on a frame by linear interpolation. Writes
    colorPath and revealPath (.png) when given. Refused, nothing made or written: empty text, size outside 16-4096, a
    pass window not 0 <= from <= to <= 1, a negative width, a path not .png, a font file that does not exist or will
    not load, minReadableArea without decalSize, the text box too small for the margins, a gradient without top and
    bottom or with mottle outside 0-0.5, a shadow without color or offset or with opacity outside (0, 1].
    """
    reject_unknown(params, _SPRAY_KEYS, "spray_text_art")
    tail = "NOTHING was made or written."
    text = take(params, "text", required=True, kind=str)
    if not isinstance(text, str) or not text.strip() or len(text) > 200:
        raise MifOpError("text is 1-200 characters, not blank, got %r. %s" % (text, tail))
    size = params.get("size", [1024, 512])
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise MifOpError("size is [width, height] in pixels, each 16-4096, got %r. %s" % (size, tail))
    try:
        w, h = finite_int(size[0], "size[0]"), finite_int(size[1], "size[1]")
    except MifOpError as exc:
        raise MifOpError("%s %s" % (str(exc).rstrip("."), tail))
    if not (16 <= w <= 4096 and 16 <= h <= 4096):
        raise MifOpError("size is [width, height] in pixels, each 16-4096, got %r. %s" % (size, tail))
    try:
        ow = take_int(params, "outlineWidth", default=max(1, int(round(0.03 * h))))
        hw = take_int(params, "highlightWidth", default=max(1, int(round(0.012 * h))))
        spread = take_int(params, "overspraySpread", default=None)
        soft = take_float(params, "softness", default=1.5)
        spray = take_float(params, "overspray", default=0.25)
        seed = take_int(params, "seed", default=0)
        samples = take_int(params, "nozzleSamples", default=32)
        sw = take_int(params, "strokeWidth", default=None)
        margin = take_int(params, "margin", default=None)
        min_area = take_float(params, "minReadableArea", default=None)
    except MifOpError as exc:
        raise MifOpError("%s %s" % (str(exc).rstrip("."), tail))
    spread = ow if spread is None else spread
    sw = max(1, 2 * ow) if sw is None else sw
    for key, val, lo, hi in (("outlineWidth", ow, 0, 512), ("highlightWidth", hw, 0, 512),
                             ("overspraySpread", spread, 0, 512), ("strokeWidth", sw, 1, 4096),
                             ("nozzleSamples", samples, 2, 4096)):
        if not lo <= val <= hi:
            raise MifOpError("%s must be %d-%d pixels, got %r. %s" % (key, lo, hi, val, tail))
    if not (math.isfinite(soft) and 0.0 <= soft <= 64.0):
        raise MifOpError("softness is 0-64 pixels of blur, got %r. %s" % (soft, tail))
    if not (math.isfinite(spray) and 0.0 <= spray <= 1.0):
        raise MifOpError("overspray is 0-1, got %r. %s" % (spray, tail))
    margin = ow + spread + 2 if margin is None else margin
    if margin < 0:
        raise MifOpError("margin must be 0 or more pixels, got %r. %s" % (margin, tail))
    colors = dict(_SPRAY_COLORS)
    raw = params.get("colors") or {}
    if not isinstance(raw, dict) or set(raw) - set(_SPRAY_LAYERS):
        raise MifOpError("colors is {fill, outline, highlight}, each [r, g, b] sRGB 0-1, got %r. %s" % (raw, tail))
    def _rgb(v, where):
        c = finite_floats(list(v) if isinstance(v, (list, tuple)) else [], where)
        if len(c) not in (3, 4) or not all(0.0 <= x <= 1.0 for x in c):
            raise MifOpError("%s is [r, g, b] sRGB 0-1, got %r. %s" % (where, v, tail))
        return c[:3]
    grad = None
    for k, v in raw.items():
        if k == "fill" and isinstance(v, dict):
            if set(v) - set(_SPRAY_GRADIENT_KEYS) or "top" not in v or "bottom" not in v:
                raise MifOpError("colors.fill is [r, g, b] or a gradient {top, bottom, mottle, mottleSize} with top "
                                 "and bottom each [r, g, b] sRGB 0-1, got %r. %s" % (v, tail))
            try:
                mot = float(v.get("mottle", 0.0))
                msz = finite_int(v.get("mottleSize", max(1, int(round(0.012 * h)))), "colors.fill.mottleSize")
            except (TypeError, ValueError, MifOpError):
                raise MifOpError("colors.fill.mottle is 0-0.5 and mottleSize 1-512 pixels, got %r and %r. %s"
                                 % (v.get("mottle"), v.get("mottleSize"), tail))
            if not (math.isfinite(mot) and 0.0 <= mot <= 0.5 and 1 <= msz <= 512):
                raise MifOpError("colors.fill.mottle is 0-0.5 and mottleSize 1-512 pixels, got %r and %r. %s"
                                 % (v.get("mottle"), v.get("mottleSize"), tail))
            grad = {"top": _rgb(v["top"], "colors.fill.top"), "bottom": _rgb(v["bottom"], "colors.fill.bottom"),
                    "mottle": mot, "mottleSize": msz}
            colors[k] = grad
        else:
            colors[k] = _rgb(v, "colors.%s" % k)
    fill_top = grad["top"] if grad else colors["fill"]
    shadow = params.get("shadow")
    if shadow is not None:
        if (not isinstance(shadow, dict) or set(shadow) - {"color", "offset", "opacity"} or "color" not in shadow
                or "offset" not in shadow):
            raise MifOpError("shadow is {color [r, g, b], offset [right, down] px, opacity 0-1}, color and offset "
                             "required, got %r. %s" % (shadow, tail))
        s_off = shadow["offset"]
        if not isinstance(s_off, (list, tuple)) or len(s_off) != 2:
            raise MifOpError("shadow.offset is [right, down] in pixels, got %r. %s" % (s_off, tail))
        try:
            s_dx, s_dy = finite_int(s_off[0], "shadow.offset[0]"), finite_int(s_off[1], "shadow.offset[1]")
        except MifOpError as exc:
            raise MifOpError("%s %s" % (str(exc).rstrip("."), tail))
        if max(abs(s_dx), abs(s_dy)) > 1024:
            raise MifOpError("shadow.offset is at most 1024 pixels each way, got %r. %s" % (s_off, tail))
        s_op = shadow.get("opacity", 1.0)
        if isinstance(s_op, bool) or not isinstance(s_op, (int, float)) or not (math.isfinite(s_op)
                                                                               and 0.0 < s_op <= 1.0):
            raise MifOpError("shadow.opacity is above 0 and at most 1, got %r. %s" % (s_op, tail))
        shadow = {"color": _rgb(shadow["color"], "shadow.color"), "offset": [s_dx, s_dy], "opacity": float(s_op)}
        if params.get("margin") is None:        # the default margin keeps the moved outline inside the image too
            margin = max(margin, ow + max(abs(s_dx), abs(s_dy)) + 2)
    passes = {k: list(v) for k, v in _SPRAY_PASSES.items()}
    raw = params.get("passes") or {}
    if not isinstance(raw, dict) or set(raw) - set(_SPRAY_LAYERS):
        raise MifOpError("passes is {outline, fill, highlight}, each [from, to], got %r. %s" % (raw, tail))
    for k, v in raw.items():
        win = finite_floats(list(v) if isinstance(v, (list, tuple)) else [], "passes.%s" % k)
        if len(win) != 2 or not 0.0 <= win[0] <= win[1] <= 1.0:
            raise MifOpError("passes.%s is [from, to] with 0 <= from <= to <= 1, got %r. %s" % (k, v, tail))
        passes[k] = win
    decal = params.get("decalSize")
    if decal is not None:
        decal = finite_floats(list(decal) if isinstance(decal, (list, tuple)) else [], "decalSize")
        if len(decal) != 2 or min(decal) <= 0:
            raise MifOpError("decalSize is [width, height] above 0 (world units), got %r. %s"
                             % (params.get("decalSize"), tail))
    if min_area is not None and decal is None:
        raise MifOpError("minReadableArea is in the decal's world units squared, so it needs decalSize. %s" % tail)
    nzp = None
    if params.get("nozzle") is not None:
        clash = [k for k in ("passes", "strokeWidth", "nozzleSamples") if params.get(k) is not None]
        if clash:
            raise MifOpError("%s shape the column sweep, and nozzle replaces it - the path's own times and its "
                             "nozzle.passes segments time each pass. %s" % (" and ".join(clash), tail))
        nzp = _nozzle_parse(params["nozzle"], w, h, decal, tail)
    if min_area is not None and not (math.isfinite(min_area) and min_area > 0):
        raise MifOpError("minReadableArea must be above 0, got %r. %s" % (min_area, tail))
    keys = params.get("revealKeys")
    if keys is not None:
        if min_area is None:
            raise MifOpError("revealKeys put firstReveal on a frame, so they need minReadableArea. %s" % tail)
        if not isinstance(keys, (list, tuple)) or len(keys) < 2:
            raise MifOpError("revealKeys is [[frame, value], ...], 2 or more, got %r. %s" % (keys, tail))
        keys = [finite_floats(list(k) if isinstance(k, (list, tuple)) else [], "revealKeys") for k in keys]
        if any(len(k) != 2 for k in keys) or any(b[0] <= a[0] or b[1] < a[1] for a, b in zip(keys, keys[1:])):
            raise MifOpError("revealKeys is [[frame, value], ...] with frames rising and values not falling. %s" % tail)
    drips = params.get("drips")
    if drips is not None:
        if (not isinstance(drips, dict) or set(drips) - {"count", "length", "width", "seed", "color"}
                or "count" not in drips or "length" not in drips):
            raise MifOpError("drips is {count, length (px), width (px), seed, color}, got %r. %s" % (drips, tail))
        d_color = _rgb(drips["color"], "drips.color") if "color" in drips else fill_top
        try:
            d_count = finite_int(drips["count"], "drips.count")
            d_len = finite_int(drips["length"], "drips.length")
            d_w = finite_int(drips.get("width", max(1, ow // 2)), "drips.width")
            d_seed = finite_int(drips.get("seed", 0), "drips.seed")
        except MifOpError as exc:
            raise MifOpError("%s %s" % (str(exc).rstrip("."), tail))
        if not (1 <= d_count <= 200 and 1 <= d_len <= 4096 and 1 <= d_w <= 64):
            raise MifOpError("drips: count 1-200, length 1-4096 px, width 1-64 px, got %d, %d, %d. %s"
                             % (d_count, d_len, d_w, tail))
    if params.get("runPath") is not None and drips is None:
        raise MifOpError("runPath writes the drips' run map, so it needs drips. %s" % tail)
    paths = {}
    for key in ("colorPath", "revealPath", "runPath"):
        rawp = take(params, key, default=None, kind=str)
        if rawp is None:
            continue
        resolved = os.path.abspath(bpy.path.abspath(str(rawp)))
        check_output_path(rawp, resolved, "made or written")
        if not resolved.lower().endswith(".png"):
            raise MifOpError("%s must end in .png, got %r. %s" % (key, rawp, tail))
        paths[key] = resolved
    overwrite = take_bool(params, "overwrite", default=True)
    if not overwrite:
        for key, p in paths.items():
            if os.path.exists(p):
                raise MifOpError("%s '%s' exists and overwrite is false. %s" % (key, p, tail))
    font_path = take(params, "font", default=None, kind=str)
    if font_path is not None and not os.path.isfile(bpy.path.abspath(str(font_path))):
        raise MifOpError("font '%s' is not a file. %s" % (font_path, tail))
    base = str(take(params, "name", default="SprayText", kind=str))

    import numpy as np   # noqa: PLC0415 - after every refusal
    # ---- the glyphs: a temporary text object, evaluated (its fill is triangles), everything it made removed
    fonts0 = set(bpy.data.fonts)
    cu = bpy.data.curves.new("_mif_spray_text", "FONT")
    ob = None
    tris = []
    font_name = font_file = None
    try:
        if font_path is not None:
            try:
                cu.font = bpy.data.fonts.load(bpy.path.abspath(str(font_path)), check_existing=True)
            except (RuntimeError, OSError) as exc:
                raise MifOpError("font '%s' did not load: %s. %s" % (font_path, exc, tail))
        cu.body = text
        cu.align_x = "CENTER"
        ob = bpy.data.objects.new("_mif_spray_text", cu)
        bpy.context.scene.collection.objects.link(ob)
        dg = bpy.context.evaluated_depsgraph_get()
        ev = ob.evaluated_get(dg)
        me = ev.to_mesh()
        try:
            co = [(v.co.x, v.co.y) for v in me.vertices]
            for p in me.polygons:
                vs = list(p.vertices)
                for k in range(1, len(vs) - 1):
                    tris.append((co[vs[0]], co[vs[k]], co[vs[k + 1]]))
        finally:
            ev.to_mesh_clear()
        font_name = cu.font.name if cu.font else None
        font_file = cu.font.filepath if cu.font else None
    finally:
        if ob is not None:
            remove_objects([ob])
        bpy.data.curves.remove(cu)
        for f in set(bpy.data.fonts) - fonts0:
            if not f.users:
                bpy.data.fonts.remove(f)
    if not tris:
        raise MifOpError("the text %r has no filled glyphs in font %s (only spaces, or glyphs the font lacks). %s"
                         % (text, font_name, tail))
    xs = [p[0] for t in tris for p in t]
    ys = [p[1] for t in tris for p in t]
    bx0, bx1, by0, by1 = min(xs), max(xs), min(ys), max(ys)
    aw, ah = w - 2 * margin, h - 2 * margin
    if aw < 4 or ah < 4 or bx1 - bx0 <= 0 or by1 - by0 <= 0:
        raise MifOpError("size %dx%d leaves %dx%d pixels inside margin %d - no room for the text. %s"
                         % (w, h, aw, ah, margin, tail))
    sc = min(aw / (bx1 - bx0), ah / (by1 - by0))
    ox = w / 2.0 - sc * (bx0 + bx1) / 2.0
    oy = h / 2.0 - sc * (by0 + by1) / 2.0
    uvs = np.array([[[(ox + sc * x) / w, (oy + sc * y) / h] for x, y in t] for t in tris], dtype=np.float64)
    fill = _paint_raster(np, w, h, uvs, np.ones((len(tris), 3))) > 0.5
    outline = _dilate(np, fill, ow) | fill
    nb = np.zeros_like(fill)                    # nb[y, x] = fill[y + hw, x - hw]: the neighbour up and left
    if hw > 0:
        nb[:h - hw, hw:] = fill[hw:, :w - hw]
        highlight = fill & ~nb
    else:
        highlight = np.zeros_like(fill)
    rng = np.random.RandomState(seed & 0xFFFFFFFF)
    band = _dilate(np, fill, ow + spread) & ~outline if spray > 0 and spread > 0 else np.zeros_like(fill)
    speck = band & (rng.random_sample((h, w)) < spray * 0.5)
    painted_r = outline | speck
    nz_out = None
    if nzp is not None:
        # THE NOZZLE PATH: each pixel's time is the first moment the nozzle came within its spray radius - the paint
        # appears where it points, not on a column sweep (which led or lagged a hand by up to half a metre)
        nz_maps, nz_out = _nozzle_paint(np, nzp, {"outline": painted_r, "fill": fill, "highlight": highlight})
        t_out, t_fill, t_hil = nz_maps["outline"], nz_maps["fill"], nz_maps["highlight"]
        box_o = box_f = box_h = None
        ncol_o = ncol_f = ncol_h = 0
    else:
        t_out, box_o, ncol_o = _spray_times(np, painted_r, passes["outline"], sw)
        t_fill, box_f, ncol_f = _spray_times(np, fill, passes["fill"], sw)
        t_hil, box_h, ncol_h = _spray_times(np, highlight, passes["highlight"], sw)
    # ---- the drop shadow: the outline (speckles included) moved offset [right, down], behind it; each shadow pixel is
    # revealed with the outline pixel it falls from, in R. Image row 0 is the BOTTOM, so down is -y.
    sh_mask = np.zeros_like(fill)
    sh_clipped = 0
    if shadow is not None:
        dx, dy = shadow["offset"]
        moved = np.zeros_like(fill)
        moved_t = np.ones((h, w), dtype=np.float32)
        y0, y1, x0, x1 = max(0, -dy), min(h, h - dy), max(0, dx), min(w, w + dx)
        if y0 < y1 and x0 < x1:                          # moved[y, x] = painted_r[y + dy, x - dx]
            moved[y0:y1, x0:x1] = painted_r[y0 + dy:y1 + dy, x0 - dx:x1 - dx]
            moved_t[y0:y1, x0:x1] = t_out[y0 + dy:y1 + dy, x0 - dx:x1 - dx]
        sh_clipped = int(np.count_nonzero(painted_r)) - int(np.count_nonzero(moved))
        sh_mask = moved & ~painted_r
        t_out = np.where(sh_mask, moved_t, t_out).astype(np.float32)
    first = np.minimum(np.minimum(t_out, t_fill), t_hil)
    # ---- drips: run-downs from the fill's lower edge, each `length` x (0.5-1) px long and `width` wide; run holds each
    # drip pixel's progress 0 (where it leaves the paint) to 1 (its tip), 1.0 elsewhere - a material shows a drip where
    # run is at or under its RunDown parameter. Image row 0 is the BOTTOM (Blender's), so down is -y.
    run = np.ones((h, w), dtype=np.float32)
    drip_mask = np.zeros_like(fill)
    drip_rows = []
    if drips is not None:
        below = np.zeros_like(fill)
        below[1:, :] = fill[:-1, :]                          # below[y, x] = fill[y - 1, x]
        edge = fill & ~below                                 # fill pixels with nothing under them
        ys, xs = np.nonzero(edge)
        drng = np.random.RandomState(d_seed & 0xFFFFFFFF)
        if len(xs):
            pick = drng.choice(len(xs), size=min(d_count, len(xs)), replace=False)
            for k in sorted(pick, key=lambda i: xs[i]):
                x, y = int(xs[k]), int(ys[k])
                length = max(1, int(round(d_len * (0.5 + 0.5 * drng.random_sample()))))
                x0, x1 = max(0, x - d_w // 2), min(w, x - d_w // 2 + d_w)
                y_end = max(0, y - length)
                for yy in range(y - 1, y_end - 1, -1):
                    prog = (y - yy - 0.5) / float(length)    # pixel centres: the tip stays under 1.0, which is "none"
                    seg = run[yy, x0:x1]
                    np.minimum(seg, prog, out=seg)
                    drip_mask[yy, x0:x1] = True
                drip_rows.append({"x": x, "y": y, "length": length, "clipped": y - length < 0})
        drip_mask &= ~fill
        run[fill] = 1.0
    # ---- the colour texture: outline (and speckles) under the fill under the highlight; alpha the paint, softened
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    if shadow is not None:
        rgb[sh_mask] = shadow["color"]
    for layer, m in (("outline", painted_r), ("fill", fill), ("highlight", highlight)):
        if layer == "fill" and grad is not None:
            # top to bottom over the fill's own rows (row 0 is the bottom), times (1 + mottle x smooth unit noise)
            rows = np.nonzero(fill.any(axis=1))[0]
            ytop, ybot = int(rows.max()), int(rows.min())
            f = np.clip((ytop - np.arange(h, dtype=np.float32)) / float(max(1, ytop - ybot)), 0.0, 1.0)[:, None]
            g = (np.asarray(grad["top"], np.float32)[None, None, :] * (1.0 - f[..., None])
                 + np.asarray(grad["bottom"], np.float32)[None, None, :] * f[..., None])
            g = np.broadcast_to(g, (h, w, 3)).copy()
            if grad["mottle"] > 0:
                mrng = np.random.RandomState((seed ^ 0x5A17) & 0xFFFFFFFF)
                n = _box_blur(np, mrng.random_sample((h, w)).astype(np.float32), grad["mottleSize"])
                n = (n - n.mean()) / (float(n.std()) + 1e-6)
                g *= (1.0 + grad["mottle"] * n)[..., None]
            rgb[m] = np.clip(g, 0.0, 1.0)[m]
        else:
            rgb[m] = colors[layer]
    rgb[drip_mask] = d_color if drips is not None else fill_top
    alpha = (painted_r | drip_mask).astype(np.float32)
    if shadow is not None:
        alpha[sh_mask & ~drip_mask] = shadow["opacity"]
    if soft > 0:
        alpha = np.clip(_box_blur(np, alpha, max(1, int(round(soft)))), 0.0, 1.0)
        alpha = np.maximum(alpha, fill.astype(np.float32))
    col_px = np.concatenate([rgb, alpha[..., None]], axis=2)
    rev_px = np.stack([t_out, t_fill, t_hil, first], axis=2).astype(np.float32)
    made, saved = [], {}
    try:
        cimg = bpy.data.images.new(base + "_Color", w, h, alpha=True)
        made.append(cimg)
        cimg.pixels.foreach_set(col_px.ravel())
        cimg.update()
        cimg.use_fake_user = True
        rimg = bpy.data.images.new(base + "_Reveal", w, h, alpha=True, float_buffer=True)
        made.append(rimg)
        rimg.colorspace_settings.name = "Non-Color"
        rimg.alpha_mode = "CHANNEL_PACKED"
        rimg.pixels.foreach_set(rev_px.ravel())
        rimg.update()
        rimg.use_fake_user = True
        if "colorPath" in paths:
            cimg.filepath_raw = paths["colorPath"]
            cimg.file_format = "PNG"
            cimg.save()
            saved["colorPath"] = paths["colorPath"]
        if "revealPath" in paths:
            # WRITTEN HERE, NOT BY Image.save: a float image's PNG is un-premultiplied on the way out whatever its
            # alpha_mode (STRAIGHT, PREMUL, CHANNEL_PACKED or NONE: [0.3, 0.5, 0.7, 0.2] came back [1, 1, 1, 0.2] on
            # 3.6.23 and 5.2.1), so every reveal time under its first-paint A read 1.0. These four channels are data.
            _write_png16(np, paths["revealPath"], rev_px)
            saved["revealPath"] = paths["revealPath"]
        if "runPath" in paths:
            run_px = np.ones((h, w, 4), dtype=np.float32)
            run_px[..., 0] = run
            _write_png16(np, paths["runPath"], run_px)
            saved["runPath"] = paths["runPath"]
    except Exception:
        for img in made:
            bpy.data.images.remove(img)
        for p in saved.values():
            if os.path.exists(p):
                os.remove(p)
        raise
    nozzle = []
    for layer, box, ncol in (("outline", box_o, ncol_o), ("fill", box_f, ncol_f), ("highlight", box_h, ncol_h)):
        if box is None:
            continue
        x0, y0, x1, y1 = box
        win = passes[layer]
        for i in range(samples):
            s = i / float(samples - 1)
            c = min(ncol - 1, int(s * ncol))
            f = s * ncol - c
            if c % 2 == 1:
                f = 1.0 - f
            nozzle.append({"layer": layer, "t": round(win[0] + (win[1] - win[0]) * s, 6),
                           "u": round((x0 + (c + 0.5) * min(sw, x1 - x0)) / w, 6),
                           "v": round((y0 + f * (y1 - y0)) / h, 6)})
    if nzp is not None:
        t0, t1 = nzp["timeRange"]
        nozzle = [{"layer": "path", "t": round(min(max((t - t0) / (t1 - t0), 0.0), 1.0), 6), "time": t,
                   "u": round(x / w, 6), "v": round(y / h, 6)} for t, x, y in zip(nzp["t"], nzp["x"], nzp["y"])]
    total = float(w * h)
    out = {"text": text, "font": {"name": font_name, "file": font_file}, "size": [w, h], "margin": margin,
           "colorImage": made[0].name, "revealImage": made[1].name, "saved": saved,
           "textBox": [round(ox + sc * bx0, 3), round(oy + sc * by0, 3), round(ox + sc * bx1, 3),
                       round(oy + sc * by1, 3)],
           "coverage": {k: round(int(np.count_nonzero(m)) / total, 6)
                        for k, m in (("outline", outline), ("fill", fill), ("highlight", highlight),
                                     ("overspray", speck), ("painted", painted_r))},
           "widths": {"outline": ow, "highlight": hw, "overspraySpread": spread, "stroke": sw, "softness": soft},
           "passes": passes, "colors": colors, "seed": seed, "nozzlePath": nozzle,
           "channels": "reveal R outline (and overspray), G fill, B highlight, A first paint; 1.0 = never"}
    if nz_out is not None:
        out["nozzle"] = nz_out
        out["passes"] = {k: [round(min(max((a - nzp["timeRange"][0]) / (nzp["timeRange"][1] - nzp["timeRange"][0]),
                                           0.0), 1.0), 6) for a in v["segment"]] for k, v in nz_out["layers"].items()}
    if shadow is not None:
        out["shadow"] = dict(shadow, coverage=round(int(np.count_nonzero(sh_mask)) / total, 6), clipped=sh_clipped,
                             reveal="with the outline pass: each shadow pixel in R at the time of the outline pixel "
                                    "it falls from")
    if drips is not None:
        out["drips"] = {"asked": d_count, "made": len(drip_rows), "width": d_w, "seed": d_seed, "color": d_color,
                        "lengthMax": max((r_["length"] for r_ in drip_rows), default=0),
                        "clipped": sum(1 for r_ in drip_rows if r_["clipped"]),
                        "coverage": round(int(np.count_nonzero(drip_mask)) / total, 6), "starts": drip_rows[:200],
                        "run": "runPath R: each drip pixel's progress 0 (leaving the paint) to 1 (its tip); 1.0 = none"}
    if min_area is not None:
        px_area = decal[0] * decal[1] / total
        need = int(math.ceil(min_area / px_area - 1e-9))
        times = np.sort(first[first < 1.0].ravel())
        read = {"decalSize": decal, "minReadableArea": min_area, "pixelArea": px_area, "pixelsNeeded": need,
                "paintedArea": round(float(len(times)) * px_area, 9)}
        if need <= len(times):
            fr = float(times[need - 1])
            read["firstReveal"] = round(fr, 6)
            if keys:
                frame = None
                for (f0, v0), (f1, v1) in zip(keys, keys[1:]):
                    if v0 <= fr <= v1:
                        frame = f0 if v1 == v0 else f0 + (f1 - f0) * (fr - v0) / (v1 - v0)
                        break
                read["firstFrame"] = None if frame is None else round(frame, 6)
                if frame is None:
                    read["firstFrameNote"] = "revealKeys never reach %g" % fr
        else:
            read["firstReveal"] = None
            read["note"] = ("all the paint covers %.6g, under minReadableArea %g - it never reads at this size"
                            % (len(times) * px_area, min_area))
        read["areaByReveal"] = [{"reveal": round(r, 2), "area": round(float((times <= r).sum()) * px_area, 9)}
                                for r in [i / 10.0 for i in range(11)]]
        out["readable"] = read
    return out


# --------------------------------------------------------------------------------------------- channel_pack_image

_PACK_KEYS = ("mode", "channels", "image", "name", "width", "height", "filepath", "format", "colorDepth", "overwrite",
              "outputs", "fakeUser")
_PACK_NOTES = {
    "colorspace": "a packed map is data, so the result is always Non-Color - an sRGB pack would bend every value",
    "scale": "give width and height in pixels; every source is resampled to that size (bilinear)",
    "resize": "give width and height; every source is resampled to that size (bilinear)",
}
_CHANNELS = ("R", "G", "B", "A")
_PACK_SOURCES = ("R", "G", "B", "A", "LUMINANCE")


def _resample(np, a, w, h):
    """Bilinear resample of an (H, W) array to (h, w), pixel centers aligned."""
    H, W = a.shape
    if (H, W) == (h, w):
        return a
    ys = np.clip((np.arange(h) + 0.5) * H / h - 0.5, 0, H - 1)
    xs = np.clip((np.arange(w) + 0.5) * W / w - 0.5, 0, W - 1)
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    y1 = np.minimum(y0 + 1, H - 1)
    x1 = np.minimum(x0 + 1, W - 1)
    fy = (ys - y0)[:, None]
    fx = (xs - x0)[None, :]
    top = a[y0][:, x0] * (1 - fx) + a[y0][:, x1] * fx
    bot = a[y1][:, x0] * (1 - fx) + a[y1][:, x1] * fx
    return (top * (1 - fy) + bot * fy).astype(np.float32)


def _pack_source(key, spec, tail):
    """One output channel's source: ("constant", value, False) or ("image", (image, channel), invert)."""
    if isinstance(spec, bool):
        raise MifOpError("channels.%s: a boolean is not a value - give a number 0-1 or {image, channel}. %s"
                         % (key, tail))
    if isinstance(spec, (int, float)):
        return ("constant", finite_floats([spec], "channels." + key)[0], False)
    if not isinstance(spec, dict):
        raise MifOpError("channels.%s takes a number 0-1 or {image, channel, invert}, got %r. %s" % (key, spec, tail))
    unknown = [k for k in spec if k not in ("image", "channel", "invert", "constant")]
    if unknown:
        raise MifOpError("channels.%s: unknown key(s) %s. Accepted: image, channel, invert, constant. %s"
                         % (key, ", ".join(sorted(unknown)), tail))
    invert = bool(spec.get("invert", False))
    if "constant" in spec:
        if "image" in spec:
            raise MifOpError("channels.%s gives both a constant and an image - one or the other. %s" % (key, tail))
        val = finite_floats([spec["constant"]], "channels.%s.constant" % key)[0]
        return ("constant", (1.0 - val) if invert else val, False)
    img = _get_image(spec.get("image"), tail)
    if img.size[0] == 0 or img.size[1] == 0:
        raise MifOpError("channels.%s: image '%s' has no pixels (0x0) - is its file missing? %s" % (key, img.name, tail))
    ch = str(spec.get("channel", "LUMINANCE" if img.channels < 3 else key)).upper()
    if ch not in _PACK_SOURCES:
        raise MifOpError("channels.%s.channel must be one of %s, got %r. %s"
                         % (key, ", ".join(_PACK_SOURCES), spec.get("channel"), tail))
    return ("image", (img, ch), invert)


def _channel_values(np, img, ch, cache):
    px = cache.get(img.name)
    if px is None:
        px = cache[img.name] = _pixels_of(np, img)
    if ch == "LUMINANCE":
        return (px[..., 0] * 0.2126 + px[..., 1] * 0.7152 + px[..., 2] * 0.0722).astype(np.float32)
    return px[..., _CHANNELS.index(ch)].astype(np.float32)


def _stats(np, a):
    return {"min": round(float(a.min()), 6), "max": round(float(a.max()), 6), "mean": round(float(a.mean()), 6)}


def _write_data_image(np, name, rgba, has_alpha, filepath, fmt, depth, overwrite, fake, tail):
    """A new Non-Color image from an (h, w, 4) array, saved through save_image when a path is given."""
    h, w = rgba.shape[:2]
    want_float = depth in (16, 32) or (fmt or "").startswith("OPEN_EXR")
    img = bpy.data.images.new(str(name), w, h, alpha=has_alpha, float_buffer=want_float)
    try:
        _colorspace_or_refuse(img, "Non-Color", tail)
        img.pixels.foreach_set(np.ascontiguousarray(rgba, dtype=np.float32).ravel())
        img.update()
        img.use_fake_user = fake
        saved = None
        if filepath:
            req = {"image": img.name, "filepath": filepath, "overwrite": overwrite}
            if fmt:
                req["format"] = fmt
            if depth is not None and fmt in ("PNG", "OPEN_EXR"):
                req["colorDepth"] = depth
            saved = op_save_image(req)
    except Exception:
        bpy.data.images.remove(img)
        raise
    return img, saved


def _read_back(np, path, chans):
    probe = bpy.data.images.load(path, check_existing=False)
    try:
        probe.colorspace_settings.name = "Non-Color"
        px = _pixels_of(np, probe)
        return {"size": [int(probe.size[0]), int(probe.size[1])],
                "channels": {c: _stats(np, px[..., _CHANNELS.index(c)]) for c in chans}}
    finally:
        bpy.data.images.remove(probe)


def _pack_depth(fmt, depth, path, tail):
    f = fmt or _format_from_path(os.path.abspath(bpy.path.abspath(path))) or "PNG"
    d = depth
    if f.startswith("OPEN_EXR"):
        d = d or 32
        if d not in (16, 32):
            raise MifOpError("an EXR is 16 or 32 bits per channel, got %r. %s" % (d, tail))
    elif f == "PNG":
        d = d or 8
        if d not in (8, 16):
            raise MifOpError("a PNG is 8 or 16 bits per channel, got %r. %s" % (d, tail))
    elif d is not None:
        raise MifOpError("colorDepth applies to PNG and OPEN_EXR; %s is written at 8 bits. %s" % (f, tail))
    return f, d


def _pack_split(np, params, fmt, depth, overwrite, fake, tail):
    for k in ("channels", "name", "width", "height", "filepath"):
        if k in params:
            raise MifOpError("'%s' is PACK's; SPLIT takes image and outputs. %s" % (k, tail))
    src = _get_image(take(params, "image", required=True, kind=str), tail)
    outputs = take(params, "outputs", required=True, kind=dict)
    if not outputs or any(str(k).upper() not in _CHANNELS for k in outputs):
        raise MifOpError("outputs maps R, G, B or A to a file path, got %r. %s" % (outputs, tail))
    if src.size[0] == 0 or src.size[1] == 0:
        raise MifOpError("'%s' has no pixels (0x0). %s" % (src.name, tail))
    plan = []
    for k, path in outputs.items():
        if not isinstance(path, str) or not path:
            raise MifOpError("outputs.%s must be a file path. %s" % (k, tail))
        resolved = os.path.abspath(bpy.path.abspath(path))
        check_output_path(path, resolved, "written")
        if file_before(resolved) is not None and not overwrite:
            raise MifOpError("'%s' exists and overwrite is false. %s" % (resolved, tail))
        plan.append((str(k).upper(), resolved) + _pack_depth(fmt, depth, path, tail))
    cache, written = {}, []
    for ch, resolved, f, d in plan:
        a = _channel_values(np, src, ch, cache)
        rgba = np.empty(a.shape + (4,), dtype=np.float32)
        rgba[..., 0] = rgba[..., 1] = rgba[..., 2] = a
        rgba[..., 3] = 1.0
        img, saved = _write_data_image(np, os.path.splitext(os.path.basename(resolved))[0], rgba, False, resolved, f,
                                       d, overwrite, fake, tail)
        row = {"channel": ch, "image": img.name, "file": resolved, "format": f, "colorDepth": d,
               "written": _stats(np, a), "bytes": saved["bytes"], "overwrote": saved.get("overwrote", False)}
        row["readBack"] = _read_back(np, resolved, ["R"])["channels"]["R"]
        written.append(row)
    return {"mode": "SPLIT", "image": src.name, "size": list(src.size), "outputs": written}


def op_channel_pack_image(params):
    """Pack channels of several images (or constants) into one data texture - ORM, RMA, a mask map - or SPLIT one.

    WHY. Unreal's ORM convention (R occlusion, G roughness, B metallic) and its variants mean every PBR hand-off packs, and
    a set baked or downloaded as separate maps has to be combined; nothing in the add-on made one. Values are read as each
    image stores them (Image.pixels) and written to a Non-Color image, so data stays data; a source of another size is
    resampled bilinearly to the output size.

    params:
      mode (str)            PACK (default) | SPLIT
      PACK:
        channels (dict)     {R, G, B, A}: each a number 0-1 (a constant: AO 1, metallic 0) or {image, channel
                            (R|G|B|A|LUMINANCE; default the same letter, LUMINANCE for a one-channel image), invert}.
                            R, G and B are required; A only when given (the result then has alpha)
        name (str)          the new image's name (default T_Packed); suffixed when taken
        width, height (int) default the largest source's size; 1-16384
        filepath (str)      write it there and point the image at it (save_image's route); omitted, in memory only
      SPLIT:
        image (str)         the packed map
        outputs (dict)      {R: filepath, G: ..., B: ..., A: ...} - one grayscale image per channel, named after its file
      format (str)          PNG | TARGA | OPEN_EXR | ... (default from the extension)
      colorDepth (int)      8 (default) or 16 for PNG; 16 or 32 (default) for OPEN_EXR
      overwrite (bool)      default true; false refuses an existing path
      fakeUser (bool)       default true - an image nothing uses is otherwise deleted by the next save of the .blend
    Replies per channel the min / max / mean written, and readBack: each written file loaded again and measured.
    """
    reject_unknown(params, _PACK_KEYS, "channel_pack_image", notes=_PACK_NOTES)
    tail = "NOTHING was written."
    import numpy as np                                   # noqa: PLC0415 - Blender bundles it
    mode = str(take(params, "mode", default="PACK", kind=str)).upper()
    if mode not in ("PACK", "SPLIT"):
        raise MifOpError("mode must be PACK or SPLIT, got %r. %s" % (params.get("mode"), tail))
    fmt = take(params, "format", default=None, kind=str)
    fmt = str(fmt).upper() if fmt else None
    if fmt is not None and fmt not in _image_formats():
        raise MifOpError("format '%s' is not a still-image format on Blender %s. Available: %s. %s"
                         % (fmt, bpy.app.version_string, ", ".join(_image_formats()), tail))
    depth = take_int(params, "colorDepth", default=None)
    overwrite = take_bool(params, "overwrite", default=True)
    fake = take_bool(params, "fakeUser", default=True)
    if mode == "SPLIT":
        return _pack_split(np, params, fmt, depth, overwrite, fake, tail)

    for k in ("image", "outputs"):
        if k in params:
            raise MifOpError("'%s' is SPLIT's; PACK takes channels. %s" % (k, tail))
    chans = take(params, "channels", required=True, kind=dict)
    norm = {}
    for k, v in chans.items():
        ku = str(k).upper()
        if ku not in _CHANNELS:
            raise MifOpError("channels keys are R, G, B and A, got %r. %s" % (k, tail))
        norm[ku] = _pack_source(ku, v, tail)
    missing = [c for c in ("R", "G", "B") if c not in norm]
    if missing:
        raise MifOpError("channels needs R, G and B (a constant for a channel with no source, e.g. 1 for occlusion, 0 "
                         "for metallic); missing %s. %s" % (", ".join(missing), tail))
    sizes = [tuple(v[1][0].size) for v in norm.values() if v[0] == "image"]
    w = take_int(params, "width", default=None)
    h = take_int(params, "height", default=None)
    if (w is None) != (h is None):
        raise MifOpError("width and height go together. %s" % tail)
    if w is None:
        if not sizes:
            raise MifOpError("every channel is a constant - give width and height for the image size. %s" % tail)
        w, h = max(sizes, key=lambda s: s[0] * s[1])
    w, h = int(w), int(h)
    for key, val in (("width", w), ("height", h)):
        if not 1 <= val <= 16384:
            raise MifOpError("%s %d is outside 1-16384. %s" % (key, val, tail))
    raw = take(params, "filepath", default=None, kind=str)
    f = d = None
    if raw is not None:
        resolved = os.path.abspath(bpy.path.abspath(raw))
        check_output_path(raw, resolved, "written")
        if file_before(resolved) is not None and not overwrite:
            raise MifOpError("'%s' exists and overwrite is false. %s" % (resolved, tail))
        f, d = _pack_depth(fmt, depth, raw, tail)
    elif fmt is not None or depth is not None:
        raise MifOpError("format and colorDepth need filepath - the image is written there. %s" % tail)
    rgba = np.ones((h, w, 4), dtype=np.float32)
    report, resampled, cache = {}, set(), {}
    for c in _CHANNELS:
        if c not in norm:
            continue
        kind, val, invert = norm[c]
        if kind == "constant":
            a = np.full((h, w), float(val), dtype=np.float32)
            src = {"constant": round(float(val), 6)}
        else:
            img, ch = val
            a = _channel_values(np, img, ch, cache)
            if a.shape != (h, w):
                resampled.add(img.name)
                a = _resample(np, a, w, h)
            if invert:
                a = 1.0 - a
            src = {"image": img.name, "channel": ch, "invert": invert}
        rgba[..., _CHANNELS.index(c)] = a
        report[c] = dict(src, written=_stats(np, a))
    clipped = False
    if not (f or "").startswith("OPEN_EXR") and (rgba.min() < 0.0 or rgba.max() > 1.0):
        rgba = np.clip(rgba, 0.0, 1.0)
        clipped = True
    name = take(params, "name", default="T_Packed", kind=str)
    img, saved = _write_data_image(np, name, rgba, "A" in norm, raw, f, d, overwrite, fake, tail)
    out = {"mode": "PACK", "image": img.name, "requestedName": name, "nameWasSuffixed": img.name != name,
           "size": [w, h], "alpha": "A" in norm, "colorSpace": img.colorspace_settings.name, "channels": report,
           "resampled": sorted(resampled), "clippedTo01": clipped, "file": None}
    if saved is not None:
        out.update({k: saved[k] for k in ("file", "bytes", "format", "colorDepth", "overwrote", "replacedBytes",
                                          "replacedModified") if k in saved})
        out["readBack"] = _read_back(np, saved["file"], [c for c in _CHANNELS if c in norm])
    else:
        out["note"] = "in memory only (kept by its fake user): pass filepath, or save_image, to write it"
    return out


_IST_KEYS = {"image", "file", "normal", "tiling", "pixels", "uvs", "tile"}
_IST_CHANNELS = ("R", "G", "B", "A")
_IST_MAX_PIXELS = 67108864          # 8192 x 8192: a 1 GB float read beyond that is refused rather than attempted


def op_image_stats(params):
    """Describe an image's pixels: what a bake check needs beyond 'did it change'.

    WHY (second 1.3 audit, 2026-10-25). A black AO, a normal map with its green inverted or a NaN-filled EXR from a
    degenerate bake all pass bake_texture's 'the image changed'; compare_to_reference compares two images and nothing
    described one. The read is Image.pixels.foreach_get - the bulk copy compare_to_reference uses (ops_render.py) - so the
    values are AS STORED: an 8-bit sRGB texture reads its encoded 0-1 values, a float image its linear ones.

    params:
      image (str) | file (str)   an image in the file, or a file on disk (loaded, read and removed again)
      normal (bool)              also the tangent-space normal-map check: decoded n = 2 x rgb - 1, the share of pixels
                                 with blue under 0.5 (pointing into the surface), the mean vector length, the mean of
                                 red and green (0.5 on a flat map)
      tiling (bool)              also the tiling check: the mean difference across the wrap (last column vs first, last
                                 row vs first) beside the mean difference between neighbouring columns / rows inside
      pixels [[x, y], ...]       sample values at pixel coordinates (0,0 bottom left, as Blender stores them)
      uvs [[u, v], ...]          sample values at UVs (nearest pixel; u, v wrap)
      tile (int)                 a UDIM tile number (default the image's first)
    Per channel: min, max, mean, stddev over finite values, NaN and Inf counts, the share at 0 and at 1; alpha
    coverage (share with alpha above 0.5); and constant when a channel holds one value.
    """
    import numpy as np                                               # noqa: PLC0415 - bundled with every Blender
    reject_unknown(params, _IST_KEYS, "image_stats")
    tail = "NOTHING was read."
    name = take(params, "image", kind=str)
    path = take(params, "file", kind=str)
    if (name is None) == (path is None):
        raise MifOpError("image_stats takes image (a name in this file) OR file (a path on disk). %s" % tail)
    loaded = None
    if path is not None:
        full = os.path.abspath(bpy.path.abspath(path))
        if not os.path.isfile(full):
            raise MifOpError("no file '%s'. %s" % (full, tail))
        try:
            loaded = bpy.data.images.load(full, check_existing=False)
        except RuntimeError as exc:
            raise MifOpError("Blender could not read '%s' as an image (%s). %s" % (full, exc, tail))
        img = loaded
    else:
        img = _get_image(name, tail)
    try:
        tile = params.get("tile")
        if tile is not None:
            if img.source != "TILED":
                raise MifOpError("'%s' is not a UDIM image; drop tile. %s" % (img.name, tail))
            if not any(t.number == tile for t in img.tiles):
                raise MifOpError("'%s' has no tile %r (it has %s). %s"
                                 % (img.name, tile, [t.number for t in img.tiles], tail))
            img.tiles.active = next(t for t in img.tiles if t.number == tile)
        w, h = int(img.size[0]), int(img.size[1])
        if not w or not h:
            raise MifOpError("'%s' has no pixels (size %dx%d; source %s, file %s). %s"
                             % (img.name, w, h, img.source, img.filepath or "-", tail))
        if w * h > _IST_MAX_PIXELS:
            raise MifOpError("'%s' is %dx%d - over %d pixels this read would need more than 1 GB. resize_image a copy "
                             "first. %s" % (img.name, w, h, _IST_MAX_PIXELS, tail))
        nch = int(img.channels) or 4
        a = np.empty(w * h * nch, dtype=np.float32)
        img.pixels.foreach_get(a)
        a = a.reshape(h, w, nch)
        chans = _IST_CHANNELS[:nch] if nch != 1 else ("V",)
        out = {"image": img.name, "size": [w, h], "channels": nch, "isFloat": bool(img.is_float),
               "colorspace": img.colorspace_settings.name, "source": img.source,
               "valuesNote": "as stored: an 8-bit sRGB texture reads its encoded values, a float image linear ones",
               "stats": {}}
        n = w * h
        for i, c in enumerate(chans):
            ch = a[:, :, i]
            nan = int(np.isnan(ch).sum())
            inf = int(np.isinf(ch).sum())
            fin = ch[np.isfinite(ch)]
            row = {"nan": nan, "inf": inf}
            if fin.size:
                row.update(min=round(float(fin.min()), 6), max=round(float(fin.max()), 6),
                           mean=round(float(fin.mean()), 6), stddev=round(float(fin.std()), 6),
                           atZero=round(float((fin <= 0.0).sum()) / n, 6), atOne=round(float((fin >= 1.0).sum()) / n, 6))
                row["constant"] = bool(fin.max() == fin.min())
            out["stats"][c] = row
        out["nanTotal"] = sum(r["nan"] for r in out["stats"].values())
        out["infTotal"] = sum(r["inf"] for r in out["stats"].values())
        if nch == 4:
            al = a[:, :, 3]
            out["alphaCoverage"] = round(float((al > 0.5).sum()) / n, 6)
        warn = []
        if out["nanTotal"] or out["infTotal"]:
            warn.append("%d NaN and %d Inf values" % (out["nanTotal"], out["infTotal"]))
        rgb = [out["stats"][c] for c in chans[:3] if c in out["stats"]]
        if rgb and all(r.get("max", 1) <= 0.0 for r in rgb):
            warn.append("every color value is 0 - a black image")
        if take_bool(params, "normal", default=False):
            if nch < 3:
                raise MifOpError("the normal check needs RGB; '%s' has %d channel(s). %s" % (img.name, nch, tail))
            v = a[:, :, :3] * 2.0 - 1.0
            fin = np.isfinite(v).all(axis=2)
            v = v[fin]
            ln = np.sqrt((v * v).sum(axis=1)) if v.size else np.zeros(0)
            nb = {"blueUnderHalf": round(float((v[:, 2] < 0.0).sum()) / max(1, len(v)), 6) if v.size else None,
                  "meanLength": round(float(ln.mean()), 6) if ln.size else None,
                  "meanRed": round(float((v[:, 0].mean() + 1) / 2), 6) if v.size else None,
                  "meanGreen": round(float((v[:, 1].mean() + 1) / 2), 6) if v.size else None,
                  "note": "a tangent-space normal map has blue above 0.5 almost everywhere, vectors of length about 1 "
                          "(8-bit storage reads 0.98-1.02) and red / green means near 0.5; a color map fails all three. "
                          "Green's direction (OpenGL Y+ or DirectX Y-) cannot be read off the pixels alone."}
            nb["looksLikeNormalMap"] = bool(nb["blueUnderHalf"] is not None and nb["blueUnderHalf"] < 0.02
                                            and 0.9 <= nb["meanLength"] <= 1.1)
            if img.colorspace_settings.name.lower() in ("srgb", "srgb - texture"):
                nb["colorspaceWarning"] = ("the image is tagged %s; a normal map is read as Non-Color data"
                                           % img.colorspace_settings.name)
            out["normal"] = nb
        if take_bool(params, "tiling", default=False):
            c3 = a[:, :, :min(3, nch)]
            if w < 3 or h < 3:
                raise MifOpError("the tiling check needs at least 3x3 pixels. %s" % tail)
            wrap_x = float(np.nanmean(np.abs(c3[:, -1] - c3[:, 0])))
            in_x = float(np.nanmean(np.abs(c3[:, 1:] - c3[:, :-1])))
            wrap_y = float(np.nanmean(np.abs(c3[-1] - c3[0])))
            in_y = float(np.nanmean(np.abs(c3[1:] - c3[:-1])))
            rx = wrap_x / in_x if in_x > 1e-9 else (0.0 if wrap_x < 1e-9 else float("inf"))
            ry = wrap_y / in_y if in_y > 1e-9 else (0.0 if wrap_y < 1e-9 else float("inf"))
            out["tiling"] = {"wrapDiffX": round(wrap_x, 6), "insideDiffX": round(in_x, 6),
                             "wrapDiffY": round(wrap_y, 6), "insideDiffY": round(in_y, 6),
                             "ratioX": round(rx, 4) if math.isfinite(rx) else None,
                             "ratioY": round(ry, 4) if math.isfinite(ry) else None,
                             "tilesX": bool(math.isfinite(rx) and rx <= 2.0), "tilesY": bool(math.isfinite(ry) and ry <= 2.0),
                             "note": "ratio = the step across the wrap over the mean step between neighbours; about 1 "
                                     "tiles, well above 2 shows a seam"}
        samples = []
        for key in ("pixels", "uvs"):
            pts = params.get(key)
            if pts is None:
                continue
            if not isinstance(pts, list) or not all(isinstance(p, (list, tuple)) and len(p) == 2 for p in pts):
                raise MifOpError("%s is a list of [x, y] pairs. %s" % (key, tail))
            for p in pts[:1000]:
                if key == "pixels":
                    x, y = finite_int(p[0], "pixels"), finite_int(p[1], "pixels")
                    if not (0 <= x < w and 0 <= y < h):
                        raise MifOpError("pixel %r is outside %dx%d. %s" % (p, w, h, tail))
                else:
                    u, v = finite_floats(p, "uvs")
                    x = int(math.floor((u % 1.0) * w)) % w
                    y = int(math.floor((v % 1.0) * h)) % h
                samples.append({key[:-1]: list(p), "pixel": [x, y],
                                "value": [None if not math.isfinite(float(v)) else round(float(v), 6) for v in a[y, x]]})
        if samples:
            out["samples"] = samples
        if warn:
            out["warning"] = "; ".join(warn)
        return out
    finally:
        if loaded is not None:
            bpy.data.images.remove(loaded)


OPS = {
    "image_stats": op_image_stats,
    "channel_pack_image": op_channel_pack_image,
    "spray_text_art": op_spray_text_art,
    "project_image_decal": op_project_image_decal,
    "paint_texture": op_paint_texture,
    "resize_image": op_resize_image,
    "create_udim_image": op_create_udim_image,
    "create_image": op_create_image,
    "bake_hidden_face_map": op_bake_hidden_face_map,
    "save_image": op_save_image,
    "derive_image_map": op_derive_image_map,
}
