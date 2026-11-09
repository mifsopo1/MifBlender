"""Video in Blender: strips in the sequencer (Video Sequencer), and the editor-level settings around them.

Sound strips are add_sound's (ops_audio); this module makes and edits every other kind - movie, image, scene, meta,
color, text, adjustment and the effects and transitions - and reads them all back.

EVERY BUILD DIFFERENCE BELOW WAS READ OFF THE RUNNING BUILD, 2026-10-06, on 4.2.23, 4.4.0, 5.0.1 and 5.2.2 (PyPI bpy):

=============================================================================
NAMES THAT MOVED
=============================================================================
SequenceEditor.sequences / sequences_all up to 4.2, strips / strips_all from 4.4 (5.0 has only the new names); the
classes are *Sequence up to 4.2 and *Strip from 4.4. new_effect takes (frame_end, seq1, seq2) up to 4.4 and (length,
input1, input2) from 5.0, so its parameters are read off bl_rna.functions["new_effect"] rather than named. The strip's
first and last frame are frame_final_start / frame_final_end up to 5.0 and left_handle / right_handle on 5.2, where the
old names print a deprecation warning (ops_audio._span reads whichever the build has). TRANSFORM and OVER_DROP effects
exist up to 4.4 only; COMPOSITOR exists on 5.2 only; the type list a refusal names is the build's own enum.

=============================================================================
WHAT CRASHES, AND WHAT THIS MODULE DOES ABOUT IT
=============================================================================
* Strip.modifiers.new() SEGFAULTS 5.0.1 when the workspace has no sequencer_scene (ops_common.sequencer_scene_guard).
* Strip.split() on a strip that feeds an effect, followed by moving the effect's other input, SEGFAULTS every build
  (4.2.23 and 5.2.2 measured, exit 139): edit_strip refuses a split of an effect input and names the effects.
* Strip.move_to_meta() moves the strip and leaves the META's own range stale - a meta holding strips over frames
  10-90 read 10-11 on 4.2.23 and 5.2.2 - so a meta is made by Blender's sequencer.meta_make in a borrowed Sequencer
  area (ops_common.editor_override), which spans its contents on every build.
* A two-input effect spans the overlap of its inputs whatever length it is given (a CROSS asked for 11 frames over
  inputs overlapping 40-51 read 40-51 on every build), so add_strip refuses frame and length for an effect with inputs.
"""
import math
import os
import time

import bpy
from mathutils import Matrix, Vector

from .ops_anim import _fcurves
from .ops_audio import _all_strips, _fps, _seq_scene, _span, _top_strips, _workspace
from .ops_common import (remove_objects, MifOpError, assign_id_name, editor_override, finite_floats, finite_int, reject_unknown, rna_properties, rnd,
                         sequencer_scene_guard, take, take_bool, take_float, take_int, write_rna_property)

_ADDED = "NOTHING was added."
_CHANGED = "NOTHING was changed."

# HOW MANY INPUTS AN EFFECT TAKES. Read off a made strip's input_count on 4.2.23 and 5.2.2; an effect the table does not
# know is made and its input_count is what decides.
_EFFECT_INPUTS = {"CROSS": 2, "ADD": 2, "SUBTRACT": 2, "ALPHA_OVER": 2, "ALPHA_UNDER": 2, "GAMMA_CROSS": 2,
                  "MULTIPLY": 2, "OVER_DROP": 2, "WIPE": 2, "COLORMIX": 2, "GLOW": 1, "TRANSFORM": 1, "SPEED": 1,
                  "GAUSSIAN_BLUR": 1, "COLOR": 0, "TEXT": 0, "ADJUSTMENT": 0, "MULTICAM": 0, "COMPOSITOR": 0}
_SOURCE_TYPES = ("MOVIE", "IMAGE", "SCENE", "META")


# ---------------------------------------------------------------------------------------------------------------
# shared readers
# ---------------------------------------------------------------------------------------------------------------

def _effect_types():
    """The build's new_effect types - StripsTopLevel from 4.4, SequencesTopLevel before."""
    holder = getattr(bpy.types, "StripsTopLevel", None) or getattr(bpy.types, "SequencesTopLevel", None)
    fn = holder.bl_rna.functions.get("new_effect") if holder is not None else None
    return [e.identifier for e in fn.parameters["type"].enum_items] if fn is not None else []


def _find_strip(sc, name, tail, key="strip"):
    ed = sc.sequence_editor
    st = _all_strips(ed).get(name) if ed is not None else None
    if st is None:
        have = sorted(s.name for s in _all_strips(ed)) if ed is not None else []
        raise MifOpError("scene '%s' has no strip named '%s' (%s). Strips: %s. %s"
                         % (sc.name, name, key, ", ".join(have[:40]) or "(none - its sequencer is empty)", tail))
    return st


def _inputs(st):
    return [getattr(st, k).name for k in ("input_1", "input_2")
            if k in st.bl_rna.properties and getattr(st, k) is not None]


def _effects_fed_by(sc, st):
    return [s.name for s in _all_strips(sc.sequence_editor) if st.name in _inputs(s)]


def _children(st):
    kids = getattr(st, "strips", None)
    if kids is None:
        kids = getattr(st, "sequences", None)
    return [k.name for k in kids] if kids is not None else None


def _modifier_rows(st):
    rows = []
    for md in st.modifiers:
        row = {"name": md.name, "type": md.type, "enabled": not md.mute}
        if "node_group" in md.bl_rna.properties:
            # A COMPOSITOR modifier's group, read back - written through settings {node_group} and shown nowhere before.
            row["nodeGroup"] = md.node_group.name if md.node_group else None
        rows.append(row)
    return rows


def _strip_row(sc, st):
    """One strip as every reply in this module reports it."""
    first, end = _span(st)
    row = {"name": st.name, "type": st.type, "channel": int(st.channel), "frameStart": first, "frameEnd": end - 1,
           "duration": end - first, "contentStart": int(st.frame_start), "mute": bool(st.mute), "lock": bool(st.lock)}
    for key, attr in (("blendType", "blend_type"), ("opacity", "blend_alpha")):
        if attr in st.bl_rna.properties:
            val = getattr(st, attr)
            row[key] = round(float(val), 6) if isinstance(val, float) else val
    ins = _inputs(st)
    if ins:
        row["inputs"] = ins
    if st.type == "MOVIE":
        row["source"] = {"filepath": st.filepath, "frames": int(st.frame_duration), "fps": round(float(st.fps), 6)}
    elif st.type == "IMAGE":
        row["source"] = {"directory": st.directory, "files": [e.filename for e in st.elements][:50],
                         "frames": len(st.elements)}
    elif st.type == "SCENE":
        row["source"] = {"scene": st.scene.name if st.scene else None, "sceneInput": st.scene_input,
                         "camera": st.scene_camera.name if st.scene_camera else None,
                         "fps": round(_fps(st.scene), 6) if st.scene else None}
    elif st.type == "SOUND":
        row["source"] = {"filepath": st.sound.filepath if st.sound else None}
        row["volume"] = round(float(st.volume), 6)
    elif st.type == "META":
        row["strips"] = _children(st)
    elif st.type == "COLOR":
        row["color"] = rnd(st.color)
    elif st.type == "TEXT":
        row["text"] = st.text
        row["fontSize"] = round(float(st.font_size), 4)
    mods = _modifier_rows(st)
    if mods:
        row["modifiers"] = mods
    conn = getattr(st, "connections", None)
    if conn:
        row["connected"] = sorted(s.name for s in conn)
    if getattr(st, "use_proxy", False) and st.type in _PROXY_TYPES:
        row["proxy"] = _proxy_row(st)
    parent = None
    try:
        parent = st.parent_meta()
    except (AttributeError, TypeError, RuntimeError):
        parent = None
    if parent is not None:
        row["meta"] = parent.name
    return row


# PROXIES (2026-10-08): the sizes a movie or image strip can have a proxy built at, and where Blender writes them.
_PROXY_TYPES = ("MOVIE", "IMAGE")
_PROXY_SIZES = (25, 50, 75, 100)
_PROXY_KEYS = {"sizes", "quality", "overwrite", "build"}


def _proxy_dir(st):
    """The folder Blender writes this strip's proxies in: the custom directory, or BL_proxy beside the source, in a
    folder named for the movie file (measured on 4.2.17, 4.4.0 and 5.2.1: <dir>/BL_proxy/m.mp4/proxy_25.avi)."""
    px = st.proxy
    if px.use_proxy_custom_directory and px.directory:
        base = bpy.path.abspath(px.directory)
    else:
        src_dir = st.directory if st.type == "IMAGE" else os.path.dirname(st.filepath)
        base = os.path.join(bpy.path.abspath(src_dir), "BL_proxy")
    name = os.path.basename(st.filepath) if st.type == "MOVIE" else (st.elements[0].filename if len(st.elements) else "")
    return os.path.normpath(os.path.join(base, name))


def _proxy_row(st):
    px = st.proxy
    folder = _proxy_dir(st)
    sizes = [n for n in _PROXY_SIZES if getattr(px, "build_%d" % n)]
    built = sorted(f for f in os.listdir(folder)) if os.path.isdir(folder) else []
    return {"sizes": sizes, "quality": int(px.quality), "directory": folder, "files": built[:50],
            "built": {n: any(f.startswith("proxy_%d" % n) for f in built) for n in sizes}}


def _connect_supported():
    """Strip connections read back only where Strip.connections exists (5.2.1 has it; 5.0.1 and 4.4.0 have the operators
    and no way to read what they did, 4.2 neither - measured 2026-10-08)."""
    cls = getattr(bpy.types, "Strip", None)
    return cls is not None and "connections" in cls.bl_rna.properties and hasattr(bpy.ops.sequencer, "connect")


def _select_strips_only(ed, strips):
    for s in _all_strips(ed):
        s.select = s in strips


def _taken(ed, first, end, skip=()):
    """{channel: strip name} of top-level strips over [first, end), leaving out `skip`."""
    taken = {}
    for st in _top_strips(ed):
        if st.name in skip:
            continue
        a, b = _span(st)
        if a < end and first < b:
            taken.setdefault(int(st.channel), st.name)
    return taken


def _pick_channel(ed, channel, first, end, tail, skip=()):
    lo, hi = 1, 128
    taken = _taken(ed, first, end, skip) if ed is not None else {}
    free = next((c for c in range(lo, hi + 1) if c not in taken), None)
    if channel is not None:
        if not (lo <= channel <= hi):
            raise MifOpError("channel must be 1-128, got %d. %s" % (channel, tail))
        if channel in taken:
            raise MifOpError("channel %d already holds strip '%s' over frames %d-%d - %s. %s"
                             % (channel, taken[channel], first, end - 1,
                                ("channel %d is free there" % free) if free else "every channel is taken there", tail))
        return channel
    if free is None:
        raise MifOpError("every channel holds a strip over frames %d-%d. %s" % (first, end - 1, tail))
    return free


def _check_settings(settings, tail):
    if settings is None:
        return {}
    if not isinstance(settings, dict):
        raise MifOpError("settings takes {property: value} of the strip's own RNA properties, got %r. %s"
                         % (settings, tail))
    return settings


def _write_settings(st, settings, tail):
    out = {}
    for key, value in settings.items():
        out[key] = write_rna_property(st, key, value, tail, "strip '%s' (%s)" % (st.name, st.type), st.type,
                                      frozenset({"name", "channel", "frame_start", "type", "select", "modifiers"}))
    return out


def _vec(params, key, n, tail):
    val = params.get(key)
    if val is None:
        return None
    if not isinstance(val, (list, tuple)) or len(val) != n:
        raise MifOpError("'%s' takes %d numbers, got %r. %s" % (key, n, val, tail))
    try:
        return finite_floats(val, key)
    except MifOpError as exc:
        raise MifOpError("%s %s" % (str(exc).rstrip("."), tail)) from None


# ---------------------------------------------------------------------------------------------------------------
# add_strip
# ---------------------------------------------------------------------------------------------------------------

_ADD_STRIP_KEYS = {"type", "scene", "name", "channel", "frame", "length", "file", "files", "fitMethod", "withSound",
                   "sourceScene", "sceneInput", "sceneCamera", "inputs", "text", "fontSize", "color", "location",
                   "alignX", "strips", "settings"}
_TYPE_KEYS = {"file": ("MOVIE", "IMAGE"), "files": ("IMAGE",), "fitMethod": ("MOVIE", "IMAGE"), "withSound": ("MOVIE",),
              "sourceScene": ("SCENE",), "sceneInput": ("SCENE",), "sceneCamera": ("SCENE",),
              "text": ("TEXT",), "fontSize": ("TEXT",), "location": ("TEXT",), "alignX": ("TEXT",),
              "color": ("COLOR", "TEXT"), "strips": ("META",)}


def op_add_strip(params):
    """Add a strip to a scene's sequencer - movie, image, scene, meta, or an effect (color, text, adjustment, a
    transition or a filter) - at a channel and frame, refusing a channel another strip occupies there.

    THE GAP: the only strips the add-on made were sound strips (add_sound); a cut, a title, a color card, a transition
    or an adjustment layer was raw Python.

    params:
      type (str, required)   MOVIE, IMAGE, SCENE, META, or one of the build's new_effect types (COLOR, TEXT,
                             ADJUSTMENT, CROSS, GAMMA_CROSS, WIPE, ADD, SUBTRACT, MULTIPLY, ALPHA_OVER, ALPHA_UNDER,
                             COLORMIX, GLOW, GAUSSIAN_BLUR, SPEED, MULTICAM; TRANSFORM and OVER_DROP up to 4.4;
                             COMPOSITOR on 5.2). Sound is add_sound's.
      scene (str)            whose sequencer; default the workspace's sequencer scene on 5.0+ when set, else the
                             window's scene
      name (str), channel (int 1-128, default the lowest free over the strip's frames), frame (int, default the
      current frame) - frame and length are refused for an effect with inputs, which spans its inputs
      length (int)           frames, required for COLOR, TEXT, ADJUSTMENT, MULTICAM and COMPOSITOR
      file (str)             MOVIE or IMAGE; files [names] for an IMAGE sequence (in file's folder, in order)
      fitMethod              FIT, FILL, STRETCH, ORIGINAL; withSound (MOVIE, default true) adds its sound strip on the
                             channel above, the same start and length, when the file has sound
      sourceScene, sceneInput CAMERA|SEQUENCER, sceneCamera   SCENE
      inputs [names]         an effect's input strips (1 or 2)
      text, fontSize, color [r,g,b,a], location [x,y] (0-1), alignX LEFT|CENTER|RIGHT   TEXT; color [r,g,b] COLOR
      strips [names]         META: the strips it gathers (Blender's meta_make)
      settings {prop: value} any other writable RNA property of the made strip (wipe angle, blur_width, size_x...)
    """
    reject_unknown(params, _ADD_STRIP_KEYS, "add_strip", notes={
        "sound": "a sound strip is add_sound's (mode strip) - it decodes the file first and keys its volume",
        "frameEnd": "pass length (frames); an effect with inputs spans its inputs",
        "filepath": "spelled file", "start": "spelled frame", "seq1": "pass inputs [names]",
    })
    tail = _ADDED
    try:
        stype = str(take(params, "type", required=True, kind=str)).strip().upper()
        sc = _seq_scene(params, tail)
        name = take(params, "name", kind=str)
        channel = take_int(params, "channel")
        frame = take_int(params, "frame")
        length = take_int(params, "length")
        settings = _check_settings(params.get("settings"), tail)
    except MifOpError as exc:
        raise MifOpError("%s %s" % (str(exc).rstrip("."), tail) if not str(exc).endswith(tail) else str(exc)) from None
    if stype == "SOUND":
        raise MifOpError("a sound strip is add_sound's (mode strip): it decodes the file before anything is made and "
                         "keys the volume. %s" % tail)
    effects = _effect_types()
    if stype not in _SOURCE_TYPES and stype not in effects:
        raise MifOpError("type must be one of %s on Blender %s, got %r. %s"
                         % (", ".join(list(_SOURCE_TYPES) + effects), bpy.app.version_string, stype, tail))
    wrong = [k for k, types in _TYPE_KEYS.items() if params.get(k) is not None and stype not in types]
    if wrong:
        raise MifOpError("%s do%s not apply to a %s strip (%s). %s"
                         % (", ".join(wrong), "es" if len(wrong) == 1 else "",
                            stype, "; ".join("%s: %s" % (k, "/".join(_TYPE_KEYS[k])) for k in wrong), tail))
    if name is not None and (not name.strip() or any(ord(c) < 32 for c in name)):
        raise MifOpError("'name' is blank or holds a control character, got %r. %s" % (name, tail))
    if length is not None and length < 1:
        raise MifOpError("length must be at least 1 frame, got %d. %s" % (length, tail))
    ed = sc.sequence_editor
    if frame is None:
        frame = int(sc.frame_current)

    if stype == "META":
        return _add_meta(sc, params, name, channel, tail)
    if stype in effects and stype not in _SOURCE_TYPES:
        return _add_effect(sc, ed, stype, params, name, channel, frame, length, settings, tail)
    if length is not None:
        raise MifOpError("length does not apply to a %s strip - its length is its source's. Trim it with edit_strip. %s"
                         % (stype, tail))
    if stype == "SCENE":
        src_name = take(params, "sourceScene", required=True, kind=str)
        src = bpy.data.scenes.get(src_name)
        if src is None:
            raise MifOpError("there is no scene named '%s' for sourceScene. Scenes: %s. %s"
                             % (src_name, ", ".join(s.name for s in bpy.data.scenes), tail))
        if src == sc:
            raise MifOpError("a scene strip cannot show the scene it is in ('%s'). %s" % (sc.name, tail))
        scene_input = params.get("sceneInput")
        if scene_input is not None and str(scene_input).upper() not in ("CAMERA", "SEQUENCER"):
            raise MifOpError("sceneInput is CAMERA or SEQUENCER, got %r. %s" % (scene_input, tail))
        cam = None
        if params.get("sceneCamera") is not None:
            cam = bpy.data.objects.get(params["sceneCamera"])
            if cam is None or cam.type != "CAMERA":
                raise MifOpError("sceneCamera '%s' is not a camera object. %s" % (params["sceneCamera"], tail))
        n = int(src.frame_end) - int(src.frame_start) + 1
        ch = _pick_channel(ed, channel, frame, frame + n, tail)
        ed = sc.sequence_editor_create()
        st = _top_strips(ed).new_scene(name=name or src.name, scene=src, channel=ch, frame_start=frame)
        if scene_input is not None:
            st.scene_input = str(scene_input).upper()
        if cam is not None:
            st.scene_camera = cam
        return _finish(sc, st, settings, name, tail)

    # MOVIE and IMAGE - a file on disk
    raw = take(params, "file", required=True, kind=str)
    path = bpy.path.abspath(raw)
    if not os.path.isfile(path):
        raise MifOpError("no file at '%s'. %s" % (path, tail))
    fit = params.get("fitMethod")
    fits = ("FIT", "FILL", "STRETCH", "ORIGINAL")
    if fit is not None and str(fit).upper() not in fits:
        raise MifOpError("fitMethod is one of %s, got %r. %s" % (", ".join(fits), fit, tail))
    kw = {"fit_method": str(fit).upper()} if fit is not None else {}
    if stype == "IMAGE":
        files = params.get("files")
        if files is not None:
            if not isinstance(files, list) or not files or not all(isinstance(f, str) and f for f in files):
                raise MifOpError("files takes a list of file names in file's folder, got %r. %s" % (files, tail))
            folder = os.path.dirname(path)
            missing = [f for f in files if not os.path.isfile(os.path.join(folder, f))]
            if missing:
                raise MifOpError("files not in '%s': %s. %s" % (folder, ", ".join(missing[:10]), tail))
            if os.path.basename(path) != files[0]:
                raise MifOpError("file must be the first of files ('%s'), got '%s'. %s"
                                 % (files[0], os.path.basename(path), tail))
        n = len(files) if files else 1
        ch = _pick_channel(ed, channel, frame, frame + n, tail)
        ed = sc.sequence_editor_create()
        st = _top_strips(ed).new_image(name=name or os.path.basename(path), filepath=path, channel=ch,
                                       frame_start=frame, **kw)
        for f in (files or [])[1:]:
            st.elements.append(f)
        return _finish(sc, st, settings, name, tail)

    # MOVIE: ITS LENGTH IS KNOWN ONLY ONCE BLENDER HAS OPENED IT, so the strip is made on the asked channel, measured,
    # and removed again when it overlaps - the one refusal here that follows a write, and it leaves nothing behind.
    with_sound = take_bool(params, "withSound", default=True)
    ed = sc.sequence_editor_create()
    coll = _top_strips(ed)
    probe_ch = channel if channel is not None else 128
    try:
        st = coll.new_movie(name=name or os.path.basename(path), filepath=path, channel=probe_ch, frame_start=frame,
                            **kw)
    except (RuntimeError, TypeError) as exc:
        # THE EMPTY EDITOR STAYS (2026-10-08). Scene.sequence_editor_clear() followed by any change of the scene's frame
        # rate crashes Blender in its sequencer evaluation (blender::seq::strip_for_each_recursive, measured on 4.4.0,
        # 5.0.1 and 5.2.1 - even right after sequence_editor_create in the same call), so a refusal leaves the new
        # editor in place, empty, as every 4.4+ default scene already has one.
        raise MifOpError("Blender would not open '%s' as a movie: %s. %s" % (path, exc, tail)) from None
    first, end = _span(st)
    if int(st.frame_duration) < 1 or not getattr(st, "elements", True):
        coll.remove(st)
        raise MifOpError("'%s' opened with no frames - it is not a movie Blender's FFmpeg reads. %s" % (path, tail))
    try:
        ch = _pick_channel(ed, channel, first, end, tail, skip=(st.name,))
        snd_ch = None
        if with_sound:
            taken = _taken(ed, first, end, skip=(st.name,))
            snd_ch = next((c for c in range(ch + 1, 129) if c not in taken), None)
    except MifOpError:
        coll.remove(st)
        # THE EMPTY EDITOR STAYS (2026-10-08). Scene.sequence_editor_clear() followed by any change of the scene's frame
        # rate crashes Blender in its sequencer evaluation (blender::seq::strip_for_each_recursive, measured on 4.4.0,
        # 5.0.1 and 5.2.1 - even right after sequence_editor_create in the same call), so a refusal leaves the new
        # editor in place, empty, as every 4.4+ default scene already has one.
        raise
    st.channel = ch
    sound = None
    sound_note = None
    if with_sound:
        try:
            sound = coll.new_sound(name=st.name + ".sound", filepath=path, channel=snd_ch or 128, frame_start=first)
        except (RuntimeError, TypeError) as exc:
            sound_note = "the file has no sound Blender could open (%s), so no sound strip was made" % exc
        if sound is not None and (sound.sound is None or getattr(sound.sound, "samplerate", 1) == 0):
            coll.remove(sound)
            sound = None
            sound_note = "the file has no audio stream, so no sound strip was made"
    out = _finish(sc, st, settings, name, tail)
    if sound is not None:
        s_first, s_end = _span(sound)
        if s_end > end:
            # MATCHED TO THE PICTURE, as Blender's own movie_strip_add does: the sound strip's length is its seconds
            # at the SCENE's rate and the picture's is its frame count, so at another rate they differ (a 24-frame,
            # one-second movie in a 25 fps scene: picture 24 frames, sound 25) and the sound's tail is cut off.
            _set_handles(sound, s_first, end)
            out["soundTrimmedFrames"] = s_end - end
            s_first, s_end = _span(sound)
        out["soundStrip"] = _strip_row(sc, sound)
        out["soundMatches"] = (s_first == first and s_end == end)
    elif sound_note:
        out["soundNote"] = sound_note
    out["sceneFps"] = round(_fps(sc), 6)
    if abs(float(st.fps) - _fps(sc)) > 1e-3:
        out["fpsNote"] = ("the movie runs at %g fps and the scene at %g, so it plays %s than it was shot - set the "
                          "scene's rate with set_frame_range fps / fpsBase, or retime the strip."
                          % (st.fps, _fps(sc), "faster" if _fps(sc) > st.fps else "slower"))
    return out


def _finish(sc, st, settings, name, tail):
    """Settings, then the reply - a setting Blender refuses takes the new strip away again."""
    try:
        applied = _write_settings(st, settings, "The strip was removed again; NOTHING was added.")
    except MifOpError:
        _top_strips(sc.sequence_editor).remove(st)
        raise
    out = {"scene": sc.name, "strip": _strip_row(sc, st), "name": st.name, "requestedName": name,
           "nameWasSuffixed": bool(name and st.name != name)}
    if applied:
        out["settingsApplied"] = applied
    return out


def _add_effect(sc, ed, stype, params, name, channel, frame, length, settings, tail):
    want_inputs = _EFFECT_INPUTS.get(stype)
    raw_inputs = params.get("inputs")
    if raw_inputs is not None and (not isinstance(raw_inputs, list) or not all(isinstance(n, str) for n in raw_inputs)):
        raise MifOpError("inputs takes a list of strip names, got %r. %s" % (raw_inputs, tail))
    inputs = [_find_strip(sc, n, tail, "inputs") for n in raw_inputs or []]
    if want_inputs is not None and len(inputs) != want_inputs:
        raise MifOpError("a %s effect takes %d input strip(s), got %d. %s" % (stype, want_inputs, len(inputs), tail))
    if inputs:
        if params.get("frame") is not None or length is not None:
            raise MifOpError("frame and length do not apply to an effect with inputs - Blender spans it over its "
                             "inputs (their overlap for two) whatever it is given. %s" % tail)
        spans = [_span(s) for s in inputs]
        first, end = max(a for a, _ in spans), min(b for _, b in spans)
        if end <= first:
            raise MifOpError("inputs %s do not overlap in time (%s), so a %s has no frames to span. %s"
                             % (", ".join(s.name for s in inputs),
                                ", ".join("%s %d-%d" % (s.name, a, b - 1) for s, (a, b) in zip(inputs, spans)),
                                stype, tail))
    else:
        if length is None:
            raise MifOpError("a %s strip needs length (frames). %s" % (stype, tail))
        first, end = frame, frame + length
    text_vals = {}
    if stype == "TEXT":
        text_vals = _text_values(params, tail)
    color = None
    if stype == "COLOR" and params.get("color") is not None:
        color = _vec(params, "color", 3, tail)
    ch = _pick_channel(ed, channel, first, end, tail)
    ed = sc.sequence_editor_create()
    coll = _top_strips(ed)
    fn = coll.bl_rna.functions["new_effect"]
    kw = {}
    if not inputs:
        if "length" in fn.parameters:
            kw["length"] = end - first
        else:
            kw["frame_end"] = end
    for slot, src in zip(("1", "2"), inputs):
        key = ("input" + slot) if ("input" + slot) in fn.parameters else ("seq" + slot)
        kw[key] = src
    try:
        st = coll.new_effect(name=name or stype.title(), type=stype, channel=ch, frame_start=first, **kw)
    except (RuntimeError, TypeError) as exc:
        raise MifOpError("Blender would not make the %s strip: %s. %s" % (stype, exc, tail)) from None
    if color is not None:
        st.color = color
    for attr, value in text_vals.items():
        setattr(st, attr, value)
    return _finish(sc, st, settings, name, tail)


def _text_values(params, tail):
    rna = (getattr(bpy.types, "TextStrip", None) or getattr(bpy.types, "TextSequence")).bl_rna.properties
    out = {}
    if params.get("text") is not None:
        if not isinstance(params["text"], str):
            raise MifOpError("text takes a string, got %r. %s" % (params["text"], tail))
        out["text"] = params["text"]
    size = take_float(params, "fontSize")
    if size is not None:
        lo, hi = rna["font_size"].hard_min, rna["font_size"].hard_max
        if not (lo <= size <= hi):
            raise MifOpError("fontSize must be %g-%g, got %g. %s" % (lo, hi, size, tail))
        out["font_size"] = size
    if params.get("color") is not None:
        out["color"] = _vec(params, "color", 4, tail)
    if params.get("location") is not None:
        out["location"] = _vec(params, "location", 2, tail)
    if params.get("alignX") is not None:
        # align_x up to 4.2, alignment_x from 4.4 - the same three words.
        attr = "alignment_x" if "alignment_x" in rna else "align_x"
        ids = [e.identifier for e in rna[attr].enum_items]
        want = str(params["alignX"]).upper()
        if want not in ids:
            raise MifOpError("alignX is one of %s on this build, got %r. %s" % (", ".join(ids), params["alignX"], tail))
        out[attr] = want
    return out


def _add_meta(sc, params, name, channel, tail):
    if params.get("frame") is not None or params.get("length") is not None or channel is not None:
        raise MifOpError("a META spans the strips it gathers - frame, length and channel do not apply (Blender's "
                         "meta_make places it). %s" % tail)
    raw = params.get("strips")
    if not isinstance(raw, list) or not raw or not all(isinstance(n, str) for n in raw):
        raise MifOpError("a META needs strips [names] to gather, got %r. %s" % (raw, tail))
    ed = sc.sequence_editor
    picked = [_find_strip(sc, n, tail, "strips") for n in raw]
    top = {s.name for s in _top_strips(ed)}
    nested = [s.name for s in picked if s.name not in top]
    if nested:
        raise MifOpError("%s already inside a meta - gather top-level strips. %s" % (", ".join(nested), tail))
    # AN EFFECT GOES WITH ITS INPUTS: meta_make takes the effects fed by a gathered strip too, so they are named.
    pulled = sorted({e for s in picked for e in _effects_fed_by(sc, s)} - set(raw))
    before = {s.name for s in _top_strips(ed)}
    selected = {s.name: s.select for s in _all_strips(ed)}
    active = ed.active_strip
    try:
        with editor_override("SEQUENCE_EDITOR", sc):
            for s in _top_strips(ed):
                s.select = s.name in raw
            ed.active_strip = picked[0]
            res = bpy.ops.sequencer.meta_make()
    except (RuntimeError, MifOpError) as exc:
        raise MifOpError("Blender's meta_make refused: %s. %s" % (exc, tail)) from None
    made = [s for s in _top_strips(ed) if s.name not in before and s.type == "META"]
    for s in _all_strips(ed):
        if s.name in selected:
            s.select = selected[s.name]
    if active is not None and active.name in {s.name for s in _all_strips(ed)}:
        ed.active_strip = active
    if "FINISHED" not in res or len(made) != 1:
        raise MifOpError("meta_make answered %s and made %d meta strip(s). %s" % (sorted(res), len(made), tail))
    meta = made[0]
    if name:
        meta.name = name
    out = {"scene": sc.name, "strip": _strip_row(sc, meta), "name": meta.name, "requestedName": name,
           "nameWasSuffixed": bool(name and meta.name != name)}
    if pulled:
        out["effectsGathered"] = pulled
    return out


# ---------------------------------------------------------------------------------------------------------------
# edit_strip
# ---------------------------------------------------------------------------------------------------------------

_EDIT_STRIP_KEYS = {"strip", "scene", "channel", "frame", "trimStart", "trimEnd", "slip", "split", "splitMethod",
                    "delete", "blendType", "opacity", "crop", "transform", "saturation", "multiply", "mute", "lock",
                    "volume", "fadeIn", "fadeOut", "settings", "modifiers", "swapInputs", "rename", "duplicate",
                    "connect", "disconnect", "proxy"}
_TRANSFORM_KEYS = {"offsetX": "offset_x", "offsetY": "offset_y", "scaleX": "scale_x", "scaleY": "scale_y",
                   "rotation": "rotation", "origin": "origin", "filter": "filter"}
_CROP_KEYS = {"minX": "min_x", "maxX": "max_x", "minY": "min_y", "maxY": "max_y"}
_MOD_KEYS = {"name", "type", "remove", "enabled", "settings", "index", "curve"}


def _set_handles(st, first, end):
    """Write a strip's first frame and the frame after its last, through the build's own names."""
    if "left_handle" in st.bl_rna.properties:
        # 5.2: the handles; the right one first when the strip grows to the right, so neither clamps on the other.
        if end > _span(st)[1]:
            st.right_handle, st.left_handle = end, first
        else:
            st.left_handle, st.right_handle = first, end
    else:
        if end > _span(st)[1]:
            st.frame_final_end, st.frame_final_start = end, first
        else:
            st.frame_final_start, st.frame_final_end = first, end


def _content_range(st):
    """(first, end) of the frames the strip's source holds, which a trim cannot go past - effects and colors have no
    such bound (None)."""
    if st.type in ("MOVIE", "IMAGE", "SCENE", "SOUND", "META", "MOVIECLIP", "MASK"):
        start = int(st.frame_start)
        return start, start + int(st.frame_duration)
    return None


def op_edit_strip(params):
    """Edit one sequencer strip: move, trim, slip, split, delete; blend, opacity, crop, transform, color; mute, lock,
    volume, fades, any RNA setting; and its modifier stack.

    THE GAP: no op could move, trim, cut, fade or grade a strip - every edit after placing one was raw Python.

    params:
      strip (str, required), scene (str)
      channel (int), frame (int: the new first frame - the strip moves whole), rename (str)
      trimStart / trimEnd (int)   the new first / last frame shown, inside what the source holds
      slip (int)                  move the content under fixed handles by this many frames
      split (int), splitMethod SOFT|HARD   cut at a frame; the reply names the right-hand strip
      delete (bool)               remove the strip
      blendType, opacity (0-1), saturation, multiply, mute, lock, volume (sound, meta and scene strips)
      crop {minX, maxX, minY, maxY} pixels; transform {offsetX, offsetY, scaleX, scaleY, rotation (degrees),
      origin [x, y], filter}
      fadeIn / fadeOut (frames)   keys opacity (volume on a sound strip) 0 -> 1 over the head / 1 -> 0 over the tail,
                                  as Blender's fades_add does, on the scene's action
      swapInputs (bool)           a two-input effect's inputs swapped
      settings {prop: value}      any other writable RNA property (wipe angle, blur_width, size_x, text...)
      duplicate {name?, channel?, frame?}   a copy of the strip (Blender's sequencer.duplicate, which leaves it on
                                  top of the original), moved to frame (default the same) on channel (default the
                                  lowest free); its own edit, as delete is; the reply's duplicate is the copy's row
      connect [strip, ...]        connect it to these strips so they move as one (where Strip.connections reads back:
                                  5.2; 5.0 and 4.4 have the operator and no read, and are refused); disconnect frees it
      proxy {sizes [25|50|75|100], quality (1-100), overwrite, build}   a MOVIE or IMAGE strip's proxies turned on at
                                  those sizes; build (default true) runs Blender's rebuild_proxy and the reply lists
                                  the files in the proxy folder
      modifiers [{name, type?, remove?, enabled?, settings?, index?, curve?}]   type adds one (COLOR_BALANCE, CURVES,
                                  HUE_CORRECT, BRIGHT_CONTRAST, MASK, WHITE_BALANCE, TONEMAP; SOUND_EQUALIZER;
                                  COMPOSITOR from 5.0 - the build's own list); index moves it; curve [[x, y], ...]
                                  replaces a CURVES modifier's combined curve
    """
    reject_unknown(params, _EDIT_STRIP_KEYS, "edit_strip", notes={
        "name": "the strip is named by strip; a new name is rename",
        "start": "spelled frame (moves the whole strip) or trimStart (moves its first frame)",
        "fade": "spelled fadeIn / fadeOut, in frames",
    })
    tail = _CHANGED
    sc = _seq_scene(params, tail)
    name = take(params, "strip", required=True, kind=str)
    st = _find_strip(sc, name, tail)
    ed = sc.sequence_editor
    try:
        channel = take_int(params, "channel")
        frame = take_int(params, "frame")
        trim_start = take_int(params, "trimStart")
        trim_end = take_int(params, "trimEnd")
        slip = take_int(params, "slip")
        split = take_int(params, "split")
        delete = take_bool(params, "delete", default=False)
        swap = take_bool(params, "swapInputs", default=False)
        fade_in = take_int(params, "fadeIn")
        fade_out = take_int(params, "fadeOut")
        opacity = take_float(params, "opacity")
        saturation = take_float(params, "saturation")
        multiply = take_float(params, "multiply")
        volume = take_float(params, "volume")
        mute = take_bool(params, "mute") if params.get("mute") is not None else None
        lock = take_bool(params, "lock") if params.get("lock") is not None else None
        rename = take(params, "rename", kind=str)
        settings = _check_settings(params.get("settings"), tail)
    except MifOpError as exc:
        raise MifOpError("%s %s" % (str(exc).rstrip("."), tail)) from None
    others = [k for k in _EDIT_STRIP_KEYS - {"strip", "scene", "delete"} if params.get(k) is not None]
    if params.get("duplicate") is not None:
        rest = [k for k in others if k != "duplicate"]
        if rest or delete:
            raise MifOpError("duplicate is its own edit - %s go in another call, on the original or the copy. %s"
                             % (", ".join(sorted(rest + (["delete"] if delete else []))), tail))
        return _duplicate(sc, st, params["duplicate"], tail)
    if delete:
        if others:
            raise MifOpError("delete removes the strip - %s would be lost with it. %s" % (", ".join(sorted(others)),
                                                                                           tail))
        fed = _effects_fed_by(sc, st)
        if fed:
            raise MifOpError("strip '%s' feeds the effect(s) %s, which Blender would delete with it - delete them "
                             "first, or pass them too. %s" % (st.name, ", ".join(fed), tail))
        row = _strip_row(sc, st)
        parent = None
        try:
            parent = st.parent_meta()
        except (AttributeError, TypeError, RuntimeError):
            parent = None
        holder = (parent.strips if hasattr(parent, "strips") else parent.sequences) if parent else _top_strips(ed)
        holder.remove(st)
        return {"scene": sc.name, "deleted": row, "strips": len(_all_strips(ed))}
    if st.lock and lock is not False and others:
        raise MifOpError("strip '%s' is locked - pass lock false with the edit to unlock it. %s" % (st.name, tail))

    first, end = _span(st)
    if _inputs(st) and any(params.get(k) is not None for k in ("frame", "trimStart", "trimEnd", "slip", "split")):
        raise MifOpError("'%s' is an effect over %s, and Blender spans it over its inputs - move or trim those. %s"
                         % (st.name, ", ".join(_inputs(st)), tail))
    if split is not None:
        if any(params.get(k) is not None for k in ("channel", "frame", "trimStart", "trimEnd", "slip")):
            raise MifOpError("split is its own edit - move or trim in another call, so it is clear which half "
                             "moves. %s" % tail)
        if not (first < split < end):
            raise MifOpError("split frame %d is not inside strip '%s' (frames %d-%d). %s"
                             % (split, st.name, first, end - 1, tail))
        fed = _effects_fed_by(sc, st)
        if fed:
            raise MifOpError("strip '%s' feeds the effect(s) %s; splitting an effect's input and then moving its other "
                             "input crashes Blender (4.2.23 and 5.2.2, measured). Split the effect's inputs before "
                             "making the effect. %s" % (st.name, ", ".join(fed), tail))
    method = str(params.get("splitMethod") or "SOFT").upper()
    if method not in ("SOFT", "HARD"):
        raise MifOpError("splitMethod is SOFT or HARD, got %r. %s" % (params.get("splitMethod"), tail))

    # THE NEW RANGE, worked out and checked before anything moves.
    new_first, new_end = first, end
    shift = 0
    if frame is not None:
        shift = frame - first
        new_first, new_end = frame, end + shift
    # trimStart / trimEnd are timeline frames, read after the move.
    if trim_start is not None:
        new_first = trim_start
    if trim_end is not None:
        new_end = trim_end + 1
    if new_end <= new_first:
        raise MifOpError("the strip would end (%d) before it starts (%d). %s" % (new_end - 1, new_first, tail))
    content = _content_range(st)
    if content is not None and (trim_start is not None or trim_end is not None or slip is not None):
        c_first, c_end = content[0] + shift + (slip or 0), content[1] + shift + (slip or 0)
        if new_first < c_first or new_end > c_end:
            raise MifOpError("strip '%s' holds source frames %d-%d%s; frames %d-%d go past them. %s"
                             % (st.name, c_first, c_end - 1, " after the slip" if slip else "", new_first,
                                new_end - 1, tail))
    elif slip is not None:
        raise MifOpError("slip moves a source's content under the handles, and a %s strip has no source. %s"
                         % (st.type, tail))
    want_ch = channel if channel is not None else int(st.channel)
    if (new_first, new_end, want_ch) != (first, end, int(st.channel)):
        _pick_channel(ed, want_ch, new_first, new_end, tail, skip=(st.name,))

    checks = {}
    for key, val, attr in (("opacity", opacity, "blend_alpha"), ("saturation", saturation, "color_saturation"),
                           ("multiply", multiply, "color_multiply"), ("volume", volume, "volume")):
        if val is None:
            continue
        prop = st.bl_rna.properties.get(attr)
        if prop is None:
            raise MifOpError("a %s strip has no %s. %s" % (st.type, key, tail))
        if not (prop.hard_min <= val <= prop.hard_max):
            raise MifOpError("%s must be %g-%g, got %g. %s" % (key, prop.hard_min, prop.hard_max, val, tail))
        checks[attr] = val
    blend = params.get("blendType")
    if blend is not None:
        ids = [e.identifier for e in st.bl_rna.properties["blend_type"].enum_items]
        if str(blend).upper() not in ids:
            raise MifOpError("blendType is one of %s, got %r. %s" % (", ".join(ids), blend, tail))
        checks["blend_type"] = str(blend).upper()
    crop = _sub_dict(params, "crop", _CROP_KEYS, tail)
    xform = _sub_dict(params, "transform", _TRANSFORM_KEYS, tail)
    for fade, key in ((fade_in, "fadeIn"), (fade_out, "fadeOut")):
        if fade is not None and not (1 <= fade <= new_end - new_first):
            raise MifOpError("%s must be 1-%d frames (the strip's length), got %d. %s"
                             % (key, new_end - new_first, fade, tail))
    fade_attr = "volume" if st.type == "SOUND" else "blend_alpha"
    if swap and len(_inputs(st)) != 2:
        raise MifOpError("swapInputs needs a two-input effect; '%s' is a %s. %s" % (st.name, st.type, tail))
    mods = _check_modifiers(sc, st, params.get("modifiers"), tail)
    conn_to, disconnect = _check_connect(sc, st, params, tail)
    proxy = _check_proxy(st, params.get("proxy"), tail)
    if rename is not None and (not rename.strip() or any(ord(c) < 32 for c in rename)):
        raise MifOpError("rename is blank or holds a control character. %s" % tail)

    # ------------------------------------------------------------------ the writes
    out = {"scene": sc.name}
    if split is not None:
        kw = {"frame": split, "split_method": method}
        right = st.split(**kw)
        if right is None:
            raise MifOpError("Blender's split returned nothing at frame %d. %s" % (split, tail))
        new_end = split
        out["splitInto"] = [st.name, right.name]
        out["right"] = _strip_row(sc, right)
    if channel is not None and channel != int(st.channel):
        st.channel = channel
    if shift:
        st.frame_start = st.frame_start + shift
    if slip:
        keep = _span(st)
        st.frame_start = st.frame_start + slip
        _set_handles(st, *keep)
    if (new_first, new_end) != _span(st):
        _set_handles(st, new_first, new_end)
    for attr, val in checks.items():
        setattr(st, attr, val)
    for key, attr in _CROP_KEYS.items():
        if key in crop:
            setattr(st.crop, attr, int(crop[key]))
    for key, attr in _TRANSFORM_KEYS.items():
        if key in xform:
            val = xform[key]
            if key == "rotation":
                val = math.radians(val)
            setattr(st.transform, attr, val)
    if mute is not None:
        st.mute = mute
    if lock is not None:
        st.lock = lock
    if swap:
        a, b = st.input_1, st.input_2
        st.input_1, st.input_2 = b, a
    if rename:
        st.name = rename
    got = _span(st)
    if (got, int(st.channel)) != ((new_first, new_end), want_ch):
        raise MifOpError("Blender placed strip '%s' at frames %d-%d on channel %d, not %d-%d on channel %d - the edit "
                         "was kept; move it again." % (st.name, got[0], got[1] - 1, st.channel, new_first,
                                                        new_end - 1, want_ch))
    applied = _write_settings(st, settings, "Edits listed before it in this call were applied.")
    if applied:
        out["settingsApplied"] = applied
    if fade_in or fade_out:
        out["fades"] = _write_fades(sc, st, fade_attr, fade_in, fade_out)
    if mods:
        out["modifiers"] = _apply_modifiers(sc, st, mods)
    if disconnect or conn_to:
        out["connections"] = _write_connections(sc, st, conn_to, disconnect)
    if proxy:
        out["proxy"] = _write_proxy(sc, st, proxy)
    out["strip"] = _strip_row(sc, st)
    if crop or xform:
        out["crop"] = {k: int(getattr(st.crop, a)) for k, a in _CROP_KEYS.items()}
        t = st.transform
        out["transform"] = {"offsetX": round(t.offset_x, 4), "offsetY": round(t.offset_y, 4),
                            "scaleX": round(t.scale_x, 6), "scaleY": round(t.scale_y, 6),
                            "rotation": round(math.degrees(t.rotation), 4), "origin": rnd(t.origin), "filter": t.filter}
    return out


def _duplicate(sc, st, raw, tail):
    """A copy of st through Blender's sequencer.duplicate (a borrowed Sequencer area), then placed."""
    if not isinstance(raw, dict) or set(raw) - {"name", "channel", "frame"}:
        raise MifOpError("duplicate takes {name?, channel?, frame?}, got %r. %s" % (raw, tail))
    if st.type == "META" or _inputs(st):
        raise MifOpError("'%s' is a %s - duplicating a meta or an effect copies what it holds or feeds on as well; "
                         "duplicate its sources and make it again. %s" % (st.name, st.type, tail))
    name = raw.get("name")
    if name is not None and (not isinstance(name, str) or not name.strip() or any(ord(c) < 32 for c in name)):
        raise MifOpError("duplicate.name is blank or not text, got %r. %s" % (name, tail))
    try:
        frame = finite_int(raw["frame"], "duplicate.frame") if raw.get("frame") is not None else None
        channel = finite_int(raw["channel"], "duplicate.channel") if raw.get("channel") is not None else None
    except MifOpError as exc:
        raise MifOpError("%s %s" % (str(exc).rstrip("."), tail)) from None
    ed = sc.sequence_editor
    first, end = _span(st)
    shift = (frame - first) if frame is not None else 0
    to_ch = _pick_channel(ed, channel, first + shift, end + shift, tail)
    if name is not None and name in _all_strips(ed):
        raise MifOpError("a strip named '%s' already exists in scene '%s'. %s" % (name, sc.name, tail))
    before = {s.as_pointer() for s in _all_strips(ed)}
    sel = [s for s in _all_strips(ed) if s.select]
    active = ed.active_strip
    try:
        _select_strips_only(ed, [st])
        ed.active_strip = st
        with editor_override("SEQUENCE_EDITOR", sc):
            res = bpy.ops.sequencer.duplicate()
    finally:
        new = [s for s in _all_strips(ed) if s.as_pointer() not in before]
        _select_strips_only(ed, sel)
        ed.active_strip = active
    if "FINISHED" not in res or len(new) != 1:
        for s in new:
            _top_strips(ed).remove(s)
        raise MifOpError("Blender's duplicate answered %s with %d new strip(s) - removed again. %s"
                         % (sorted(res), len(new), tail))
    cp = new[0]
    cp.channel = to_ch
    if shift:
        cp.frame_start = cp.frame_start + shift
    if name:
        cp.name = name
    got = _span(cp)
    if (got, int(cp.channel)) != ((first + shift, end + shift), to_ch):
        raise MifOpError("Blender placed the copy '%s' at frames %d-%d on channel %d, not %d-%d on channel %d - it was "
                         "kept; move it with edit_strip." % (cp.name, got[0], got[1] - 1, cp.channel, first + shift,
                                                             end + shift - 1, to_ch))
    return {"scene": sc.name, "original": _strip_row(sc, st), "duplicate": _strip_row(sc, cp),
            "requestedName": name, "nameWasSuffixed": bool(name and cp.name != name)}


def _check_connect(sc, st, params, tail):
    raw, disc = params.get("connect"), params.get("disconnect")
    if raw is None and disc is None:
        return [], False
    if not _connect_supported():
        raise MifOpError("strip connections are read back from Strip.connections, which Blender %s does not have "
                         "(5.2 has it; 5.0 and 4.4 have the operators but no way to read what they did, 4.2 neither). %s"
                         % (bpy.app.version_string, tail))
    disc = take_bool(params, "disconnect", default=False)
    if raw is not None and disc:
        raise MifOpError("pass connect OR disconnect - one call either joins strips or frees this one. %s" % tail)
    names = []
    if raw is not None:
        if not isinstance(raw, list) or not raw or not all(isinstance(n, str) for n in raw):
            raise MifOpError("connect takes a list of strip names, got %r. %s" % (raw, tail))
        for n in raw:
            if n == st.name:
                raise MifOpError("connect names '%s' itself. %s" % (n, tail))
            _find_strip(sc, n, tail, key="connect")
        names = list(dict.fromkeys(raw))
    return names, disc


def _write_connections(sc, st, names, disconnect):
    ed = sc.sequence_editor
    sel = [s for s in _all_strips(ed) if s.select]
    active = ed.active_strip
    try:
        group = [st] + [_all_strips(ed)[n] for n in names]
        _select_strips_only(ed, group)
        ed.active_strip = st
        with editor_override("SEQUENCE_EDITOR", sc):
            res = bpy.ops.sequencer.disconnect() if disconnect else bpy.ops.sequencer.connect()
    finally:
        _select_strips_only(ed, sel)
        ed.active_strip = active
    now = sorted(s.name for s in st.connections)
    want_ok = (not now) if disconnect else set(names) <= set(now)
    if "FINISHED" not in res or not want_ok:
        raise MifOpError("Blender's %s answered %s and '%s' is connected to %s - edits before it in this call were "
                         "applied." % ("disconnect" if disconnect else "connect", sorted(res), st.name, now or "nothing"))
    return {"action": "disconnect" if disconnect else "connect", "connected": now}


def _check_proxy(st, raw, tail):
    if raw is None:
        return None
    if st.type not in _PROXY_TYPES:
        raise MifOpError("proxies are built for MOVIE and IMAGE strips; '%s' is a %s. %s" % (st.name, st.type, tail))
    if not isinstance(raw, dict) or not raw or set(raw) - _PROXY_KEYS:
        raise MifOpError("proxy takes {%s}, got %r. %s" % (", ".join(sorted(_PROXY_KEYS)), raw, tail))
    sizes = raw.get("sizes", [25])
    if not isinstance(sizes, list) or not sizes or any(s not in _PROXY_SIZES for s in sizes):
        raise MifOpError("proxy.sizes is a list of 25, 50, 75 and 100 (percent of the source), got %r. %s"
                         % (sizes, tail))
    out = {"sizes": sorted(set(sizes)), "build": bool(raw.get("build", True)), "overwrite": raw.get("overwrite")}
    if raw.get("quality") is not None:
        q = finite_int(raw["quality"], "proxy.quality")
        if not 1 <= q <= 100:
            raise MifOpError("proxy.quality is 1-100 (JPEG quality of the proxy), got %d. %s" % (q, tail))
        out["quality"] = q
    return out


def _write_proxy(sc, st, cfg):
    st.use_proxy = True
    px = st.proxy
    for n in _PROXY_SIZES:
        setattr(px, "build_%d" % n, n in cfg["sizes"])
    if cfg.get("quality") is not None:
        px.quality = cfg["quality"]
    if cfg.get("overwrite") is not None:
        px.use_overwrite = bool(cfg["overwrite"])
    out = {"rebuilt": False}
    if cfg["build"]:
        ed = sc.sequence_editor
        sel = [s for s in _all_strips(ed) if s.select]
        active = ed.active_strip
        try:
            _select_strips_only(ed, [st])
            ed.active_strip = st
            with editor_override("SEQUENCE_EDITOR", sc):
                res = bpy.ops.sequencer.rebuild_proxy()
        finally:
            _select_strips_only(ed, sel)
            ed.active_strip = active
        out["rebuilt"] = "FINISHED" in res
    out.update(_proxy_row(st))
    missing = [n for n in cfg["sizes"] if not out["built"].get(n)] if isinstance(out.get("built"), dict) else []
    if cfg["build"] and missing:
        out["note"] = ("Blender's rebuild_proxy left no proxy file for size(s) %s in %s." % (missing, out["directory"]))
    return out


def _sub_dict(params, key, names, tail):
    val = params.get(key)
    if val is None:
        return {}
    if not isinstance(val, dict) or not val or set(val) - set(names):
        raise MifOpError("%s takes {%s}, got %r. %s" % (key, ", ".join(names), val, tail))
    out = {}
    for k, v in val.items():
        if k == "filter":
            if not isinstance(v, str):
                raise MifOpError("transform.filter takes a word, got %r. %s" % (v, tail))
            out[k] = v.upper()
        elif k == "origin":
            if not isinstance(v, (list, tuple)) or len(v) != 2:
                raise MifOpError("transform.origin takes [x, y], got %r. %s" % (v, tail))
            out[k] = finite_floats(v, "transform.origin")
        else:
            try:
                out[k] = finite_int(v, "%s.%s" % (key, k)) if key == "crop" else float(finite_floats([v], k)[0])
            except MifOpError as exc:
                raise MifOpError("%s %s" % (str(exc).rstrip("."), tail)) from None
            if key == "crop" and out[k] < 0:
                raise MifOpError("crop.%s is pixels cut off, at least 0, got %d. %s" % (k, out[k], tail))
    return out


def _write_fades(sc, st, attr, fade_in, fade_out):
    """Keys on the scene's action the way Blender's sequencer.fades_add writes them (bl_operators/sequencer.py):
    0 at the first frame and 1 fade_in frames later; 1 fade_out frames before the end and 0 at the end."""
    first, end = _span(st)
    full = getattr(st, attr) if attr == "volume" else 1.0
    keys = []
    if fade_in:
        keys += [(first, 0.0), (first + fade_in, full)]
    if fade_out:
        keys += [(end - fade_out, full), (end, 0.0)]
    for frame, value in keys:
        setattr(st, attr, value)
        st.keyframe_insert(attr, frame=frame)
    setattr(st, attr, full)
    path = st.path_from_id(attr)
    curve = next((fc for fc in _fcurves(sc) if fc.data_path == path), None)
    rows = []
    if curve is not None:
        for kp in curve.keyframe_points:
            kp.interpolation = "LINEAR"
        curve.update()
        rows = [[round(float(kp.co[0]), 4), round(float(kp.co[1]), 6)] for kp in curve.keyframe_points]
    return {"property": attr, "dataPath": path, "keys": rows}


def _modifier_types():
    cls = getattr(bpy.types, "StripModifier", None) or getattr(bpy.types, "SequenceModifier")
    return [e.identifier for e in cls.bl_rna.properties["type"].enum_items]


def _check_modifiers(sc, st, raw, tail):
    if raw is None:
        return []
    if not isinstance(raw, list) or not raw:
        raise MifOpError("modifiers takes a list of {name, type?, remove?, enabled?, settings?, index?, curve?}. %s"
                         % tail)
    types = _modifier_types()
    have = [m.name for m in st.modifiers]
    future = list(have)
    out = []
    for i, row in enumerate(raw):
        if not isinstance(row, dict) or set(row) - _MOD_KEYS or not isinstance(row.get("name"), str):
            raise MifOpError("modifiers[%d] takes {%s} with a name, got %r. %s"
                             % (i, ", ".join(sorted(_MOD_KEYS)), row, tail))
        mname = row["name"]
        if row.get("type") is not None:
            mtype = str(row["type"]).upper()
            if mtype not in types:
                raise MifOpError("modifiers[%d].type is one of %s on Blender %s, got %r. %s"
                                 % (i, ", ".join(types), bpy.app.version_string, row["type"], tail))
            if row.get("remove"):
                raise MifOpError("modifiers[%d] both adds and removes '%s'. %s" % (i, mname, tail))
            if mname in future:
                raise MifOpError("strip '%s' already has a modifier '%s'. %s" % (st.name, mname, tail))
            future.append(mname)
        elif mname not in future:
            raise MifOpError("strip '%s' has no modifier '%s' (it has: %s) - pass type to add one. %s"
                             % (st.name, mname, ", ".join(future) or "none", tail))
        elif row.get("remove"):
            future.remove(mname)
        if row.get("settings") is not None and not isinstance(row["settings"], dict):
            raise MifOpError("modifiers[%d].settings takes {property: value}. %s" % (i, tail))
        if row.get("index") is not None:
            idx = finite_int(row["index"], "modifiers[%d].index" % i)
            if not (0 <= idx < len(future)):
                raise MifOpError("modifiers[%d].index must be 0-%d, got %d. %s" % (i, len(future) - 1, idx, tail))
        if row.get("curve") is not None:
            pts = row["curve"]
            if (row.get("type") or "").upper() not in ("CURVES", "") or not isinstance(pts, list) or len(pts) < 2:
                raise MifOpError("modifiers[%d].curve takes [[x, y], ...] (2 or more points, 0-1) for a CURVES "
                                 "modifier. %s" % (i, tail))
            for p in pts:
                if not isinstance(p, (list, tuple)) or len(p) != 2 or not all(0 <= float(v) <= 1 for v in p):
                    raise MifOpError("modifiers[%d].curve point %r is not [x, y] in 0-1. %s" % (i, p, tail))
        out.append(row)
    return out


def _apply_modifiers(sc, st, rows):
    done = []
    with sequencer_scene_guard(sc):
        for row in rows:
            mname = row["name"]
            if row.get("type") is not None:
                md = st.modifiers.new(name=mname, type=str(row["type"]).upper())
                md.name = mname
                action = "added"
            else:
                md = st.modifiers.get(mname)
                action = "edited"
            if row.get("remove"):
                st.modifiers.remove(md)
                done.append({"name": mname, "action": "removed"})
                continue
            if row.get("enabled") is not None:
                md.mute = not bool(row["enabled"])
            if row.get("curve") is not None:
                if md.type != "CURVES":
                    raise MifOpError("modifier '%s' is %s, and curve is for CURVES. Edits before it in this call were "
                                     "applied." % (mname, md.type))
                _write_curve(md, row["curve"])
            applied = {}
            for key, value in (row.get("settings") or {}).items():
                applied[key] = write_rna_property(md, key, value, "Edits before it in this call were applied.",
                                                  "modifier '%s' (%s)" % (mname, md.type), md.type,
                                                  frozenset({"name", "type"}))
            if row.get("index") is not None:
                _move_modifier(sc, st, md, int(row["index"]))
            entry = {"name": md.name, "type": md.type, "action": action}
            if applied:
                entry["settingsApplied"] = applied
            done.append(entry)
    return {"applied": done, "stack": _modifier_rows(st)}


def _write_curve(md, pts):
    cm = md.curve_mapping
    curve = cm.curves[3] if len(cm.curves) > 3 else cm.curves[0]
    pts = sorted((float(x), float(y)) for x, y in pts)
    while len(curve.points) > 2:
        curve.points.remove(curve.points[-1])
    curve.points[0].location = pts[0]
    curve.points[1].location = pts[-1]
    for x, y in pts[1:-1]:
        curve.points.new(x, y)
    cm.update()


def _move_modifier(sc, st, md, index):
    """Move a modifier to a stack position: Blender's strip_modifier_move (UP / DOWN, every build) in a borrowed
    Sequencer area, once per step - StripModifiers has no move of its own."""
    names = [m.name for m in st.modifiers]
    cur = names.index(md.name)
    ed = sc.sequence_editor
    active = ed.active_strip
    with editor_override("SEQUENCE_EDITOR", sc):
        ed.active_strip = st
        while cur != index:
            direction = "UP" if index < cur else "DOWN"
            bpy.ops.sequencer.strip_modifier_move(name=md.name, direction=direction)
            now = [m.name for m in st.modifiers].index(md.name)
            if now == cur:
                break
            cur = now
    ed.active_strip = active
    if cur != index:
        raise MifOpError("modifier '%s' stopped at position %d, not %d - Blender's move would not take it further."
                         % (md.name, cur, index))


# ---------------------------------------------------------------------------------------------------------------
# list_strips and set_sequencer
# ---------------------------------------------------------------------------------------------------------------

def _channel_rows(ed):
    rows = []
    for i, ch in enumerate(ed.channels):
        if i == 0:
            continue
        if ch.mute or ch.lock or ch.name != "Channel %d" % i:
            rows.append({"channel": i, "name": ch.name, "mute": bool(ch.mute), "lock": bool(ch.lock)})
    return rows


def _sequencer_state():
    ws = _workspace()
    if ws is None or "sequencer_scene" not in ws.bl_rna.properties:
        return {"sequencerScene": None, "sequencerSceneNote": "Blender %s has no workspace sequencer scene - the "
                "Sequencer shows the window's scene (5.0 added it)." % bpy.app.version_string}
    return {"workspace": ws.name, "sequencerScene": ws.sequencer_scene.name if ws.sequencer_scene else None,
            "sceneTimeSync": bool(ws.use_scene_time_sync)}


def op_list_strips(params):
    """Every strip in a scene's sequencer (inside metas too), in channel and time order, with its range, source,
    inputs and modifiers; the named channels; and on 5.0+ the workspace's sequencer scene. Read-only.

    params: scene (str), limit (int, default 200)
    """
    reject_unknown(params, {"scene", "limit"}, "list_strips")
    sc = _seq_scene(params, "")
    limit = take_int(params, "limit", default=200)
    if limit < 1:
        raise MifOpError("limit must be at least 1, got %d." % limit)
    ed = sc.sequence_editor
    rows = sorted((_strip_row(sc, s) for s in _all_strips(ed)), key=lambda r: (r["frameStart"], r["channel"])) \
        if ed is not None else []
    out = {"scene": sc.name, "count": len(rows), "strips": rows[:limit], "truncated": len(rows) > limit,
           "sceneRange": [int(sc.frame_start), int(sc.frame_end)], "fps": round(_fps(sc), 6)}
    if rows:
        out["stripsRange"] = [min(r["frameStart"] for r in rows), max(r["frameEnd"] for r in rows)]
    if ed is not None:
        out["channels"] = _channel_rows(ed)
    out.update(_sequencer_state())
    return out


def op_set_sequencer(params):
    """The sequencer's own settings: channel names, mute and lock; and on 5.0+ the workspace's sequencer scene and
    Sync Scene Time.

    params:
      scene (str)               whose channels (default as add_strip)
      channels {"<n>": {name?, mute?, lock?}}
      sequencerScene (str)      5.0+: the scene the Sequencer shows and edits
      sceneTimeSync (bool)      5.0+: the scene's frame follows the sequencer's
    """
    reject_unknown(params, {"scene", "channels", "sequencerScene", "sceneTimeSync"}, "set_sequencer")
    tail = _CHANGED
    sc = _seq_scene(params, tail)
    ws = _workspace()
    has_ws = ws is not None and "sequencer_scene" in ws.bl_rna.properties
    if not has_ws and (params.get("sequencerScene") is not None or params.get("sceneTimeSync") is not None):
        raise MifOpError("sequencerScene and sceneTimeSync are 5.0's (WorkSpace.sequencer_scene, use_scene_time_sync); "
                         "on Blender %s the Sequencer always shows the window's scene - set that with "
                         "set_current_scene. %s" % (bpy.app.version_string, tail))
    target = None
    if params.get("sequencerScene") is not None:
        target = bpy.data.scenes.get(str(params["sequencerScene"]))
        if target is None:
            raise MifOpError("there is no scene named '%s'. %s" % (params["sequencerScene"], tail))
    sync = take_bool(params, "sceneTimeSync") if params.get("sceneTimeSync") is not None else None
    chans = params.get("channels")
    writes = []
    if chans is not None:
        if not isinstance(chans, dict) or not chans:
            raise MifOpError("channels takes {\"<n>\": {name, mute, lock}}, got %r. %s" % (chans, tail))
        for key, row in chans.items():
            try:
                n = int(key)
            except (TypeError, ValueError):
                raise MifOpError("channels key %r is not a channel number. %s" % (key, tail)) from None
            if not (1 <= n <= 128):
                raise MifOpError("channel %d is outside 1-128. %s" % (n, tail))
            if not isinstance(row, dict) or not row or set(row) - {"name", "mute", "lock"}:
                raise MifOpError("channels[%s] takes {name, mute, lock}, got %r. %s" % (key, row, tail))
            if "name" in row and (not isinstance(row["name"], str) or not row["name"].strip()):
                raise MifOpError("channels[%s].name must be text. %s" % (key, tail))
            for flag in ("mute", "lock"):
                if flag in row and not isinstance(row[flag], bool):
                    raise MifOpError("channels[%s].%s must be true or false. %s" % (key, flag, tail))
            writes.append((n, row))
    if not writes and target is None and sync is None:
        raise MifOpError("set_sequencer was given nothing to set - channels, sequencerScene or sceneTimeSync. %s"
                         % tail)
    if writes:
        ed = sc.sequence_editor_create()
        for n, row in writes:
            ch = ed.channels[n]
            for k, v in row.items():
                setattr(ch, k, v)
    if target is not None:
        ws.sequencer_scene = target
    if sync is not None:
        ws.use_scene_time_sync = sync
    out = {"scene": sc.name}
    if sc.sequence_editor is not None:
        out["channels"] = _channel_rows(sc.sequence_editor)
    out.update(_sequencer_state())
    return out


def strips_range(sc):
    """(first, last) frame over every strip in the scene's sequencer, or None - what sequencer.set_range_to_strips
    sets the scene range to."""
    ed = sc.sequence_editor
    spans = [_span(s) for s in _all_strips(ed)] if ed is not None else []
    if not spans:
        return None
    return min(a for a, _ in spans), max(b for _, b in spans) - 1


# ---------------------------------------------------------------------------------------------------------------
# MOVIE CLIPS AND TRACKING
#
# MEASURED 2026-10-06 on 4.2.23, 4.4.0, 5.0.1 and 5.2.2 as the PyPI bpy module: clip.detect_features, track_markers,
# solve_camera, delete_track, set_plane, set_origin, set_axis, set_scale and set_solution_scale all run headless in a
# borrowed Clip Editor (ops_common.editor_override) with its space's clip set; the settings, tracks, markers, objects
# and reconstruction are plain RNA. MovieTrackingTracks has new() and no remove(), so a track is deleted by
# delete_track on a selection. A synthetic sequence of 30 frames of dots solved at 2.97 px with every detected track
# and at 1.0 px after four rounds that each deleted the worst tenth - the clean-up rounds are native for that reason.
# ---------------------------------------------------------------------------------------------------------------

def _get_clip(name, tail):
    # getattr with defaults: the offline refusal sweep (test_blender_refusals B111) runs on a stub bpy with neither
    clips = getattr(bpy.data, "movieclips", None) or []
    if name is not None:
        clip = next((c for c in clips if c.name == name), None) if isinstance(name, str) else None
        if clip is None:
            raise MifOpError("no movie clip named %r. This file has: %s. %s"
                             % (name, ", ".join(c.name for c in clips) or "(none - load_clip loads one)", tail))
        return clip
    clip = getattr(getattr(bpy.context, "scene", None), "active_clip", None)
    if clip is None and len(clips) == 1:
        clip = clips[0]
    if clip is None:
        raise MifOpError("name the clip - the scene has no active clip and this file has %s. %s"
                         % (", ".join(c.name for c in clips) or "none (load_clip loads one)", tail))
    return clip


def _tracking_object(clip, name, tail):
    tr = clip.tracking
    if name is None:
        return next((o for o in tr.objects if o.is_camera), tr.objects.active)
    obj = tr.objects.get(name)
    if obj is None:
        raise MifOpError("clip '%s' has no tracking object '%s' (it has %s; markers given with a new object name make one). %s"
                         % (clip.name, name, ", ".join(o.name for o in tr.objects), tail))
    return obj


def _track_row(t):
    frames = sorted(m.frame for m in t.markers if not m.mute)
    row = {"name": t.name, "first": frames[0] if frames else None, "last": frames[-1] if frames else None,
           "markers": len(frames), "hasBundle": bool(t.has_bundle), "lock": bool(t.lock)}
    if t.has_bundle:
        row["averageError"] = round(t.average_error, 4)
        row["bundle"] = rnd(t.bundle, 4)
    return row


def _clip_frames(clip):
    first = clip.frame_start
    return first, first + clip.frame_duration - 1


def _tracks_summary(clip, obj):
    first, last = _clip_frames(clip)
    rows = [_track_row(t) for t in obj.tracks]
    return {"trackCount": len(rows), "reachStart": sum(1 for r in rows if r["first"] == first),
            "reachEnd": sum(1 for r in rows if r["last"] == last),
            "reachBoth": sum(1 for r in rows if r["first"] == first and r["last"] == last), "tracks": rows}


def _recon(obj):
    rec = obj.reconstruction
    out = {"isValid": bool(rec.is_valid), "averageError": round(rec.average_error, 4) if rec.is_valid else None,
           "cameraCount": len(rec.cameras)}
    if rec.is_valid:
        out["perFrameError"] = [[c.frame, round(c.average_error, 4)] for c in rec.cameras]
        # THE SOLVED PATH, in the reconstruction's own space (before any orient_clip): [frame, x, y, z] per frame - what
        # a solve is checked against when the real camera path is known (2026-10-08).
        out["path"] = [[c.frame] + rnd(c.matrix.to_translation(), 5) for c in rec.cameras]
    return out


def _clip_row(clip):
    sc = bpy.context.scene
    first, last = _clip_frames(clip)
    fps = float(getattr(clip, "fps", 0.0) or 0.0)
    scene_fps = sc.render.fps / sc.render.fps_base
    return {"name": clip.name, "filepath": clip.filepath, "source": clip.source, "size": list(clip.size),
            "frameStart": first, "frameEnd": last, "frameDuration": clip.frame_duration,
            "fps": round(fps, 4), "sceneFps": round(scene_fps, 4),
            "fpsMatchesScene": abs(fps - scene_fps) < 1e-3 if fps else None,
            "colorspace": clip.colorspace_settings.name, "active": sc.active_clip == clip,
            "objects": [o.name for o in clip.tracking.objects]}


_LOAD_CLIP_KEYS = ("filepath", "path", "name", "colorspace", "active", "setSceneRange")


def op_load_clip(params):
    """Load a movie or an image sequence as a movie clip, for tracking.

    THE GAP: nothing loaded a MovieClip, so the tracker could not be reached at all.

    params:
      filepath (str, required)  a movie, or the first frame of an image sequence (its siblings are found by number)
      name (str)                rename the clip
      colorspace (str)          the clip's input color space (the build's OCIO names: "ACEScg" on 4.2-5.2)
      active (bool)             make it the scene's active clip (default true) - the clip the solver constraints use
      setSceneRange (bool)      set the scene's frame range to the clip's (default false)

    Reply: the clip - size, frame range, source MOVIE or SEQUENCE, its fps against the scene's, color space read back.
    """
    reject_unknown(params, _LOAD_CLIP_KEYS, "load_clip")
    raw = take(params, "filepath", "path", kind=str)
    if not raw:
        raise MifOpError("'filepath' is required - a movie or the first frame of an image sequence. %s" % _ADDED)
    path = os.path.abspath(os.path.expanduser(raw))
    if not os.path.isfile(path):
        raise MifOpError("no file at '%s'. %s" % (path, _ADDED))
    name = take(params, "name", kind=str)
    if name is not None and not name.strip():
        raise MifOpError("name is blank. %s" % _ADDED)
    cs = take(params, "colorspace", kind=str)
    if cs is not None:
        valid = [e.identifier for e in
                 bpy.types.ColorManagedInputColorspaceSettings.bl_rna.properties["name"].enum_items]
        if valid and cs not in valid:
            raise MifOpError("colorspace %r is not one of this build's: %s. %s" % (cs, ", ".join(valid), _ADDED))
    active = take_bool(params, "active", default=True)
    set_range = take_bool(params, "setSceneRange", default=False)
    try:
        clip = bpy.data.movieclips.load(path)
    except RuntimeError as exc:
        raise MifOpError("Blender could not load '%s' as a clip (%s). %s" % (path, exc, _ADDED))
    if clip.frame_duration < 1 or not clip.size[0]:
        bpy.data.movieclips.remove(clip)
        raise MifOpError("'%s' loaded with no frames or no size - not a movie or image Blender can read. %s"
                         % (path, _ADDED))
    if name is not None:
        assign_id_name(clip, name, bpy.data.movieclips)   # 3.6 renames the OTHER clip on a clash (ledger B12)
    if cs is not None:
        try:
            clip.colorspace_settings.name = cs
        except TypeError as exc:
            bpy.data.movieclips.remove(clip)
            raise MifOpError("colorspace %r was refused by this clip (%s). %s" % (cs, exc, _ADDED))
    sc = bpy.context.scene
    if active:
        sc.active_clip = clip
    if set_range:
        first, last = _clip_frames(clip)
        sc.frame_start, sc.frame_end = first, last
    out = {"clip": _clip_row(clip), "sceneRange": [sc.frame_start, sc.frame_end], "requestedName": name,
           "nameWasSuffixed": name is not None and clip.name != name}
    if out["clip"]["fpsMatchesScene"] is False:
        out["fpsNote"] = ("the clip runs at %g fps and the scene at %g - set the scene's rate (set_frame_range fps) "
                          "before tracking, or the solve is timed against the wrong frames."
                          % (out["clip"]["fps"], out["clip"]["sceneFps"]))
    return out


def op_list_clip(params):
    """A movie clip and its tracking, read-only: the clip, each tracking object's tracks (frame span, markers, bundle,
    error) and reconstruction (average and per-frame error), and the camera, tracker and stabilization settings.

    params: clip (str; default the scene's active clip), limit (int, track rows per object, default 500)
    """
    reject_unknown(params, ("clip", "limit"), "list_clip")
    clip = _get_clip(take(params, "clip", kind=str), "")
    limit = take_int(params, "limit", default=500)
    if limit < 1:
        raise MifOpError("limit must be 1 or more.")
    tr = clip.tracking
    objs = []
    for o in tr.objects:
        s = _tracks_summary(clip, o)
        if len(s["tracks"]) > limit:
            s["tracks"] = s["tracks"][:limit]
            s["tracksTruncated"] = True
        objs.append(dict({"name": o.name, "isCamera": bool(o.is_camera), "keyframes": [o.keyframe_a, o.keyframe_b],
                          "reconstruction": _recon(o)}, **s))
    return {"clip": _clip_row(clip), "objects": objs,
            "camera": rna_properties(tr.camera), "settings": rna_properties(tr.settings),
            "stabilization": rna_properties(tr.stabilization, skip=frozenset({"tracks", "rotation_tracks"})),
            "stabilizationResidual": _stab_residual(clip)}


_SET_CLIP_KEYS = ("clip", "camera", "settings", "stabilization", "tracks", "copyToSceneCamera", "proxy",
                  "stabilizationTracks")
_CLIP_PROXY_KEYS = {"sizes", "quality", "undistorted", "build"}


def _check_group(holder, values, label, deny):
    if not isinstance(values, dict):
        raise MifOpError("%s takes {property: value}, got %r. %s" % (label, values, _CHANGED))
    for key, value in values.items():
        rna = holder.bl_rna.properties.get(key)
        if rna is None or rna.is_readonly or key in deny or key.startswith("bl_"):
            names = sorted(p.identifier for p in holder.bl_rna.properties
                           if not p.is_readonly and p.identifier not in deny and not p.identifier.startswith("bl_"))
            raise MifOpError("%s has no writable property '%s'. It has: %s. %s"
                             % (label, key, ", ".join(names), _CHANGED))
        if rna.type == "ENUM" and not rna.is_enum_flag:
            valid = [e.identifier for e in rna.enum_items]
            if value not in valid:
                raise MifOpError("%s.%s %r is not one of this build's: %s (identifiers are case-sensitive). %s"
                                 % (label, key, value, ", ".join(valid), _CHANGED))


def op_set_clip(params):
    """Set a movie clip's lens, tracker defaults, 2D stabilization and per-track settings - by their RNA names.

    THE GAP: the solver needs the real lens (sensor width, focal length, principal point, distortion model and its
    coefficients) and the tracker its pattern, search, motion model and matching settings; none was reachable.

    params:
      clip (str)                default the scene's active clip
      camera (dict)             tracking.camera: sensor_width, focal_length, units, pixel_aspect, principal_point,
                                distortion_model POLYNOMIAL / DIVISION / NUKE / BROWN, k1-k3, division_k1/2, nuke_k1/2
                                (nuke_p1/2 on 5.2), brown_k1-k4, brown_p1/2
      settings (dict)           tracking.settings: default_pattern_size, default_search_size, default_motion_model
                                ('Loc', 'LocRot', 'LocScale', 'LocRotScale', 'Affine', 'Perspective' - mixed case, as
                                Blender names them), default_pattern_match KEYFRAME / PREV_FRAME,
                                use_default_normalization, default_correlation_min, default_margin, ...
      stabilization (dict)      tracking.stabilization: use_2d_stabilization, anchor_frame, use_autoscale, ...
      tracks (dict)             {track name: {motion_model, pattern_match, correlation_min, lock, weight, ...}}
      copyToSceneCamera (bool)  copy focal length and sensor width onto the scene camera
      proxy (dict)              {sizes [25|50|75|100], undistorted [sizes], quality 1-100, build (default true)} - the
                                clip's proxies turned on and built (rebuild_proxy); the reply lists the proxy files
      stabilizationTracks (dict) {location: [names], rotation: [names]} - the 2D stabilization's track lists, each
                                replaced whole (location needs one or more, rotation and scale two or more)

    Every key and enum value is checked against this build before anything is written, and a write Blender refuses
    puts every earlier write in the call back. Reply: each group as written, read back.
    """
    reject_unknown(params, _SET_CLIP_KEYS, "set_clip")
    clip = _get_clip(take(params, "clip", kind=str), _CHANGED)
    tr = clip.tracking
    groups = []
    for key, holder, deny in (("camera", tr.camera, frozenset()), ("settings", tr.settings, frozenset()),
                              ("stabilization", tr.stabilization, frozenset({"active_track_index",
                                                                             "active_rotation_track_index"}))):
        vals = params.get(key)
        if vals is not None:
            _check_group(holder, vals, key, deny)
            groups.append((key, None, holder, vals))
    tracks = params.get("tracks")
    if tracks is not None:
        if not isinstance(tracks, dict) or not tracks:
            raise MifOpError("tracks takes {track name: {property: value}}. %s" % _CHANGED)
        for tname, vals in tracks.items():
            t = tr.tracks.get(tname)
            if t is None:
                raise MifOpError("clip '%s' has no track '%s'. %s" % (clip.name, tname, _CHANGED))
            _check_group(t, vals, "track '%s'" % tname, frozenset({"name", "select", "select_anchor",
                                                                  "select_pattern", "select_search"}))
            groups.append(("tracks", tname, t, vals))
    proxy = _check_clip_proxy(params.get("proxy"))
    stab = _check_stab_tracks(clip, params.get("stabilizationTracks"))
    copy_cam = take_bool(params, "copyToSceneCamera", default=False)
    cam = bpy.context.scene.camera
    if copy_cam and (cam is None or cam.type != "CAMERA"):
        raise MifOpError("copyToSceneCamera: the scene has no camera. %s" % _CHANGED)
    if not groups and not copy_cam and proxy is None and stab is None:
        raise MifOpError("nothing to set - pass camera, settings, stabilization, tracks, stabilizationTracks, proxy "
                         "or copyToSceneCamera. %s" % _CHANGED)
    undo = []
    out = {"clip": clip.name}
    try:
        for key, tname, holder, vals in groups:
            got = {}
            for k, v in vals.items():
                undo.append((holder, k, getattr(holder, k)))
                got[k] = write_rna_property(holder, k, v, "", "%s of clip '%s'" % (key, clip.name), key, frozenset())
            if tname is None:
                out[key] = got
            else:
                out.setdefault("tracks", {})[tname] = got
    except MifOpError as exc:
        for holder, k, old in reversed(undo):
            try:
                setattr(holder, k, old)
            except (TypeError, ValueError, AttributeError):
                pass
        raise MifOpError("%s Every earlier write in this call was put back. %s" % (str(exc).strip(), _CHANGED))
    if stab is not None:
        out["stabilizationTracks"] = _write_stab_tracks(clip, stab)
    if stab is not None or params.get("stabilization") is not None or params.get("tracks") is not None:
        # how much the stabilized tracks still move, with what was just written (see _stab_residual)
        out["stabilizationResidual"] = _stab_residual(clip)
    if proxy is not None:
        out["proxy"] = _write_clip_proxy(clip, proxy)
    if copy_cam:
        cam.data.sensor_width = tr.camera.sensor_width
        cam.data.lens = tr.camera.focal_length
        out["sceneCamera"] = {"name": cam.name, "lens": round(cam.data.lens, 4),
                              "sensorWidth": round(cam.data.sensor_width, 4)}
    return out


def _check_clip_proxy(raw):
    if raw is None:
        return None
    if not isinstance(raw, dict) or not raw or set(raw) - _CLIP_PROXY_KEYS:
        raise MifOpError("proxy takes {%s}, got %r. %s" % (", ".join(sorted(_CLIP_PROXY_KEYS)), raw, _CHANGED))
    out = {}
    for key in ("sizes", "undistorted"):
        val = raw.get(key, [25] if key == "sizes" else [])
        if not isinstance(val, list) or any(s not in _PROXY_SIZES for s in val):
            raise MifOpError("proxy.%s is a list of 25, 50, 75 and 100, got %r. %s" % (key, val, _CHANGED))
        out[key] = sorted(set(val))
    if raw.get("quality") is not None:
        q = finite_int(raw["quality"], "proxy.quality")
        if not 1 <= q <= 100:
            raise MifOpError("proxy.quality is 1-100, got %d. %s" % (q, _CHANGED))
        out["quality"] = q
    out["build"] = bool(raw.get("build", True))
    return out


def _clip_proxy_row(clip):
    px = clip.proxy
    folder = bpy.path.abspath(px.directory) if clip.use_proxy_custom_directory and px.directory else os.path.join(
        os.path.dirname(bpy.path.abspath(clip.filepath)), "BL_proxy", os.path.basename(clip.filepath))
    files = []
    if os.path.isdir(folder):
        for root, _dirs, names in os.walk(folder):
            files += [os.path.relpath(os.path.join(root, n), folder) for n in names]
    return {"sizes": [n for n in _PROXY_SIZES if getattr(px, "build_%d" % n)],
            "undistorted": [n for n in _PROXY_SIZES if getattr(px, "build_undistorted_%d" % n)],
            "quality": int(px.quality), "directory": os.path.normpath(folder), "files": sorted(files)[:100],
            "fileCount": len(files)}


def _write_clip_proxy(clip, cfg):
    clip.use_proxy = True
    px = clip.proxy
    for n in _PROXY_SIZES:
        setattr(px, "build_%d" % n, n in cfg["sizes"])
        setattr(px, "build_undistorted_%d" % n, n in cfg["undistorted"])
    if cfg.get("quality") is not None:
        px.quality = cfg["quality"]
    out = {}
    if cfg["build"]:
        with _clip_editor(clip):
            res = bpy.ops.clip.rebuild_proxy()
        out["rebuilt"] = "FINISHED" in res
        # A JOB, NOT A CALL: a clip's rebuild_proxy returns FINISHED and its worker writes the frames after (measured
        # 2026-10-08 on 5.2.1: none on disk at the return, all 30 a moment later). Waited for until every size has a
        # file per frame (a movie: one file a size), or 120 s.
        want = len(cfg["sizes"] + cfg["undistorted"]) * (clip.frame_duration if clip.source == "SEQUENCE" else 1)
        t0 = time.time()
        while time.time() - t0 < 120.0 and _clip_proxy_row(clip)["fileCount"] < want:
            time.sleep(0.1)
        out["waitedSeconds"] = round(time.time() - t0, 2)
        out["expectedFiles"] = want
    out.update(_clip_proxy_row(clip))
    if cfg["build"] and bpy.app.background:
        out["jobNote"] = ("the build ran as a window-manager job, which a background Blender never retires: Blender "
                          "cancels every later tracking operator on this clip in this session (track_clip and "
                          "edit_clip_tracks refuse, naming it). Track before building proxies.")
    if cfg["build"] and out["fileCount"] < out["expectedFiles"]:
        out["note"] = "rebuild_proxy left %d of %d files in %s after %g s." % (
            out["fileCount"], out["expectedFiles"], out["directory"], out["waitedSeconds"])
    return out


def _check_stab_tracks(clip, raw):
    if raw is None:
        return None
    if not isinstance(raw, dict) or not raw or set(raw) - {"location", "rotation"}:
        raise MifOpError("stabilizationTracks takes {location: [names], rotation: [names]}, got %r. %s"
                         % (raw, _CHANGED))
    obj = next((o for o in clip.tracking.objects if o.is_camera), None)
    out = {}
    for key, least in (("location", 1), ("rotation", 2)):
        if raw.get(key) is None:
            continue
        names = raw[key]
        if not isinstance(names, list) or (names and len(names) < least):
            raise MifOpError("stabilizationTracks.%s takes a list of %d or more track names (or [] to clear). %s"
                             % (key, least, _CHANGED))
        out[key] = _named_tracks(obj, names, _CHANGED) if names else []
    return out


def _stab_residual(clip):
    """How much the stabilized tracks still move after 2D stabilization, in pixels - or None when it is off or has no
    location tracks.

    NO PYTHON API RETURNS THE STABILIZER'S PER-FRAME TRANSFORM (MovieTrackingStabilization, MovieClip and the
    Stabilize node have none, 3.6-5.2), so this applies Blender's own model to the markers: the frame is shifted by the
    weight_stab-weighted mean of the location tracks' displacement from the anchor frame (times influence_location),
    then turned by the weighted mean change of the rotation tracks' angle about that weighted centre (times
    influence_rotation) and scaled by their weighted mean length ratio (geometric, times influence_scale). MEASURED
    against the compositor's Stabilize node on 2026-10-11 (two dots at different speeds, every marker placed exactly):
    location with equal and unequal weights matched the rendered centroids to the pixel; with rotation on, within
    0.3 px (the resampled dot's centroid). use_autoscale's zoom is not in it - it scales the whole frame about its
    centre and moves every track alike, so the relative motion reported here is before it.

    A track counts at a frame where it has an enabled marker there and at the anchor. Reply: the largest residual and
    the frame, each stabilized track's largest residual and frame, and the frames measured."""
    import math
    st = clip.tracking.stabilization
    loc = list(st.tracks)
    if not st.use_2d_stabilization or not loc:
        return None
    rot = list(st.rotation_tracks) if st.use_stabilize_rotation or st.use_stabilize_scale else []
    w, h = clip.size
    anchor = int(st.anchor_frame)

    def at(track, f):
        m = track.markers.find_frame(f, exact=True)
        if m is None or m.mute:
            return None
        return ((m.co[0] + track.offset[0]) * w, (m.co[1] + track.offset[1]) * h)

    def centre(tracks, f):
        sx = sy = sw = 0.0
        for t in tracks:
            p, pa = at(t, f), at(t, anchor)
            if p is None or pa is None:
                continue
            sx += t.weight_stab * p[0]
            sy += t.weight_stab * p[1]
            sw += t.weight_stab
        return (sx / sw, sy / sw) if sw > 0 else None

    ca = centre(loc, anchor)
    if ca is None:
        return None
    start = int(clip.frame_start)
    frames = range(start, start + int(clip.frame_duration))
    tracked = {t.name: t for t in loc + rot}
    worst, per = (0.0, None), {}
    measured = 0
    for f in frames:
        cf = centre(loc, f)
        if cf is None:
            continue
        measured += 1
        dx, dy = (cf[0] - ca[0]) * st.influence_location, (cf[1] - ca[1]) * st.influence_location
        ang = lsc = sw = 0.0
        for t in rot:
            p, pa = at(t, f), at(t, anchor)
            if p is None or pa is None or t.weight_stab <= 0:
                continue
            vf, va = (p[0] - cf[0], p[1] - cf[1]), (pa[0] - ca[0], pa[1] - ca[1])
            lf, la = math.hypot(*vf), math.hypot(*va)
            if lf < 1e-9 or la < 1e-9:
                continue
            d = math.atan2(vf[1], vf[0]) - math.atan2(va[1], va[0])
            d = (d + math.pi) % (2 * math.pi) - math.pi
            ang += t.weight_stab * d
            lsc += t.weight_stab * math.log(lf / la)
            sw += t.weight_stab
        theta = -(ang / sw) * st.influence_rotation if sw and st.use_stabilize_rotation else 0.0
        scale = math.exp(-(lsc / sw) * st.influence_scale) if sw and st.use_stabilize_scale else 1.0
        cs, sn = math.cos(theta), math.sin(theta)
        # the pivot the frame turns about, once shifted: the weighted centre where the shift puts it
        px, py = cf[0] - dx, cf[1] - dy
        for name, t in tracked.items():
            p, pa = at(t, f), at(t, anchor)
            if p is None or pa is None:
                continue
            qx, qy = p[0] - dx - px, p[1] - dy - py
            qx, qy = scale * (cs * qx - sn * qy) + px, scale * (sn * qx + cs * qy) + py
            r = math.hypot(qx - pa[0], qy - pa[1])
            if r > per.get(name, (-1.0, None))[0]:
                per[name] = (r, f)
            if r > worst[0]:
                worst = (r, f)
    if not measured:
        return None
    return {"maxPixels": round(worst[0], 3), "frame": worst[1], "framesMeasured": measured,
            "tracks": {n: {"maxPixels": round(v[0], 3), "frame": v[1]} for n, v in sorted(per.items())},
            "autoscaleIncluded": False if st.use_autoscale else None}


def _write_stab_tracks(clip, want):
    st = clip.tracking.stabilization
    obj = next((o for o in clip.tracking.objects if o.is_camera), None)
    out = {}
    with _clip_editor(clip):
        for key, coll_name, idx_name, add, remove in (
                ("location", "tracks", "active_track_index", bpy.ops.clip.stabilize_2d_add,
                 bpy.ops.clip.stabilize_2d_remove),
                ("rotation", "rotation_tracks", "active_rotation_track_index", bpy.ops.clip.stabilize_2d_rotation_add,
                 bpy.ops.clip.stabilize_2d_rotation_remove)):
            if key not in want:
                continue
            coll = getattr(st, coll_name)
            while len(coll):
                setattr(st, idx_name, 0)
                n = len(coll)
                remove()
                if len(coll) == n:
                    raise MifOpError("Blender's %s would not take '%s' off the %s list."
                                     % (remove.idname_py(), coll[0].name, key))
            if want[key]:
                _select_only(obj.tracks, want[key])
                add()
                _select_only(obj.tracks, [])
            out[key] = sorted(t.name for t in coll)
            if out[key] != sorted(t.name for t in want[key]):
                raise MifOpError("the %s list reads %s after the write, not %s." % (key, out[key],
                                                                                  sorted(t.name for t in want[key])))
    return out


class _clip_editor(editor_override):
    """editor_override on a Clip Editor showing `clip` at the scene frame - restored after."""

    def __init__(self, clip):
        super().__init__("CLIP_EDITOR")
        self.clip = clip

    def __enter__(self):
        ctx = super().__enter__()
        space = ctx["space"]
        self.keep(space, "clip")
        space.clip = self.clip
        self.keep(self.scene, "frame_current")
        # THE FULL FRAME, NOT A PROXY. With the clip's proxies on, track_markers in a borrowed editor tracked nothing -
        # every marker stayed on its first frame, FINISHED (measured 2026-10-08 on 5.2.1) - so the editor's render size
        # is the original footage while an op works, and put back after.
        cu = space.clip_user
        for attr, val in (("proxy_render_size", "FULL"), ("use_render_undistorted", False)):
            if attr in cu.bl_rna.properties:
                self.keep(cu, attr)
                setattr(cu, attr, val)
        self.space = space
        return self

    def frame(self, f):
        """Go to frame f in the scene AND in the borrowed editor's clip user. The clip operators (track_markers,
        detect_features, clear_track_path, ...) read space.clip_user.frame_current, which only a redraw copies from the
        scene - and a background Blender never redraws (measured 2026-10-08 on 5.2.1: an object's markers placed at
        frame 1 were tracked from wherever the clip user had been left, after another call moved it)."""
        self.scene.frame_set(f)
        self.space.clip_user.frame_current = f


_JOB_BLOCK = ("a clip job in this session never ended - in a background Blender a finished window-manager job is never "
              "retired, and after a clip's rebuild_proxy (set_clip proxy) Blender cancels every later tracking operator "
              "(measured 2026-10-08 on 5.2.1: track_markers CANCELLED, markers left on their first frame). Track before "
              "building proxies, or restart this Blender")


def _clip_op(op, **kw):
    """Run a clip operator and refuse on CANCELLED rather than report a step that did nothing."""
    res = op(**kw)
    if "CANCELLED" in res:
        raise MifOpError("Blender's %s was cancelled: %s. Steps before it in this call were applied."
                         % (op.idname_py(), _JOB_BLOCK if getattr(bpy.app, "background", False)
                            else "the clip editor refused it"))
    return res


def _select_only(tracks, chosen):
    keep = set(chosen)
    for t in tracks:
        t.select = t in keep


def _frame_list(raw, clip, key, tail):
    first, last = _clip_frames(clip)
    if raw is None:
        return None
    if not isinstance(raw, list) or not raw:
        raise MifOpError("%s takes a list of frames. %s" % (key, tail))
    out = []
    for f in raw:
        f = finite_int(f, key)
        if not first <= f <= last:
            raise MifOpError("%s: frame %d is outside the clip's %d-%d. %s" % (key, f, first, last, tail))
        out.append(f)
    return sorted(set(out))


_TRACK_CLIP_KEYS = ("clip", "object", "markers", "detect", "track")
_DIRECTIONS = ("FORWARDS", "BACKWARDS", "BOTH")


def op_track_clip(params):
    """Place markers, detect features and track them through a movie clip - Blender's tracker, headless.

    params:
      clip (str), object (str)   the clip (default the scene's active clip) and tracking object (default the camera's;
                                 a new name with markers makes the object - an object track for an object solve)
      markers (list)             hand-placed markers [{track, frame, co: [x, y]}] in 0-1 frame coordinates; a new
                                 track name makes a track (tracks.new), an existing one gets a keyframed marker
      detect (dict)              {frames: [..] (default the first), threshold, minDistance, margin,
                                 placement FRAME | INSIDE_GPENCIL | OUTSIDE_GPENCIL} - clip.detect_features per frame
      track (dict)               {direction FORWARDS | BACKWARDS | BOTH (default BOTH), tracks: [names] (default
                                 every unlocked track)} - each track runs from its last marker forwards and from its
                                 first marker backwards, so features found mid-clip cover both ends

    Steps run in that order. Reply: detectedAt {frame: new tracks}, each track's first and last frame and marker count,
    and how many reach the clip's start, its end and both.
    """
    reject_unknown(params, _TRACK_CLIP_KEYS, "track_clip")
    clip = _get_clip(take(params, "clip", kind=str), _CHANGED)
    oname = take(params, "object", kind=str)
    new_obj = oname is not None and clip.tracking.objects.get(oname) is None
    if new_obj and (not oname.strip() or params.get("markers") is None):
        raise MifOpError("clip '%s' has no tracking object %r - an object track starts from hand-placed markers, so "
                         "pass markers with it to make one. %s" % (clip.name, oname, _CHANGED))
    obj = None if new_obj else _tracking_object(clip, oname, _CHANGED)
    first, last = _clip_frames(clip)
    markers = params.get("markers")
    planned = []
    if markers is not None:
        if not isinstance(markers, list) or not markers:
            raise MifOpError("markers takes [{track, frame, co: [x, y]}]. %s" % _CHANGED)
        for m in markers:
            if not isinstance(m, dict) or set(m) - {"track", "frame", "co"} or not isinstance(m.get("track"), str) \
                    or not m.get("track"):
                raise MifOpError("each marker is {track, frame, co: [x, y]}, got %r. %s" % (m, _CHANGED))
            f = finite_int(m.get("frame", first), "markers.frame")
            if not first <= f <= last:
                raise MifOpError("marker frame %d is outside the clip's %d-%d. %s" % (f, first, last, _CHANGED))
            co = m.get("co")
            if not isinstance(co, list) or len(co) != 2:
                raise MifOpError("marker co takes [x, y] in 0-1 frame coordinates, got %r. %s" % (co, _CHANGED))
            planned.append((m["track"], f, finite_floats(co, "markers.co")))
    detect = params.get("detect")
    det = None
    if detect is not None:
        if not isinstance(detect, dict) or set(detect) - {"frames", "threshold", "minDistance", "margin", "placement"}:
            raise MifOpError("detect takes {frames, threshold, minDistance, margin, placement}, got %r. %s"
                             % (detect, _CHANGED))
        frames = _frame_list(detect.get("frames"), clip, "detect.frames", _CHANGED) or [first]
        kw = {}
        if detect.get("threshold") is not None:
            kw["threshold"] = take_float(detect, "threshold")
        if detect.get("minDistance") is not None:
            kw["min_distance"] = take_int(detect, "minDistance")
        if detect.get("margin") is not None:
            kw["margin"] = take_int(detect, "margin")
        if detect.get("placement") is not None:
            if detect["placement"] not in ("FRAME", "INSIDE_GPENCIL", "OUTSIDE_GPENCIL"):
                raise MifOpError("detect.placement is FRAME, INSIDE_GPENCIL or OUTSIDE_GPENCIL. %s" % _CHANGED)
            kw["placement"] = detect["placement"]
        det = (frames, kw)
    track = params.get("track")
    trk = None
    if track is not None:
        if not isinstance(track, dict) or set(track) - {"direction", "tracks"}:
            raise MifOpError("track takes {direction, tracks}, got %r. %s" % (track, _CHANGED))
        direction = str(track.get("direction", "BOTH")).upper()
        if direction not in _DIRECTIONS:
            raise MifOpError("track.direction is FORWARDS, BACKWARDS or BOTH. %s" % _CHANGED)
        names = track.get("tracks")
        if names is not None:
            known = {t.name for t in (obj.tracks if obj is not None else [])} | {p[0] for p in planned}
            if not isinstance(names, list) or not names or any(n not in known for n in names):
                raise MifOpError("track.tracks names tracks of '%s'; it has %s. %s"
                                 % (obj.name, ", ".join(sorted(known))[:400] or "(none)", _CHANGED))
        trk = (direction, names)
    if not planned and det is None and trk is None:
        raise MifOpError("nothing to do - pass markers, detect or track. %s" % _CHANGED)
    if det is not None and (obj is None or not obj.is_camera):
        raise MifOpError("detect places features on the camera's tracks; for tracking object '%s' place markers by "
                         "hand. %s" % (oname if obj is None else obj.name, _CHANGED))

    tr = clip.tracking
    if obj is None:
        obj = tr.objects.new(oname)
    if tr.objects.active != obj:
        tr.objects.active = obj
    made = []
    for tname, f, co in planned:
        t = obj.tracks.get(tname)
        if t is None:
            t = obj.tracks.new(name=tname, frame=f)
            made.append(t.name)
            mk = t.markers.find_frame(f)
            mk.co = co
        else:
            t.markers.insert_frame(f, co=co)
    out = {"clip": clip.name, "object": obj.name, "madeObject": new_obj, "madeTracks": made}
    with _clip_editor(clip) as ed_:
        if det is not None:
            frames, kw = det
            detected = {}
            for f in frames:
                ed_.frame(f)
                before = set(t.name for t in obj.tracks)
                _select_only(obj.tracks, [])
                _clip_op(bpy.ops.clip.detect_features, **kw)
                detected[str(f)] = len([t for t in obj.tracks if t.name not in before])
            out["detectedAt"] = detected
        if trk is not None:
            direction, names = trk
            chosen = [t for t in obj.tracks if (names is None and not t.lock) or (names and t.name in names)]
            for backwards in ((False,) if direction == "FORWARDS" else (True,) if direction == "BACKWARDS"
                              else (False, True)):
                groups = {}
                for t in chosen:
                    fr = [m.frame for m in t.markers if not m.mute]
                    if fr:
                        groups.setdefault(min(fr) if backwards else max(fr), []).append(t)
                for f in sorted(groups):
                    if (f == first and backwards) or (f == last and not backwards):
                        continue
                    ed_.frame(f)
                    _select_only(obj.tracks, groups[f])
                    _clip_op(bpy.ops.clip.track_markers, backwards=backwards, sequence=True)
            _select_only(obj.tracks, [])
            out["tracked"] = len(chosen)
    out.update(_tracks_summary(clip, obj))
    out["clipRange"] = [first, last]
    return out


_SOLVE_KEYS = ("clip", "object", "keyframes", "tripod", "refine", "clean", "rounds")
_REFINE = {"FOCAL_LENGTH": "refine_intrinsics_focal_length", "PRINCIPAL_POINT": "refine_intrinsics_principal_point",
           "RADIAL_DISTORTION": "refine_intrinsics_radial_distortion",
           "TANGENTIAL_DISTORTION": "refine_intrinsics_tangential_distortion"}


def _delete_tracks(obj, doomed):
    if doomed:
        _select_only(obj.tracks, doomed)
        bpy.ops.clip.delete_track(confirm=False)


def op_solve_clip(params):
    """Solve a movie clip's camera (or an object's) motion, clean the tracks by error and solve again.

    params:
      clip (str), object (str)  the clip and tracking object (default the camera's; track_clip makes an object)
      keyframes ([a, b] | "AUTO")  the two keyframes (tracking_object.keyframe_a / _b), or Blender's keyframe selection
      tripod (bool)             a nodal pan - the tripod solver
      refine (list)             FOCAL_LENGTH, PRINCIPAL_POINT, RADIAL_DISTORTION, TANGENTIAL_DISTORTION (none default)
      clean (dict)              {frames: N, error: px} after the first solve, delete tracks shorter than N frames or
                                above the error, and solve again (Blender's clean_tracks, done natively so a short-track
                                rule is never applied unasked)
      rounds (dict)             {dropFraction (0.1), targetError (px), minTracks (8), maxRounds (5)} - delete the worst
                                fraction by reprojection error and re-solve until the target, the floor or the cap

    Reply: rounds [{averageError, tracks, deleted}], the final reconstruction (average error and the per-frame error,
    the curve that should be flat) and every track's error.
    """
    reject_unknown(params, _SOLVE_KEYS, "solve_clip")
    clip = _get_clip(take(params, "clip", kind=str), _CHANGED)
    tr = clip.tracking
    obj = _tracking_object(clip, take(params, "object", kind=str), _CHANGED)
    first, last = _clip_frames(clip)
    kf = params.get("keyframes")
    if kf is not None and kf != "AUTO":
        if not isinstance(kf, list) or len(kf) != 2:
            raise MifOpError("keyframes takes [a, b] or \"AUTO\". %s" % _CHANGED)
        kf = [finite_int(k, "keyframes") for k in kf]
        if not (first <= kf[0] < kf[1] <= last):
            raise MifOpError("keyframes %s must be two rising frames inside the clip's %d-%d. %s"
                             % (kf, first, last, _CHANGED))
    refine = params.get("refine")
    if refine is not None and (not isinstance(refine, list) or any(r not in _REFINE for r in refine)):
        raise MifOpError("refine takes a list of %s. %s" % (", ".join(_REFINE), _CHANGED))
    clean = params.get("clean")
    if clean is not None and (not isinstance(clean, dict) or not clean or set(clean) - {"frames", "error"}):
        raise MifOpError("clean takes {frames, error}. %s" % _CHANGED)
    c_frames = finite_int(clean["frames"], "clean.frames") if clean and clean.get("frames") is not None else None
    c_error = take_float(clean, "error") if clean and clean.get("error") is not None else None
    rounds = params.get("rounds")
    if rounds is not None and (not isinstance(rounds, dict)
                               or set(rounds) - {"dropFraction", "targetError", "minTracks", "maxRounds"}):
        raise MifOpError("rounds takes {dropFraction, targetError, minTracks, maxRounds}. %s" % _CHANGED)
    rounds = rounds or {}
    drop = take_float(rounds, "dropFraction", default=0.1)
    target = take_float(rounds, "targetError")
    floor = take_int(rounds, "minTracks", default=8)
    cap = take_int(rounds, "maxRounds", default=5 if params.get("rounds") is not None else 0)
    if not 0 < drop < 1 or floor < 8 or cap < 0:
        raise MifOpError("rounds: dropFraction between 0 and 1, minTracks 8 or more (the solver needs 8), maxRounds "
                         "0 or more. %s" % _CHANGED)
    tripod = params.get("tripod")
    tripod = None if tripod is None else take_bool(params, "tripod")
    if len([t for t in obj.tracks if sum(1 for m in t.markers if not m.mute) > 1]) < 8:
        raise MifOpError("'%s' has fewer than 8 tracked tracks - the solver needs at least 8 (track_clip). %s"
                         % (obj.name, _CHANGED))

    if isinstance(kf, list):
        # THE SOLVER'S OWN RULE, checked before anything is written: 8 tracks with a marker on both keyframes. It
        # raised "At least 8 common tracks on both keyframes are needed" out of the operator after the keyframes and
        # settings were already set (measured 2026-10-08 on 5.2.1, an object track lost before frame 25).
        def _on(t, f):
            m = t.markers.find_frame(f)
            return m is not None and not m.mute
        common = [t.name for t in obj.tracks if _on(t, kf[0]) and _on(t, kf[1])]
        if len(common) < 8:
            raise MifOpError("only %d track(s) of '%s' have a marker on both keyframes %d and %d, and the solver needs "
                             "8 - pick keyframes the tracks span, or track further. %s"
                             % (len(common), obj.name, kf[0], kf[1], _CHANGED))
    st = tr.settings
    if tr.objects.active != obj:
        tr.objects.active = obj
    if kf == "AUTO":
        st.use_keyframe_selection = True
    elif kf is not None:
        st.use_keyframe_selection = False
        obj.keyframe_a, obj.keyframe_b = kf
    if tripod is not None:
        st.use_tripod_solver = tripod
    if refine is not None:
        for name, attr in _REFINE.items():
            if attr in st.bl_rna.properties:
                setattr(st, attr, name in refine)
    rows = []
    with _clip_editor(clip):
        try:
            bpy.ops.clip.solve_camera()
        except RuntimeError as exc:
            raise MifOpError("Blender's solver refused: %s. The keyframes and solver settings asked for were written."
                             % str(exc).replace("Error:", "").strip()) from None
        rec = obj.reconstruction
        rows.append({"averageError": round(rec.average_error, 4) if rec.is_valid else None,
                     "tracks": len(obj.tracks), "deleted": 0})
        if c_frames is not None or c_error is not None:
            doomed = [t for t in obj.tracks
                      if (c_frames is not None and sum(1 for m in t.markers if not m.mute) < c_frames)
                      or (c_error is not None and t.has_bundle and t.average_error > c_error)]
            if len(obj.tracks) - len(doomed) < 8:
                doomed = sorted(doomed, key=lambda t: -t.average_error)[:max(0, len(obj.tracks) - 8)]
            _delete_tracks(obj, doomed)
            bpy.ops.clip.solve_camera()
            rows.append({"averageError": round(rec.average_error, 4) if rec.is_valid else None,
                         "tracks": len(obj.tracks), "deleted": len(doomed), "step": "clean"})
        for _ in range(cap):
            if not rec.is_valid or (target is not None and rec.average_error <= target):
                break
            room = len(obj.tracks) - floor
            if room <= 0:
                break
            k = min(room, max(1, int(round(len(obj.tracks) * drop))))
            worst = sorted([t for t in obj.tracks if t.has_bundle], key=lambda t: -t.average_error)[:k]
            unbundled = [t for t in obj.tracks if not t.has_bundle][:max(0, k - len(worst))]
            _delete_tracks(obj, worst + unbundled)
            bpy.ops.clip.solve_camera()
            rows.append({"averageError": round(rec.average_error, 4) if rec.is_valid else None,
                         "tracks": len(obj.tracks), "deleted": len(worst) + len(unbundled), "step": "round"})
        _select_only(obj.tracks, [])
    out = {"clip": clip.name, "object": obj.name, "keyframes": [obj.keyframe_a, obj.keyframe_b],
           "keyframeSelection": bool(st.use_keyframe_selection), "tripod": bool(st.use_tripod_solver),
           "rounds": rows, "reconstruction": _recon(obj)}
    out.update(_tracks_summary(clip, obj))
    if target is not None:
        out["targetReached"] = bool(rec.is_valid and rec.average_error <= target)
    if not rec.is_valid:
        out["warning"] = "the solve did not reconstruct - more tracks, longer tracks or other keyframes are needed."
    return out


_ORIENT_KEYS = ("clip", "object", "floor", "wall", "origin", "useMedian", "xAxis", "yAxis", "scale")


def op_orient_clip(params):
    """Orient a solved movie clip's scene: a floor or wall plane from three tracks, the origin at a track, an axis
    through a track, and a real distance between two tracks.

    params:
      clip, object       as solve_clip
      floor / wall       three track names (clip.set_plane FLOOR / WALL)
      origin (list)      one or more track names (set_origin; useMedian true takes their median, else the mean)
      xAxis / yAxis      a track name the axis points at (set_axis)
      scale (dict)       {tracks: [a, b], distance: meters} - set_scale for the camera, set_solution_scale for an object

    The operators move the SCENE CAMERA (or the object's parent): the scene needs a camera. Every track must be solved
    (have a bundle) before anything moves. Reply: the camera's location, rotation and scale after, and the floor
    tracks' residual height (their bundles' spread from the plane they set).
    """
    reject_unknown(params, _ORIENT_KEYS, "orient_clip")
    clip = _get_clip(take(params, "clip", kind=str), _CHANGED)
    obj = _tracking_object(clip, take(params, "object", kind=str), _CHANGED)
    sc = bpy.context.scene
    if sc.camera is None:
        raise MifOpError("the scene has no camera, and orienting a solve moves it. %s" % _CHANGED)
    if not obj.reconstruction.is_valid:
        raise MifOpError("'%s' of clip '%s' is not solved - solve_clip first. %s" % (obj.name, clip.name, _CHANGED))
    if params.get("floor") is not None and params.get("wall") is not None:
        raise MifOpError("floor or wall, not both. %s" % _CHANGED)
    if params.get("xAxis") is not None and params.get("yAxis") is not None:
        raise MifOpError("xAxis or yAxis, not both. %s" % _CHANGED)

    def _named(key, val, n=None):
        if val is None:
            return None
        names = [val] if isinstance(val, str) else val
        if not isinstance(names, list) or not names or (n is not None and len(names) != n):
            raise MifOpError("%s takes %s track name%s. %s" % (key, n or "one or more", "" if n == 1 else "s",
                                                              _CHANGED))
        out = []
        for nm in names:
            t = obj.tracks.get(nm) if isinstance(nm, str) else None
            if t is None or not t.has_bundle:
                raise MifOpError("%s: '%s' is %s. %s" % (key, nm, "not solved (no bundle)" if t is not None
                                                         else "not a track of '%s'" % obj.name, _CHANGED))
            out.append(t)
        return out

    plane = _named("floor", params.get("floor"), 3) or _named("wall", params.get("wall"), 3)
    origin = _named("origin", params.get("origin"))
    axis = _named("xAxis", params.get("xAxis"), 1) or _named("yAxis", params.get("yAxis"), 1)
    scale = params.get("scale")
    s_tracks = s_dist = None
    if scale is not None:
        if not isinstance(scale, dict) or set(scale) != {"tracks", "distance"}:
            raise MifOpError("scale takes {tracks: [a, b], distance}. %s" % _CHANGED)
        s_tracks = _named("scale.tracks", scale["tracks"], 2)
        s_dist = take_float(scale, "distance")
        if s_dist <= 0:
            raise MifOpError("scale.distance must be above 0. %s" % _CHANGED)
    # DEGENERATE INPUT IS REFUSED BEFORE ANYTHING MOVES. Measured 2026-10-06 on 5.2.2: features detected at several
    # frames find the same dot again, and three such tracks given as a floor - or two as a scale - left the camera at
    # scale 0 with ok from every operator.
    if plane:
        a, b, c = (Vector(t.bundle) for t in plane)
        span = max((a - b).length, (b - c).length, (a - c).length)
        if span < 1e-6 or (b - a).cross(c - a).length < 1e-3 * span * span:
            raise MifOpError("the three plane tracks (%s) are one point or one line in the solve - pick three apart. %s"
                             % (", ".join(t.name for t in plane), _CHANGED))
    if s_tracks and (Vector(s_tracks[0].bundle) - Vector(s_tracks[1].bundle)).length < 1e-6:
        raise MifOpError("the two scale tracks sit at one point in the solve (the same feature tracked twice?) - pick "
                         "two apart. %s" % _CHANGED)
    use_median = take_bool(params, "useMedian", default=False)
    if not (plane or origin or axis or s_tracks):
        raise MifOpError("nothing to do - pass floor or wall, origin, xAxis or yAxis, or scale. %s" % _CHANGED)
    tr = clip.tracking
    if tr.objects.active != obj:
        tr.objects.active = obj
    done = []
    with _clip_editor(clip):
        if plane:
            _select_only(obj.tracks, plane)
            bpy.ops.clip.set_plane(plane="FLOOR" if params.get("floor") is not None else "WALL")
            done.append("plane")
        if origin:
            _select_only(obj.tracks, origin)
            bpy.ops.clip.set_origin(use_median=use_median)
            done.append("origin")
        if axis:
            _select_only(obj.tracks, axis)
            bpy.ops.clip.set_axis(axis="X" if params.get("xAxis") is not None else "Y")
            done.append("axis")
        if s_tracks:
            _select_only(obj.tracks, s_tracks)
            if obj.is_camera:
                bpy.ops.clip.set_scale(distance=s_dist)
            else:
                bpy.ops.clip.set_solution_scale(distance=s_dist)
            done.append("scale")
        _select_only(obj.tracks, [])
    cam = sc.camera
    bpy.context.view_layer.update()
    out = {"clip": clip.name, "object": obj.name, "applied": done, "camera": cam.name,
           "cameraLocation": rnd(cam.matrix_world.to_translation(), 5),
           "cameraRotation": rnd(cam.matrix_world.to_euler(), 5), "cameraScale": rnd(cam.matrix_world.to_scale(), 5)}
    # THE SOLVE IN THE WORLD: a camera solver puts the camera at (its own base matrix) @ (the reconstructed camera),
    # and the bundles sit in the reconstruction's space, so a bundle's world position is that base matrix @ bundle.
    base = (cam.parent.matrix_world if cam.parent is not None else Matrix.Identity(4)) @ cam.matrix_basis

    def world(t):
        return base @ Vector(t.bundle)
    if plane:
        pts = [world(t) for t in plane]
        out["planeTracks"] = [t.name for t in plane]
        out["planeResidual"] = round(max(abs(p.z if params.get("floor") is not None else p.y) for p in pts), 5)
    if s_tracks and obj.is_camera:
        out["scaleMeasured"] = round((world(s_tracks[0]) - world(s_tracks[1])).length, 5)
    return out


# ---------------------------------------------------------------------------------------------------------------
# edit_clip_tracks and build_tracking_scene (2026-10-08)
# ---------------------------------------------------------------------------------------------------------------
# MEASURED FIRST, 2026-10-08 on 5.2.1 (and the suite on every build): every clip operator below answers FINISHED in a
# background Blender inside a borrowed Clip Editor (_clip_editor) - clear_track_path, refine_markers, lock_tracks,
# average_tracks, join_tracks, create_plane_track, stabilize_2d_add / _rotation_add, bundles_to_mesh, track_to_empty,
# setup_tracking_scene, set_viewport_background and rebuild_proxy.

_TRACK_ACTIONS = ("CLEAR_PATH", "REFINE", "LOCK", "UNLOCK", "JOIN", "AVERAGE", "COPY_SETTINGS", "FILTER_SPIKES",
                  "PLANE", "TO_MESH", "TO_EMPTIES", "DELETE")
_EDIT_TRACK_KEYS = ("clip", "object", "action", "tracks", "frame", "clearMode", "backwards", "keepOriginal", "source",
                    "threshold", "name", "image", "opacity", "autoKeying")
_ACTION_KEYS = {"CLEAR_PATH": {"frame", "clearMode"}, "REFINE": {"frame", "backwards"}, "LOCK": set(), "UNLOCK": set(),
                "JOIN": set(), "AVERAGE": {"keepOriginal"}, "COPY_SETTINGS": {"source"}, "FILTER_SPIKES": {"threshold"},
                "PLANE": {"frame", "name", "image", "opacity", "autoKeying"}, "TO_MESH": {"name"}, "TO_EMPTIES": set(),
                "DELETE": set()}
# THE SETTINGS ONE TRACK HANDS ANOTHER: what Blender's track_settings_to_track copies (5.0 bl_operators/clip.py :981),
# done natively - its pattern and search areas excepted, which are marker data.
_COPY_TRACK_PROPS = ("motion_model", "pattern_match", "use_brute", "use_normalization", "use_mask", "use_red_channel",
                     "use_green_channel", "use_blue_channel", "correlation_min", "margin", "frames_limit", "weight",
                     "weight_stab", "use_grayscale_preview", "use_alpha_preview", "color", "use_custom_color")


def _named_tracks(obj, names, tail, need=None):
    if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
        raise MifOpError("tracks takes a list of track names. %s" % tail)
    out = []
    for n in dict.fromkeys(names):
        t = obj.tracks.get(n)
        if t is None:
            raise MifOpError("'%s' has no track '%s'. %s" % (obj.name, n, tail))
        out.append(t)
    if need is not None and len(out) != need:
        raise MifOpError("this action takes exactly %d tracks, got %d. %s" % (need, len(out), tail))
    return out


def _spikes(t, threshold, size):
    """Frames where a track's speed changes by more than threshold pixels a frame from one frame to the next - the jump
    a marker makes when it slips onto another feature for a frame."""
    pts = sorted((m.frame, m.co[0] * size[0], m.co[1] * size[1]) for m in t.markers if not m.mute)
    vel = [((b[0]), (b[1] - a[1]) / max(1, b[0] - a[0]), (b[2] - a[2]) / max(1, b[0] - a[0]))
           for a, b in zip(pts, pts[1:])]
    out = []
    for v0, v1 in zip(vel, vel[1:]):
        if math.hypot(v1[1] - v0[1], v1[2] - v0[2]) > threshold:
            out.append(v1[0])
    return out


def op_edit_clip_tracks(params):
    """Edit a movie clip's tracks: clear a path, refine, lock, join, average, copy settings, filter spikes, make a
    plane track, turn tracks into a mesh or empties, or delete them.

    THE GAP (spec, tracking LEFT list 2026-10-06): after track_clip and solve_clip every clean-up step was the Clip
    Editor's - clear_track_path, refine_markers, join_tracks, average_tracks, lock_tracks, copying one track's settings
    to others - and the plane track, the bundles as a mesh and the empties that follow tracks were raw Python.

    params:
      clip, object (str)        as track_clip (the camera's tracking object by default)
      action (str, required)    CLEAR_PATH {frame, clearMode UPTO | REMAINED | ALL} - markers before / after frame;
                                REFINE {frame, backwards}; LOCK / UNLOCK; JOIN (two or more into the first);
                                AVERAGE {keepOriginal}; COPY_SETTINGS {source} - the source track's tracker settings
                                onto tracks; FILTER_SPIKES {threshold px a frame, default 5} - delete tracks whose speed
                                jumps by more than threshold between frames (not Blender's filter_tracks, which first
                                deletes every track shorter than 10 frames); PLANE {name, frame, image, opacity,
                                autoKeying} - a plane track from exactly 4 tracks with markers at frame; TO_MESH {name}
                                - a vertex per solved track (Blender's bundles_to_mesh); TO_EMPTIES - an empty per
                                track with a Follow Track constraint (track_to_empty); DELETE
      tracks (list)             the track names it acts on (FILTER_SPIKES: default every track)

    Reply: before and after track counts, what the action made or changed (marker counts, the joined or averaged
    track, the plane track's corners, the objects made), and the clip's track summary.
    """
    reject_unknown(params, _EDIT_TRACK_KEYS, "edit_clip_tracks")
    tail = _CHANGED
    clip = _get_clip(take(params, "clip", kind=str), tail)
    obj = _tracking_object(clip, take(params, "object", kind=str), tail)
    action = str(take(params, "action", required=True, kind=str)).upper()
    if action not in _TRACK_ACTIONS:
        raise MifOpError("action is one of %s, got %r. %s" % (", ".join(_TRACK_ACTIONS), params.get("action"), tail))
    extra = [k for k in _EDIT_TRACK_KEYS if k not in ("clip", "object", "action", "tracks") and params.get(k) is not None
             and k not in _ACTION_KEYS[action]]
    if extra:
        raise MifOpError("%s does not take %s. %s" % (action, ", ".join(extra), tail))
    first, last = _clip_frames(clip)
    raw = params.get("tracks")
    if raw is None and action not in ("FILTER_SPIKES",):
        raise MifOpError("%s needs tracks. %s" % (action, tail))
    need = {"PLANE": 4}.get(action)
    tracks = _named_tracks(obj, raw, tail, need) if raw is not None else list(obj.tracks)
    if action == "JOIN" and len(tracks) < 2:
        raise MifOpError("JOIN takes two or more tracks. %s" % tail)
    if action == "AVERAGE" and len(tracks) < 2:
        raise MifOpError("AVERAGE takes two or more tracks. %s" % tail)
    frame = finite_int(params["frame"], "frame") if params.get("frame") is not None else first
    if not first <= frame <= last:
        raise MifOpError("frame %d is outside the clip's %d-%d. %s" % (frame, first, last, tail))
    mode = str(params.get("clearMode") or "UPTO").upper()
    if action == "CLEAR_PATH" and mode not in ("UPTO", "REMAINED", "ALL"):
        raise MifOpError("clearMode is UPTO, REMAINED or ALL. %s" % tail)
    source = None
    if action == "COPY_SETTINGS":
        sname = take(params, "source", required=True, kind=str)
        source = obj.tracks.get(sname)
        if source is None:
            raise MifOpError("'%s' has no source track '%s'. %s" % (obj.name, sname, tail))
    threshold = take_float(params, "threshold", default=5.0)
    if action == "FILTER_SPIKES" and not (math.isfinite(threshold) and threshold > 0):
        raise MifOpError("threshold must be above 0 pixels a frame. %s" % tail)
    if action == "TO_MESH" and not all(t.has_bundle for t in tracks):
        raise MifOpError("TO_MESH places a vertex at each track's solved bundle, and %s %s not solved - solve_clip "
                         "first. %s" % (", ".join(t.name for t in tracks if not t.has_bundle)[:300],
                                        "is" if sum(1 for t in tracks if not t.has_bundle) == 1 else "are", tail))
    if action == "PLANE":
        missing = [t.name for t in tracks if t.markers.find_frame(frame) is None or t.markers.find_frame(frame).mute]
        if missing:
            raise MifOpError("a plane track starts from four markers at one frame; %s %s none at frame %d. %s"
                             % (", ".join(missing), "has" if len(missing) == 1 else "have", frame, tail))
        if params.get("image") is not None and bpy.data.images.get(str(params["image"])) is None:
            raise MifOpError("no image named %r (load one first). %s" % (params["image"], tail))
        if params.get("opacity") is not None:
            op_v = take_float(params, "opacity")
            if not 0 <= op_v <= 1:
                raise MifOpError("opacity is 0-1. %s" % tail)
    # EVERY VALUE READ BEFORE THE FIRST WRITE. autoKeying was read after create_plane_track had made the plane track, so
    # a typo in it refused with "nothing was changed" and left the track behind; opacity went in through a bare float().
    backwards = take_bool(params, "backwards", default=False)
    keep_original = take_bool(params, "keepOriginal", default=True)
    opacity = take_float(params, "opacity")
    auto_keying = take_bool(params, "autoKeying") if params.get("autoKeying") is not None else None

    tr = clip.tracking
    if tr.objects.active != obj:
        tr.objects.active = obj
    before = len(obj.tracks)
    names = [t.name for t in tracks]
    out = {"clip": clip.name, "object": obj.name, "action": action, "tracks": names, "tracksBefore": before}
    if action in ("LOCK", "UNLOCK"):
        for t in tracks:
            t.lock = action == "LOCK"
        out["locked"] = [t.name for t in tracks if t.lock]
    elif action == "COPY_SETTINGS":
        copied = [p for p in _COPY_TRACK_PROPS if p in source.bl_rna.properties]
        for t in tracks:
            for p in copied:
                setattr(t, p, getattr(source, p))
        out["copied"] = copied
        out["source"] = source.name
        # READ BACK off the targets, two settings a caller can compare at a glance.
        out["readBack"] = {t.name: {"motion_model": t.motion_model, "correlation_min": round(t.correlation_min, 4)}
                           for t in tracks}
    elif action == "FILTER_SPIKES":
        doomed = {t.name: _spikes(t, threshold, clip.size) for t in tracks}
        doomed = {k: v for k, v in doomed.items() if v}
        if len(obj.tracks) - len(doomed) < 8 and doomed:
            out["note"] = "this leaves fewer than 8 tracks, too few to solve."
        with _clip_editor(clip):
            _delete_tracks(obj, [obj.tracks[n] for n in doomed])
        out["deleted"] = sorted(doomed)
        out["spikeFrames"] = {k: v[:20] for k, v in sorted(doomed.items())[:50]}
        out["threshold"] = threshold
    else:
        made_objs = set(bpy.data.objects.keys())
        planes0 = {p.name for p in obj.plane_tracks}
        names0 = {t.name for t in obj.tracks}
        with _clip_editor(clip) as ctx:
            # THE CLIP EDITOR'S OWN FRAME too (_clip_editor.frame): CLEAR_PATH UPTO 10 cleared nothing with only the
            # scene frame set (measured on 5.2.1).
            ctx.frame(frame)
            _select_only(obj.tracks, tracks)
            if tracks and obj.is_camera:
                tr.tracks.active = tracks[0]
            if action == "CLEAR_PATH":
                res = bpy.ops.clip.clear_track_path(action=mode, clear_active=False)
            elif action == "REFINE":
                res = bpy.ops.clip.refine_markers(backwards=backwards)
            elif action == "JOIN":
                res = bpy.ops.clip.join_tracks()
            elif action == "AVERAGE":
                res = bpy.ops.clip.average_tracks(keep_original=keep_original)
            elif action == "PLANE":
                res = bpy.ops.clip.create_plane_track()
            elif action == "TO_MESH":
                res = bpy.ops.clip.bundles_to_mesh()
            elif action == "TO_EMPTIES":
                res = bpy.ops.clip.track_to_empty()
            else:
                res = bpy.ops.clip.delete_track(confirm=False)
            _select_only(obj.tracks, [])
        if "FINISHED" not in res:
            raise MifOpError("Blender's %s answered %s%s." % (action.lower(), sorted(res),
                                                              (" - " + _JOB_BLOCK) if bpy.app.background else ""))
        out["madeTracks"] = sorted(t.name for t in obj.tracks if t.name not in names0)
        made = sorted(set(bpy.data.objects.keys()) - made_objs)
        if made:
            out["madeObjects"] = made
        if action == "CLEAR_PATH":
            out["markers"] = {t.name: sum(1 for m in t.markers if not m.mute) for t in tracks}
        elif action == "PLANE":
            new = [p for p in obj.plane_tracks if p.name not in planes0]
            if len(new) != 1:
                raise MifOpError("create_plane_track made %d plane tracks." % len(new))
            pt = new[0]
            if params.get("name"):
                pt.name = str(params["name"])
            if params.get("image") is not None:
                pt.image = bpy.data.images[str(params["image"])]
            if opacity is not None:
                pt.image_opacity = opacity
            if auto_keying is not None:
                pt.use_auto_keying = auto_keying
            mk = pt.markers.find_frame(frame)
            out["planeTrack"] = {"name": pt.name, "image": pt.image.name if pt.image else None,
                                 "opacity": round(pt.image_opacity, 4), "autoKeying": bool(pt.use_auto_keying),
                                 "markers": len(pt.markers), "corners": [rnd(c, 5) for c in mk.corners] if mk else None}
        elif action == "TO_MESH" and made:
            me = bpy.data.objects[made[0]].data
            out["vertices"] = len(me.vertices)
        elif action == "TO_EMPTIES":
            out["constraints"] = {n: [c.type for c in bpy.data.objects[n].constraints] for n in made}
    out["tracksAfter"] = len(obj.tracks)
    out.update(_tracks_summary(clip, obj))
    return out


_SCENE_PARTS = ("light", "ground", "cube", "backgroundLayer")
_BUILD_SCENE_KEYS = ("clip", "skip", "background")


def op_build_tracking_scene(params):
    """Build the scene a camera solve is composited in: the camera on a Camera Solver constraint, the Foreground and
    Background view layers and collections, a shadow-catcher ground, a light, a sample cube and the compositor's
    Movie Clip over-the-plate tree - Blender's setup_tracking_scene - with the parts asked to be left out removed.

    THE GAP (spec, tracking (h)): setup_tracking_scene and set_viewport_background poll for a Clip Editor holding a
    solved clip, so they were unreachable; they run in a borrowed Clip Editor.

    params:
      clip (str)            default the scene's active clip; its camera must be solved
      skip (list)           parts to leave out: light, ground, cube, backgroundLayer
      background (bool)     also show the clip behind the camera's view (set_viewport_background, default true)

    Reply: every object, collection, view layer and compositor node made, the camera's constraints and background.
    """
    reject_unknown(params, _BUILD_SCENE_KEYS, "build_tracking_scene")
    tail = _CHANGED
    clip = _get_clip(take(params, "clip", kind=str), tail)
    cam_obj = next((o for o in clip.tracking.objects if o.is_camera), None)
    if cam_obj is None or not cam_obj.reconstruction.is_valid:
        raise MifOpError("clip '%s' has no solved camera - solve_clip first. %s" % (clip.name, tail))
    skip = params.get("skip") or []
    if not isinstance(skip, list) or any(s not in _SCENE_PARTS for s in skip):
        raise MifOpError("skip takes a list of %s. %s" % (", ".join(_SCENE_PARTS), tail))
    bg = take_bool(params, "background", default=True)
    sc = bpy.context.scene
    if sc.camera is None:
        raise MifOpError("the scene has no camera - the tracking scene is built round it. %s" % tail)
    objs0, cols0 = set(bpy.data.objects.keys()), set(bpy.data.collections.keys())
    layers0 = {v.name for v in sc.view_layers}
    tree = getattr(sc, "compositing_node_group", None) or getattr(sc, "node_tree", None)
    nodes0 = {n.name for n in tree.nodes} if tree is not None else set()
    # THE CAMERA'S OBJECT ACTIVE: setup_tracking_scene polls the ACTIVE tracking object's reconstruction, so with an
    # object track active it failed its poll on a clip whose camera was solved (measured 2026-10-08, 3.6.23 and 5.2.1).
    if clip.tracking.objects.active != cam_obj:
        clip.tracking.objects.active = cam_obj
    with _clip_editor(clip):
        res = bpy.ops.clip.setup_tracking_scene()
        if bg:
            bpy.ops.clip.set_viewport_background()
    if "FINISHED" not in res:
        raise MifOpError("Blender's setup_tracking_scene answered %s." % sorted(res))
    made = sorted(set(bpy.data.objects.keys()) - objs0)
    removed = []
    by_part = {"light": "LIGHT", "ground": "Ground", "cube": "Cube"}
    for part in skip:
        if part == "backgroundLayer":
            vl = sc.view_layers.get("Background")
            if vl is not None and "Background" not in layers0:
                sc.view_layers.remove(vl)
                removed.append("view layer Background")
            continue
        for n in list(made):
            o = bpy.data.objects.get(n)
            if o is None:
                continue
            if (part == "light" and o.type == "LIGHT") or (part != "light" and n.startswith(by_part[part])):
                remove_objects([o])
                made.remove(n)
                removed.append(n)
    cam = sc.camera
    tree = getattr(sc, "compositing_node_group", None) or getattr(sc, "node_tree", None)
    out = {"clip": clip.name, "camera": cam.name, "objects": made,
           "collections": sorted(set(bpy.data.collections.keys()) - cols0),
           "viewLayers": [v.name for v in sc.view_layers], "removed": removed,
           "cameraConstraints": [c.type for c in cam.constraints],
           "cameraBackground": [b.source for b in cam.data.background_images] if bg else None,
           "compositorNodes": sorted({n.bl_idname for n in tree.nodes if n.name not in nodes0}) if tree else []}
    ground = next((bpy.data.objects[n] for n in made if n.startswith("Ground")), None)
    if ground is not None:
        out["groundShadowCatcher"] = bool(getattr(ground, "is_shadow_catcher", False))
    return out


# ---------------------------------------------------------------------------------------------------------------
# create_mask (second 1.3 audit, 2026-10-26): the mask editor was the one editor with no op. bpy.data.masks is plain data,
# so a mask is built here without any editor context - layers, splines of points or primitives, feather - and wired into
# a compositor Mask node or a strip's MASK modifier.
# ---------------------------------------------------------------------------------------------------------------
_MASK_KEYS = {"name", "layers", "compositor", "strip", "scene"}
_MASK_LAYER_KEYS = {"name", "splines", "invert", "blend", "alpha", "falloff"}
_MASK_SPLINE_KEYS = {"points", "primitive", "center", "size", "feather", "cyclic", "keys", "parent"}
_MASK_KEY_KEYS = {"frame", "points", "center", "size", "feather"}
_MASK_PARENT_KEYS = {"clip", "track", "object", "type", "frame"}
# KEYED POINTS AND TRACK PARENTS (1.3 wave 3, 2026-10-26). A key writes each point's co (and feather weight) as an
# F-curve on the mask at a frame. A parent is MaskSplinePoint.parent: id_type MOVIECLIP, the clip, the tracking object
# (parent) and the track (sub_parent). Blender's own Parent operator also stores the track's position at the moment of
# parenting (parent_orig) and evaluates the point at co + track(now) - parent_orig; that field is DNA only, not in RNA on
# 3.6-5.0 (read in the bundled binaries: "parent_orig[2]" and "parent_corners_orig[4][2]" appear only as DNA field
# names), so it stays 0 here and the same offset is taken off co instead: every point (and every keyed position) is
# written minus the track's position at the reference frame, so at that frame it sits where it was asked for and then
# follows the track. A PLANE_TRACK parent evaluates through parent_corners_orig, which cannot be set from Python, so it is
# refused by name.


def _mask_coord_from_clip(clip, co):
    """Blender's BKE_mask_coord_from_movieclip: a clip-normalized position in the mask's space (the frame's aspect, with
    the camera's pixel aspect, folded into the shorter axis)."""
    w, h = float(clip.size[0]), float(clip.size[1])
    pa = float(getattr(clip.tracking.camera, "pixel_aspect", 1.0) or 1.0)
    h = h / pa
    x, y = float(co[0]), float(co[1])
    if w == h:
        return x, y
    if w < h:
        return (x - 0.5) * (w / h) + 0.5, y
    return x, (y - 0.5) * (h / w) + 0.5


def _mask_parent_plan(sp, where, tail):
    """Resolve a spline's parent before anything is created: (clip, tracking object, track, offset [x, y] in mask
    space, reference frame)."""
    par = sp.get("parent")
    if not isinstance(par, dict) or set(par) - _MASK_PARENT_KEYS or not par.get("track"):
        raise MifOpError("%s.parent is {track (required), clip?, object?, type?, frame?}. %s" % (where, tail))
    ptype = str(par.get("type") or "POINT_TRACK").upper()
    if ptype == "PLANE_TRACK":
        raise MifOpError("%s.parent type PLANE_TRACK: a plane-track parent is evaluated through the corners stored when "
                         "it was parented (parent_corners_orig), which this Blender's Python API does not expose - "
                         "parent to a point track, or parent in the Mask editor. %s" % (where, tail))
    if ptype != "POINT_TRACK":
        raise MifOpError("%s.parent type is POINT_TRACK, got %r. %s" % (where, ptype, tail))
    clip = _get_clip(par.get("clip"), tail)
    if not clip.size[0] or not clip.size[1]:
        raise MifOpError("the clip '%s' has no frame size (its footage is not loaded), so a track position cannot be "
                         "put in the mask's space. %s" % (clip.name, tail))
    oname = par.get("object")
    tobj = clip.tracking.objects.get(oname) if oname else next((o for o in clip.tracking.objects if o.is_camera), None)
    if tobj is None:
        raise MifOpError("the clip '%s' has no tracking object %r. Objects: %s. %s"
                         % (clip.name, oname, ", ".join(o.name for o in clip.tracking.objects), tail))
    track = tobj.tracks.get(str(par["track"]))
    if track is None:
        raise MifOpError("the tracking object '%s' has no track %r. Tracks: %s. %s"
                         % (tobj.name, par["track"], ", ".join(t.name for t in tobj.tracks)[:400] or "(none)", tail))
    sc = bpy.context.scene
    ref = finite_int(par.get("frame", sc.frame_current), "parent.frame")
    clip_frame = ref - int(clip.frame_start) + 1
    marker = track.markers.find_frame(clip_frame)
    if marker is None or marker.mute:
        raise MifOpError("the track '%s' has no %smarker at scene frame %d (clip frame %d) - the reference frame is "
                         "where the mask sits as given; pick a frame the track has. %s"
                         % (track.name, "enabled " if marker is not None else "", ref, clip_frame, tail))
    off = _mask_coord_from_clip(clip, marker.co)
    return clip, tobj, track, [off[0], off[1]], ref


def _mask_keys_plan(sp, npts, where, tail):
    """[(frame, [[x, y], ...], feather or None)] - each key gives every point of the spline."""
    keys = sp.get("keys")
    if not isinstance(keys, list) or not keys:
        raise MifOpError("%s.keys is a non-empty list of {frame, points | center + size, feather?}. %s" % (where, tail))
    out, seen = [], set()
    for ki, k in enumerate(keys):
        kw = "%s.keys[%d]" % (where, ki)
        if not isinstance(k, dict) or set(k) - _MASK_KEY_KEYS or k.get("frame") is None:
            raise MifOpError("%s takes %s with frame required. %s" % (kw, sorted(_MASK_KEY_KEYS), tail))
        frame = finite_int(k["frame"], "frame")
        if frame in seen:
            raise MifOpError("%s: frame %d is keyed twice. %s" % (kw, frame, tail))
        seen.add(frame)
        if k.get("points") is not None:
            raw = k["points"]
            if not isinstance(raw, list) or len(raw) != npts:
                raise MifOpError("%s.points gives every point of the spline - %d [x, y]. %s" % (kw, npts, tail))
            pts = [finite_floats(list(p)[:2], "points") for p in raw]
        elif sp.get("primitive") is not None and k.get("center") is not None:
            c = finite_floats(list(k["center"])[:2], "center")
            size = finite_floats([k.get("size", sp.get("size", 0.5))], "size")[0]
            if not size > 0:
                raise MifOpError("%s.size is above 0. %s" % (kw, tail))
            pts = _mask_primitive(str(sp["primitive"]).upper(), c, size, tail)[0]
        else:
            raise MifOpError("%s needs points (or, on a primitive spline, center and size). %s" % (kw, tail))
        feather = None
        if k.get("feather") is not None:
            feather = finite_floats([k["feather"]], "feather")[0]
            if not 0.0 <= feather <= 1.0:
                raise MifOpError("%s.feather is 0-1. %s" % (kw, tail))
        out.append((frame, pts, feather))
    return sorted(out, key=lambda r: r[0])


def _mask_primitive(kind, center, size, tail):
    cx, cy = center
    if kind == "SQUARE":
        h = size / 2.0
        return [[cx - h, cy - h], [cx + h, cy - h], [cx + h, cy + h], [cx - h, cy + h]], "VECTOR"
    if kind == "CIRCLE":
        r = size / 2.0
        return [[cx + r * math.cos(a * math.pi / 2), cy + r * math.sin(a * math.pi / 2)] for a in range(4)], "AUTO"
    raise MifOpError("primitive is CIRCLE or SQUARE, got %r. %s" % (kind, tail))


def op_create_mask(params):
    """A mask - layers of splines, each a list of points or a CIRCLE / SQUARE primitive, with feather - wired into a
    compositor Mask node and / or a sequencer strip's MASK modifier.

    params:
      name (str, required)       the new mask (an existing name is refused)
      layers [{name?, splines: [{points [[x, y], ...] | primitive CIRCLE|SQUARE + center [x, y] + size, feather (0-1,
                                 each point's feather weight), cyclic (default true),
                                 keys [{frame, points [[x, y], ...] (every point) | center + size (a primitive),
                                        feather?}] - the points keyed at those frames (F-curves on the mask),
                                 parent {track, clip?, object? (tracking object, default the camera's), type? POINT_TRACK,
                                         frame? (scene frame the points are given at, default the current)} - every
                                 point follows that motion-tracking track}], invert, blend, alpha}]
                                 coordinates are the mask's own space: 0 to 1 across the frame's width, (0, 0) at the
                                 bottom left; CIRCLE gets AUTO handles (a smooth round), SQUARE and point lists VECTOR
      compositor (bool)          add a Mask node with it to the scene's compositing tree (refused when it has none)
      strip (str)                add a MASK modifier using it to that strip
      scene                      the scene (default the current one; the strip's own with strip)
    Reply: per layer the splines and points read back, and what it was wired into.
    """
    reject_unknown(params, _MASK_KEYS, "create_mask")
    tail = _ADDED
    name = take(params, "name", required=True, kind=str)
    if bpy.data.masks.get(name) is not None:
        raise MifOpError("a mask named '%s' exists - give another name. %s" % (name, tail))
    layers = params.get("layers")
    if not isinstance(layers, list) or not layers or not all(isinstance(lr, dict) for lr in layers):
        raise MifOpError("layers is a non-empty list of {name?, splines, invert?, blend?, alpha?}. %s" % tail)
    plan = []
    for li, lr in enumerate(layers):
        extra = set(lr) - _MASK_LAYER_KEYS
        if extra:
            raise MifOpError("layers[%d] takes %s; unknown %s. %s" % (li, sorted(_MASK_LAYER_KEYS), sorted(extra), tail))
        sps = lr.get("splines")
        if not isinstance(sps, list) or not sps:
            raise MifOpError("layers[%d].splines is a non-empty list. %s" % (li, tail))
        # THE LAYER'S OWN VALUES BEFORE masks.new (2026-10-28): name, blend, alpha, falloff and invert went through
        # str() / float() / bool() after the mask was made, so a bad one raised raw out of a half-built mask, and a
        # mapping named the layer or switched invert on.
        lset = {}
        if lr.get("name") is not None and not isinstance(lr["name"], str):
            raise MifOpError("layers[%d].name is text, got %r. %s" % (li, lr["name"], tail))
        if lr.get("invert") is not None:
            if not isinstance(lr["invert"], bool):
                raise MifOpError("layers[%d].invert is true or false, got %r. %s" % (li, lr["invert"], tail))
            lset["invert"] = lr["invert"]
        if lr.get("alpha") is not None:
            lset["alpha"] = finite_floats([lr["alpha"]], "layers[%d].alpha" % li)[0]
            if not 0.0 <= lset["alpha"] <= 1.0:
                raise MifOpError("layers[%d].alpha is 0-1, got %g. %s" % (li, lset["alpha"], tail))
        for k in ("blend", "falloff"):
            if lr.get(k) is not None:
                items = [e.identifier for e in bpy.types.MaskLayer.bl_rna.properties[k].enum_items]
                v = lr[k].upper() if isinstance(lr[k], str) else None
                if v not in items:
                    raise MifOpError("layers[%d].%s is one of %s, got %r. %s" % (li, k, ", ".join(items), lr[k], tail))
                lset[k] = v
        rows = []
        for si, sp in enumerate(sps):
            if not isinstance(sp, dict) or set(sp) - _MASK_SPLINE_KEYS:
                raise MifOpError("layers[%d].splines[%d] takes %s. %s" % (li, si, sorted(_MASK_SPLINE_KEYS), tail))
            if (sp.get("points") is None) == (sp.get("primitive") is None):
                raise MifOpError("layers[%d].splines[%d]: give points OR primitive. %s" % (li, si, tail))
            if sp.get("primitive") is not None:
                c = finite_floats(list(sp.get("center") or [0.5, 0.5])[:2], "center")
                size = finite_floats([sp.get("size", 0.5)], "size")[0]
                if not size > 0:
                    raise MifOpError("layers[%d].splines[%d].size is above 0. %s" % (li, si, tail))
                pts, handle = _mask_primitive(str(sp["primitive"]).upper(), c, size, tail)
            else:
                raw = sp["points"]
                if not isinstance(raw, list) or len(raw) < 2:
                    raise MifOpError("layers[%d].splines[%d].points needs at least 2 [x, y]. %s" % (li, si, tail))
                pts, handle = [finite_floats(list(p)[:2], "points") for p in raw], "VECTOR"
            feather = finite_floats([sp.get("feather", 0.0)], "feather")[0]
            if not 0.0 <= feather <= 1.0:
                raise MifOpError("feather is 0-1, got %g. %s" % (feather, tail))
            where = "layers[%d].splines[%d]" % (li, si)
            keys = _mask_keys_plan(sp, len(pts), where, tail) if sp.get("keys") is not None else None
            parent = _mask_parent_plan(sp, where, tail) if sp.get("parent") is not None else None
            if sp.get("cyclic") is not None and not isinstance(sp["cyclic"], bool):
                raise MifOpError("%s.cyclic is true or false, got %r. %s" % (where, sp["cyclic"], tail))
            rows.append((pts, handle, feather, sp.get("cyclic", True) is not False, keys, parent))
        plan.append((lr, lset, rows))
    sc = _seq_scene(params, tail) if params.get("strip") else (bpy.data.scenes.get(params["scene"]) if params.get("scene")
                                                                else bpy.context.scene)
    if sc is None:
        raise MifOpError("no scene named '%s'. %s" % (params.get("scene"), tail))
    st = _find_strip(sc, params["strip"], tail) if params.get("strip") else None
    if st is not None and tuple(bpy.app.version[:2]) == (5, 0):
        # MEASURED 2026-10-26: a MASK strip modifier given a mask takes Blender 5.0.1 down (the session exits); 3.6, 4.2,
        # 4.4 and 5.2 wire it. Refused here rather than crash somebody's Blender.
        raise MifOpError("a strip MASK modifier crashes Blender %s (measured on 5.0.1); wire the mask in the compositor "
                         "here, or use 4.4 or 5.2 for the strip. %s" % (bpy.app.version_string, tail))
    tree = None
    if params.get("compositor"):
        from .ops_nodes import _scene_tree                                # noqa: PLC0415
        tree = _scene_tree()
    mask = bpy.data.masks.new(name)
    out_layers = []
    for lr, lset, rows in plan:
        layer = mask.layers.new(name=lr.get("name") or "Layer")
        for k, v in lset.items():
            setattr(layer, k, v)
        extras = []
        for si, (pts, handle, feather, cyclic, keys, parent) in enumerate(rows):
            spl = layer.splines.new()
            spl.points.add(len(pts) - len(spl.points))     # a new spline starts with ONE point (measured)
            off = parent[3] if parent is not None else [0.0, 0.0]
            for p, xy in zip(spl.points, pts):
                p.co = (xy[0] - off[0], xy[1] - off[1])
                p.handle_left_type = p.handle_right_type = handle
                p.weight = feather
            spl.use_cyclic = cyclic
            extra = {}
            if parent is not None:
                clip, tobj, track, _off, ref = parent
                for p in spl.points:
                    p.parent.id_type = "MOVIECLIP"
                    p.parent.id = clip
                    p.parent.parent = tobj.name
                    p.parent.sub_parent = track.name
                    p.parent.type = "POINT_TRACK"
                p0 = spl.points[0].parent
                extra["parent"] = {"clip": getattr(p0.id, "name", None), "object": p0.parent, "track": p0.sub_parent,
                                   "type": p0.type, "referenceFrame": ref, "offset": [round(c, 6) for c in off],
                                   "note": "points are stored minus the track's position at the reference frame (the "
                                           "offset), so they sit as given there and move with the track"}
            if keys is not None:
                # KEYED THROUGH THE MASK, by path. A MaskSplinePoint cannot make its own RNA path, so
                # point.keyframe_insert("co") raises "could not make path to 'co'" - measured 2026-10-28 on 3.6.23 and
                # 5.2.1, found by the version matrix: keys had never worked on any build. The Mask ID keys the same
                # property by its full path, which is the path _mask_key_readback reads.
                base = 'layers["%s"].splines[%d].points[%%d].' % (bpy.utils.escape_identifier(layer.name),
                                                                    len(layer.splines) - 1)
                try:
                    for frame, kpts, kfeather in keys:
                        for pi, (p, xy) in enumerate(zip(spl.points, kpts)):
                            p.co = (xy[0] - off[0], xy[1] - off[1])
                            if not mask.keyframe_insert(base % pi + "co", frame=frame):
                                raise RuntimeError("keyframe_insert('co') answered False")
                            if kfeather is not None:
                                p.weight = kfeather
                                mask.keyframe_insert(base % pi + "weight", frame=frame)
                except (TypeError, RuntimeError, ValueError) as exc:
                    bpy.data.masks.remove(mask)
                    raise MifOpError("this Blender (%s) did not key a mask point: %s. The mask was removed. %s"
                                     % (bpy.app.version_string, exc, tail))
                extra["keys"] = _mask_key_readback(mask, layer, si, keys, off)
            extras.append(extra)
        out_layers.append({"name": layer.name, "splines": [dict({"points": [[round(c, 6) for c in p.co] for p in s.points],
                                                                 "cyclic": s.use_cyclic,
                                                                 "feather": round(s.points[0].weight, 6) if len(s.points)
                                                                 else None}, **ex)
                                                           for s, ex in zip(layer.splines, extras)]})
    wired = {}
    if tree is not None:
        node = tree.nodes.new("CompositorNodeMask")
        node.mask = mask
        wired["compositorNode"] = node.name
    if st is not None:
        mod = st.modifiers.new(name="Mask_" + mask.name, type="MASK")
        mod.input_mask_type = "ID"
        mod.input_mask_id = mask
        wired["stripModifier"] = {"strip": st.name, "modifier": mod.name,
                                  "readBack": getattr(mod.input_mask_id, "name", None)}
    return {"mask": mask.name, "layers": out_layers, "wired": wired or None,
            "space": "0 to 1 across the frame's width, (0, 0) bottom left"}


def _mask_key_readback(mask, layer, si, keys, off):
    """Each key's points as the mask's F-curves give them at its frame (the parent offset put back), and the channels."""
    curves = {}
    for fc in _fcurves(mask):
        dp = fc.data_path
        if '"%s"' % layer.name in dp and (".splines[%d].points[" % si) in dp and dp.endswith("].co"):
            try:
                pi = int(dp.rsplit("points[", 1)[1].split("]", 1)[0])
            except (IndexError, ValueError):
                continue
            curves[(pi, fc.array_index)] = fc
    rows = []
    for frame, kpts, _f in keys:
        got = []
        for pi in range(len(kpts)):
            fx, fy = curves.get((pi, 0)), curves.get((pi, 1))
            got.append([round(fx.evaluate(frame) + off[0], 6) if fx else None,
                        round(fy.evaluate(frame) + off[1], 6) if fy else None])
        rows.append({"frame": frame, "points": got})
    return {"channels": len(curves), "readBack": rows}


OPS = {
    "create_mask": op_create_mask,
    "add_strip": op_add_strip,
    "edit_strip": op_edit_strip,
    "list_strips": op_list_strips,
    "set_sequencer": op_set_sequencer,
    "load_clip": op_load_clip,
    "list_clip": op_list_clip,
    "set_clip": op_set_clip,
    "track_clip": op_track_clip,
    "solve_clip": op_solve_clip,
    "orient_clip": op_orient_clip,
    "edit_clip_tracks": op_edit_clip_tracks,
    "build_tracking_scene": op_build_tracking_scene,
}
