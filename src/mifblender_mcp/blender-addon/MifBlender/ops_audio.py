"""Sound in a Blender scene: its 3D-audio settings, speakers and sequencer sound strips, the audio mixdown, and
keys baked from a sound's loudness.

EVERY BEHAVIOR THIS MODULE LEANS ON WAS MEASURED before it was written - on 3.6.23, 4.2.17 LTS, 4.4.0, 5.0.1 and
5.2.1 LTS, headless (--background --factory-startup), 2026-09-23 - by mixing scenes down and reading the files back
sample by sample. An operator's FINISHED was never taken as evidence: on 5.2 the mixdown answers FINISHED before a
byte exists, and a speaker set up the obvious way mixes to silence on every build.

=============================================================================
A SPEAKER PLAYS THROUGH AN NLA SOUND STRIP, AND ONLY AN OPERATOR MAKES ONE
=============================================================================
Blender adds one playback per NLA strip of type SOUND on a speaker object, starting at strip.start / fps, and ignores
the strip's end (source/blender/blenkernel/intern/sound.cc, sound_update_base). A speaker with no such strip is
silent. NlaStrips.new() takes an Action and makes a CLIP strip on every build, so a speaker built through bpy.data
alone never plays. bpy.ops.object.speaker_add makes the object AND a SOUND strip at the scene's current frame, and it
works headless on every build - so add_sound moves frame_current to the start frame around that one call.

=============================================================================
A SOUND ASSIGNED TO A SPEAKER IS NOT LOADED UNTIL THE DEPENDENCY GRAPH IS REBUILT
=============================================================================
Speaker.sound written from Python leaves the EVALUATED speaker holding the original sound (is_evaluated False),
which Blender never loads for playback: the mixdown is silent. update_tag() alone does not change that, and neither
does a frame change or a view layer update. Re-assigning the object's data (which rebuilds relations) and then
tagging the speaker does: the evaluated sound becomes the evaluated copy and the sound lands on its exact sample, on
all five builds. add_sound does that and checks the result; mixdown_audio repairs any speaker left the other way.

=============================================================================
TIME IN A MIXDOWN
=============================================================================
Sample 0 of a mixdown is its first frame, so a sound placed at frame F starts at sample (F - frameStart) x rate / fps
- to the sample, for strips and for speakers. A strip's volume key at frame K is NOT sample-exact: Blender samples the
curve once a frame, and the level reaches the key's value between half a frame before K and one frame after it. On
3.6-5.0 it lands on K's first sample or ramps over the frame after K, depending on K and identically on those three
builds (on time at 20-43, 97, 150 and 200; late at 13, 49-85 and 113; LINEAR and CONSTANT alike); on 5.2 it falls over
the half frame before K. A strip lasts round(seconds x fps) frames, at least one.

=============================================================================
Scene.use_audio MEANS OPPOSITE THINGS ON 3.6 AND 4.2+
=============================================================================
On 3.6 the property is "Audio Muted" and True silences the mixdown; from 4.2 it is "Play Audio" and False does. A
check written against one build reads the other backwards - see _scene_muted.

=============================================================================
5.2 MIXES DOWN IN A JOB, AND A BACKGROUND BLENDER RUNS ONLY ONE
=============================================================================
Up to 5.0 sound.mixdown writes the file before it returns. On 5.2 it starts a window-manager job (sound_ops.cc,
WM_JOB_TYPE_SOUND_MIXDOWN) and returns FINISHED at once; a worker thread writes the file about 20 ms later. The job
is retired only by the UI's job timer, which a --background Blender never runs - so the finished job stays
"running", and every later mixdown job in that process is suspended behind it and never writes, whatever scene it
belongs to (measured: the second mixdown of a session, same scene or another, writes nothing). mixdown_audio waits
for the file, and refuses with that reason when none appears.
"""
import math
import os
import struct
import time
import uuid

import bpy

from .ops_anim import _fcurves
from .ops_common import (remove_objects, MifOpError, assign_id_name, check_axis_dict, check_output_path, check_writable_dir,
                         finite_floats, get_object, reject_unknown, require_editable, rnd, selection_restore,
                         selection_snapshot, take, take_float, take_int)
from .ops_undo import _run as _run_with_window

# ---------------------------------------------------------------------------------------------------------------
# shared readers
# ---------------------------------------------------------------------------------------------------------------


def _fps(sc):
    """The scene's real frame rate - fps over fps_base, so 29.97 is 29.97 and not 30."""
    return float(sc.render.fps) / float(sc.render.fps_base or 1.0)


# THE SCENE A SEQUENCER CALL ACTS ON (moved here from ops_video 2026-10-08 so add_sound and mixdown_audio share it;
# ops_video imports it from here).
def _workspace():
    win = bpy.context.window
    if win is None:
        wm = bpy.context.window_manager
        win = wm.windows[0] if wm is not None and wm.windows else None
    return win.workspace if win is not None else getattr(bpy.context, "workspace", None)


def _sequencer_scene_of_workspace():
    """The workspace's sequencer_scene on 5.0 and later (None when unset), or None on a build without one."""
    ws = _workspace()
    if ws is None or "sequencer_scene" not in ws.bl_rna.properties:
        return None
    return ws.sequencer_scene


def _seq_scene(params, tail):
    """The scene whose sequencer a call acts on: `scene` by name; else on 5.0+ the workspace's sequencer scene when one
    is set (the Sequencer shows it, not the window's scene); else the window's scene."""
    name = take(params, "scene", kind=str)
    if name is not None:
        sc = bpy.data.scenes.get(name)
        if sc is None:
            raise MifOpError("there is no scene named '%s'. Scenes: %s. %s"
                             % (name, ", ".join(s.name for s in bpy.data.scenes), tail))
        return sc
    return _sequencer_scene_of_workspace() or bpy.context.scene


def _hard_range(struct_rna, prop):
    """(hard_min, hard_max) of an RNA property, read off the running build rather than remembered."""
    p = struct_rna.bl_rna.properties[prop]
    return float(p.hard_min), float(p.hard_max)


def _in_range(value, lo, hi, key, tail):
    """Refuse a value Blender would CLAMP - it stores the nearest bound without a word, and the read-back then agrees
    with itself while disagreeing with the request."""
    if not (lo <= value <= hi):
        raise MifOpError("'%s' must be between %g and %g on this Blender - it clamps anything outside that without a "
                         "word, so %g would be stored as %g. %s" % (key, lo, hi, value, min(max(value, lo), hi), tail))


def _clean_name(name, tail):
    """A requested name, or a refusal. A control character collapses on the way into Blender's fixed-size name."""
    if name is None:
        return None
    if not name.strip():
        raise MifOpError("'name' is blank - leave it out to take Blender's default name. %s" % tail)
    if any(ord(ch) < 32 for ch in name):
        raise MifOpError("'name' contains a control character, which Blender's names cannot hold - got %r. %s"
                         % (name, tail))
    return name


def _round_half_up(x):
    return int(math.floor(x + 0.5))


def _read_sound(raw, tail):
    """(absolute path, sample rate, channels, samples) of a sound file as Blender's own audio module (aud) reads it.

    READ BEFORE ANYTHING IS CREATED, and it has to be this module: bpy.data.sounds.load() accepts a path that does
    not exist and a text file named .wav alike - both load "without error" into a datablock with samplerate 0 and
    channels INVALID, on every build (measured) - and a sequencer strip is made from either just as happily. aud is
    what the mixer reads with, so a file it can read is a file that will play.
    """
    path = bpy.path.abspath(raw)
    if not os.path.isfile(path):
        raise MifOpError("no file at '%s' - pass the path of a sound file. %s" % (path, tail))
    try:
        import aud
    except ImportError:
        raise MifOpError("this Blender has no aud module, which is how it reads sound files. %s" % tail) from None
    try:
        snd = aud.Sound(path)
        rate, channels = snd.specs
        samples = int(snd.length)
    except aud.error as exc:
        raise MifOpError("'%s' could not be read as sound by Blender's audio module, aud (%s) - it reads WAV, FLAC, "
                         "MP3, OGG and whatever else its FFmpeg build decodes. %s" % (path, exc, tail)) from None
    if samples <= 0 or rate <= 0:
        raise MifOpError("'%s' decodes to no samples (length %d at %g Hz), so there is nothing to play. %s"
                         % (path, samples, rate, tail))
    return path, float(rate), int(channels), samples


def _top_strips(ed):
    """The editor's own strip list - `strips` from 4.4 (5.0 has only that name), `sequences` before. By capability."""
    return ed.strips if hasattr(ed, "strips") else ed.sequences


def _all_strips(ed):
    """Every strip, inside metas too - strips_all from 4.4, sequences_all before."""
    return ed.strips_all if hasattr(ed, "strips_all") else ed.sequences_all


def _span(st):
    """(first frame, frame after the last) a strip occupies. 5.2 renamed these to left_handle / right_handle and
    prints a deprecation warning for frame_final_*, so the new names are used wherever the build has them."""
    if "left_handle" in st.bl_rna.properties:
        return int(st.left_handle), int(st.right_handle)
    return int(st.frame_final_start), int(st.frame_final_end)


def _strip_rna():
    """The sound strip's RNA class on this build - SoundStrip from 4.4, SoundSequence before."""
    return getattr(bpy.types, "SoundStrip", None) or getattr(bpy.types, "SoundSequence", None)


def _vec3(params, key, default, tail):
    """[x, y, z] or {x, y, z} - a partial dict keeps the default's other axes, a dict naming none is a typo."""
    val = params.get(key)
    if val is None:
        return list(default)
    if isinstance(val, dict):
        check_axis_dict(val, key, ("x", "y", "z"), tail)
        return finite_floats([val.get("x", default[0]), val.get("y", default[1]), val.get("z", default[2])], key)
    if isinstance(val, (list, tuple)) and len(val) == 3:
        return finite_floats(val, key)
    raise MifOpError("'%s' takes [x, y, z] or {x, y, z}, got %r. %s" % (key, val, tail))


def _evaluated_sound(obj):
    """The sound the EVALUATED speaker plays - None, the original (never loaded: silent) or the evaluated copy."""
    dg = bpy.context.evaluated_depsgraph_get()
    return obj.evaluated_get(dg).data.sound


def _speaker_plays(obj):
    """(plays, cause, reason) for a speaker object, judged the way Blender's mixer does. cause is a fixed word a
    caller can branch on - no_sound, muted, no_strip, not_loaded - and reason is the sentence."""
    spk = obj.data
    if spk.sound is None:
        return False, "no_sound", "it has no sound"
    if spk.muted:
        return False, "muted", "the speaker is muted"
    ad = obj.animation_data
    strips = [s for t in (ad.nla_tracks if ad else ()) for s in t.strips if s.type == "SOUND" and not s.mute]
    if not strips:
        return False, "no_strip", ("it has no unmuted NLA sound strip, and a speaker plays only through one - "
                                   "add_sound makes the speaker with its strip")
    ev = _evaluated_sound(obj)
    if ev is None or not ev.is_evaluated:
        return False, "not_loaded", "its sound was set after the dependency graph was built, so Blender never loaded it"
    return True, None, None


def _speaker_readback(obj):
    """What a speaker IS, read off the object and its data - one reader for add_sound's reply."""
    spk = obj.data
    ad = obj.animation_data
    strips = [(t, s) for t in (ad.nla_tracks if ad else ()) for s in t.strips if s.type == "SOUND"]
    ev = _evaluated_sound(obj)
    sc = bpy.context.scene
    listener = sc.camera
    out = {
        "object": obj.name, "speakerData": spk.name, "speakerSound": spk.sound.name if spk.sound else None,
        "location": rnd(list(obj.matrix_world.to_translation())),
        "volume": round(float(spk.volume), 6), "pitch": round(float(spk.pitch), 6),
        "distanceReference": round(float(spk.distance_reference), 6),
        "distanceMax": round(float(spk.distance_max), 6),
        "attenuation": round(float(spk.attenuation), 6),
        # DEGREES, not radians: the RNA range is 0-360 with no angle subtype, on every build.
        "coneAngleInner": round(float(spk.cone_angle_inner), 6),
        "coneAngleOuter": round(float(spk.cone_angle_outer), 6),
        "coneVolumeOuter": round(float(spk.cone_volume_outer), 6),
        "soundStrips": [{"track": t.name, "strip": s.name, "frameStart": round(float(s.frame_start), 4),
                         "frameEnd": round(float(s.frame_end), 4), "mute": bool(s.mute)} for t, s in strips],
        # THE MEASUREMENT THAT MATTERS: whether the depsgraph holds a LOADED copy of the sound. False is a speaker the
        # mixdown drops in silence.
        "soundLoaded": bool(ev is not None and ev.is_evaluated),
        "collections": sorted(c.name for c in obj.users_collection),
        "distanceModel": sc.audio_distance_model,
        "listener": listener.name if listener else None,
    }
    if listener is not None:
        out["distanceToListener"] = round(float((listener.matrix_world.to_translation()
                                                 - obj.matrix_world.to_translation()).length), 6)
    return out


# ---------------------------------------------------------------------------------------------------------------
# set_scene_audio
# ---------------------------------------------------------------------------------------------------------------

_SCENE_AUDIO_KEYS = {"distanceModel", "dopplerSpeed", "dopplerFactor", "volume"}
_SCENE_AUDIO_FLOATS = (("dopplerSpeed", "audio_doppler_speed"), ("dopplerFactor", "audio_doppler_factor"),
                       ("volume", "audio_volume"))


def _scene_audio(sc):
    return {"distanceModel": sc.audio_distance_model,
            "dopplerSpeed": round(float(sc.audio_doppler_speed), 6),
            "dopplerFactor": round(float(sc.audio_doppler_factor), 6),
            "volume": round(float(sc.audio_volume), 6)}


def op_set_scene_audio(params):
    """Set the active scene's 3D audio - the distance model, the Doppler speed and factor, and the master volume -
    and read every value back with the distance models this Blender offers.

    params:
      distanceModel (str)   how a speaker's level falls with distance - one of this build's own list, read from RNA
      dopplerSpeed (float)  the speed of sound in m/s that Doppler shifts are computed from; at least 0.01
      dopplerFactor (float) how strong the Doppler shift is; 0 turns it off
      volume (float)        the scene's master volume, 0-100

    With no parameters it only reads. Every value is checked against the property's own range BEFORE any is written:
    Blender clamps an out-of-range value without a word. Works on the active scene - set_current_scene switches it.
    The listener is the scene camera (sound.cc takes its position and rotation), so a scene with no camera has no
    listener to measure speaker distances from; `listener` says which.
    """
    reject_unknown(params, _SCENE_AUDIO_KEYS, "set_scene_audio", notes={
        "scene": "set_scene_audio works on the active scene - set_current_scene switches it",
        "listener": ("the listener is the scene camera; move the camera, or make another the scene camera with "
                     "set_camera makeActive"),
        "speedOfSound": "spelled dopplerSpeed - Blender's audio_doppler_speed, the speed of sound in m/s",
        "doppler": "two settings: dopplerSpeed (the speed of sound, m/s) and dopplerFactor (0 turns Doppler off)",
    })
    sc = bpy.context.scene
    tail = "NOTHING was changed."
    models = [i.identifier for i in sc.bl_rna.properties["audio_distance_model"].enum_items]
    want = {}
    model = take(params, "distanceModel", default=None, kind=str)
    if model is not None:
        match = [m for m in models if m == model.strip().upper()]
        if not match:
            raise MifOpError("distanceModel '%s' is not one this Blender (%s) offers. Valid: %s. %s"
                             % (model, bpy.app.version_string, ", ".join(models), tail))
        want["audio_distance_model"] = match[0]
    for key, attr in _SCENE_AUDIO_FLOATS:
        try:
            value = take_float(params, key, default=None)
        except MifOpError as exc:
            raise MifOpError("%s %s" % (str(exc).rstrip("."), tail)) from None
        if value is None:
            continue
        lo, hi = _hard_range(sc, attr)
        _in_range(value, lo, hi, key, tail)
        want[attr] = value

    before = _scene_audio(sc)
    for attr, value in want.items():
        setattr(sc, attr, value)
    after = _scene_audio(sc)
    out = dict(after)
    out.update({
        "scene": sc.name,
        "distanceModels": models,
        "changed": sorted(k for k in after if after[k] != before[k]),
        "before": before,
        "listener": sc.camera.name if sc.camera else None,
    })
    if sc.camera is None:
        out["listenerNote"] = ("this scene has no camera, and Blender takes the 3D-audio listener's position from the "
                               "scene camera - speaker distances and directions have nothing to be measured from.")
    # AN ANIMATED MASTER VOLUME WINS OVER THIS WRITE on every frame the scene is evaluated at, so say so.
    if any(fc.data_path == "audio_volume" for fc in _fcurves(sc)):
        out["volumeNote"] = ("the scene's audio_volume is animated - its keys, not this value, set the level during "
                             "playback and the mixdown.")
    return out


# ---------------------------------------------------------------------------------------------------------------
# add_sound
# ---------------------------------------------------------------------------------------------------------------

_ADD_SOUND_KEYS = {"file", "mode", "name", "frame", "volume", "pitch", "location", "distanceReference",
                   "distanceMax", "attenuation", "coneAngleInner", "coneAngleOuter", "coneVolumeOuter", "channel",
                   "volumeKeys", "scene"}
_SPEAKER_FLOATS = (("distanceReference", "distance_reference"), ("distanceMax", "distance_max"),
                   ("attenuation", "attenuation"), ("coneAngleInner", "cone_angle_inner"),
                   ("coneAngleOuter", "cone_angle_outer"), ("coneVolumeOuter", "cone_volume_outer"))
_SPEAKER_ONLY = ("location", "distanceReference", "distanceMax", "attenuation", "coneAngleInner", "coneAngleOuter",
                 "coneVolumeOuter")
_STRIP_ONLY = ("channel", "volumeKeys", "scene")
_SOUND_MODES = ("speaker", "strip")
_MAX_VOLUME_KEYS = 10000


def _free_channel(ed, first, end, lo, hi):
    """The lowest channel no top-level strip occupies over [first, end), or None if every channel is taken."""
    taken = {}
    for st in _top_strips(ed):
        a, b = _span(st)
        if a < end and first < b:
            taken.setdefault(int(st.channel), st.name)
    for ch in range(int(lo), int(hi) + 1):
        if ch not in taken:
            return ch, taken
    return None, taken


def op_add_sound(params):
    """Put a sound file into the active scene at a frame - as a 3D speaker object, or as a sequencer sound strip.

    params:
      file (str, required)      the sound file; it must exist and decode - read by Blender's own audio module first
      mode (str, required)      speaker - a Speaker object heard in 3D from the scene camera; or strip - a sequencer
                                sound strip, heard as it is
      frame (int)               the frame the sound's first sample plays at; default the current frame
      name (str)                the object or strip name; Blender's own default when omitted
      volume (float)            speaker 0-1, strip 0-100
      pitch (float)             speaker only, 0.1-10 - no build's sound strip has a pitch (3.6-5.2, measured)
      location ([x,y,z])        speaker only; default the origin
      distanceReference, distanceMax, attenuation (float)   speaker only - how its level falls with distance
      coneAngleInner, coneAngleOuter (float, degrees 0-360), coneVolumeOuter (float 0-1)   speaker only - its cone
      channel (int)             strip only, 1-128; default the lowest channel free over the strip's frames
      scene (str)               strip only - the scene whose sequencer gets the strip. Default: on 5.0+ the
                                workspace's sequencer scene when one is set (what the Sequencer shows), else the
                                window's scene; the reply says which (sceneSource). A speaker is always the window's.
      volumeKeys ([[frame, volume], ...])   strip only, instead of volume - keys the strip's volume on the scene's
                                action, LINEAR between keys; Blender samples them once a frame, so a key's value is
                                heard from between half a frame before its frame to one frame after it

    Everything is validated before the first write - the file is decoded, the channel is checked for a strip it would
    overlap - and the sound is loaded with check_existing, so a file already in the scene is reused rather than
    duplicated. The reply reports the name Blender gave (requestedName / nameWasSuffixed), the sound's length in
    seconds and frames, and every value read back. A speaker's reply says whether Blender LOADED its sound
    (soundLoaded): one that is not is silent, and this op does not leave one behind.
    """
    reject_unknown(params, _ADD_SOUND_KEYS, "add_sound", notes={
        "start": "spelled frame - the frame the sound's first sample plays at",
        "frameStart": "spelled frame - the frame the sound's first sample plays at",
        "filepath": "spelled file",
        "path": "spelled file",
        "sound": "pass the sound FILE as `file`; it is loaded with check_existing, so one already loaded is reused",
        "object": ("add_sound creates the speaker. To drive an existing object's property from a sound's loudness, "
                   "use bake_sound_keys"),
        "loop": ("neither a speaker nor a sound strip loops in Blender - place the sound again at a later frame"),
        "pan": "not written by this op - a speaker is panned by where it stands relative to the scene camera",
    })
    tail = "NOTHING was added."
    sc = bpy.context.scene
    try:
        mode = str(take(params, "mode", required=True, kind=str)).strip().lower()
        if mode not in _SOUND_MODES:
            raise MifOpError("mode must be speaker (a 3D Speaker object) or strip (a sequencer sound strip), got %r"
                             % mode)
        # THE SEQUENCER'S SCENE FOR A STRIP (2026-10-08). On 5.0+ the Sequencer shows the workspace's sequencer scene,
        # which need not be the window's - a strip added to bpy.context.scene landed in a sequencer nobody was looking at.
        if mode == "strip":
            sc = _seq_scene(params, tail)
        scene_source = ("named" if params.get("scene") is not None else
                        "sequencerScene" if mode == "strip" and sc is not bpy.context.scene else "window")
        raw_file = take(params, "file", required=True, kind=str)
        name = _clean_name(take(params, "name", default=None, kind=str), tail)
        frame = take_int(params, "frame", default=None)
        volume = take_float(params, "volume", default=None)
        pitch = take_float(params, "pitch", default=None)
    except MifOpError as exc:
        text = str(exc)
        raise MifOpError(text if text.endswith(tail) else "%s %s" % (text.rstrip("."), tail)) from None
    fps = _fps(sc)
    if frame is None:
        frame = int(sc.frame_current)
    lo, hi = _hard_range(sc, "frame_current")
    _in_range(frame, lo, hi, "frame", tail)

    # KEYS OF THE OTHER MODE ARE REFUSED BY NAME, never ignored: a channel on a speaker or a location on a strip is a
    # caller who meant the other mode, and a silent drop would place the sound somewhere they did not ask for. A key
    # sent as null is absent, as take() reads it everywhere else.
    wrong = [k for k in (_STRIP_ONLY if mode == "speaker" else _SPEAKER_ONLY) if params.get(k) is not None]
    if wrong:
        raise MifOpError("%s belong%s to mode %s, and this call is mode %s. %s"
                         % (", ".join(wrong), "s" if len(wrong) == 1 else "",
                            "strip" if mode == "speaker" else "speaker", mode, tail))

    speaker_values, location, channel, keys = {}, None, None, []
    if mode == "speaker":
        spk_rna = bpy.types.Speaker
        if volume is not None:
            _in_range(volume, *_hard_range(spk_rna, "volume"), key="volume", tail=tail)
            speaker_values["volume"] = volume
        if pitch is not None:
            _in_range(pitch, *_hard_range(spk_rna, "pitch"), key="pitch", tail=tail)
            speaker_values["pitch"] = pitch
        for key, attr in _SPEAKER_FLOATS:
            try:
                value = take_float(params, key, default=None)
            except MifOpError as exc:
                raise MifOpError("%s %s" % (str(exc).rstrip("."), tail)) from None
            if value is not None:
                _in_range(value, *_hard_range(spk_rna, attr), key=key, tail=tail)
                speaker_values[attr] = value
        inner = speaker_values.get("cone_angle_inner", spk_rna.bl_rna.properties["cone_angle_inner"].default)
        outer = speaker_values.get("cone_angle_outer", spk_rna.bl_rna.properties["cone_angle_outer"].default)
        if inner > outer:
            # Blender stores the pair either way round (measured: inner 120 beside outer 90 reads back as set), but
            # its own tooltips define the inner cone as the full-volume one INSIDE the outer - a contradiction.
            raise MifOpError("coneAngleInner (%g) is wider than coneAngleOuter (%g) - the inner cone is the "
                             "full-volume one inside the outer cone. %s" % (inner, outer, tail))
        location = _vec3(params, "location", (0.0, 0.0, 0.0), tail)
        mode_now = getattr(bpy.context, "mode", "OBJECT")
        if mode_now != "OBJECT":
            raise MifOpError("Blender is in %s mode, and the speaker is made by an object operator that runs only in "
                             "OBJECT mode. Leave %s mode first. %s" % (mode_now, mode_now, tail))
    else:
        strip_rna = _strip_rna()
        if pitch is not None and (strip_rna is None or "pitch" not in strip_rna.bl_rna.properties):
            raise MifOpError("a sequencer sound strip has no pitch on Blender %s (absent on every build from 3.6 to "
                             "5.2, measured) - use mode speaker, whose pitch is a real property, or retime the strip "
                             "in the sequencer. %s" % (bpy.app.version_string, tail))
        if volume is not None:
            _in_range(volume, *_hard_range(strip_rna, "volume"), key="volume", tail=tail)
        try:
            channel = take_int(params, "channel", default=None)
        except MifOpError as exc:
            raise MifOpError("%s %s" % (str(exc).rstrip("."), tail)) from None
        ch_lo, ch_hi = _hard_range(strip_rna, "channel")
        if channel is not None:
            _in_range(channel, ch_lo, ch_hi, "channel", tail)
        raw_keys = params.get("volumeKeys")
        if raw_keys is not None and volume is not None:
            raise MifOpError("pass volume OR volumeKeys - once a strip's volume is keyed, the keys set it on every "
                             "frame and a static volume is never heard. %s" % tail)
        if raw_keys is not None:
            if not isinstance(raw_keys, (list, tuple)) or not raw_keys:
                raise MifOpError("volumeKeys takes a non-empty list of [frame, volume] pairs, got %r. %s"
                                 % (raw_keys, tail))
            if len(raw_keys) > _MAX_VOLUME_KEYS:
                raise MifOpError("volumeKeys holds %d pairs; the limit is %d. %s"
                                 % (len(raw_keys), _MAX_VOLUME_KEYS, tail))
            vlo, vhi = _hard_range(strip_rna, "volume")
            seen = set()
            for i, pair in enumerate(raw_keys):
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    raise MifOpError("volumeKeys[%d] must be [frame, volume], got %r. %s" % (i, pair, tail))
                try:
                    kf, kv = finite_floats(pair, "volumeKeys[%d]" % i)
                except MifOpError as exc:
                    raise MifOpError("%s %s" % (str(exc).rstrip("."), tail)) from None
                _in_range(kf, lo, hi, "volumeKeys[%d] frame" % i, tail)
                _in_range(kv, vlo, vhi, "volumeKeys[%d] volume" % i, tail)
                if kf in seen:
                    raise MifOpError("volumeKeys has two keys on frame %g - a frame holds one volume. %s" % (kf, tail))
                seen.add(kf)
                keys.append((kf, kv))
            keys.sort()

    # THE FILE, DECODED BEFORE ANYTHING EXISTS. See _read_sound for why loading it is not a check.
    path, rate, channels, samples = _read_sound(raw_file, tail)
    seconds = samples / rate
    # round(seconds x fps), at least one - how every build turns a sound's length into a strip's frames (measured at
    # 24, 25 and 29.97 fps: 5.52 frames -> 6, 6.24 -> 6, 4.995 -> 5, 0.0005 -> 1).
    length_frames = max(1, _round_half_up(seconds * fps))

    ed = sc.sequence_editor
    if mode == "strip":
        # A STRIP ON TOP OF ANOTHER, refused before it exists. new_sound puts it on the channel asked for even when a
        # strip is already there (measured on every build: no shuffle), and overlapping strips on one channel are the
        # state the sequencer shows in red.
        first, end = frame, frame + length_frames
        if ed is not None:
            free, taken = _free_channel(ed, first, end, ch_lo, ch_hi)
            if channel is not None and channel in taken:
                raise MifOpError("channel %d already holds strip '%s' over frames %d-%d, which this %d-frame sound "
                                 "would overlap - %s. %s"
                                 % (channel, taken[channel], first, end - 1, length_frames,
                                    ("channel %d is free there" % free) if free is not None
                                    else "every channel is taken there", tail))
            if channel is None:
                if free is None:
                    raise MifOpError("every channel from %d to %d holds a strip over frames %d-%d. %s"
                                     % (ch_lo, ch_hi, first, end - 1, tail))
                channel = free
        elif channel is None:
            channel = int(ch_lo)

    # ------------------------------------------------------------------ the writes
    sounds_before = {s.as_pointer() for s in bpy.data.sounds}
    try:
        snd = bpy.data.sounds.load(path, check_existing=True)
    except RuntimeError as exc:
        raise MifOpError("Blender would not load '%s' as a sound: %s. %s" % (path, exc, tail)) from None
    made_sound = snd.as_pointer() not in sounds_before
    sound_row = {"name": snd.name, "filepath": path, "sampleRate": rate, "channels": channels,
                 "durationSeconds": round(seconds, 6), "durationFrames": length_frames, "fps": round(fps, 6),
                 "reusedExisting": not made_sound}

    if mode == "speaker":
        out = _add_speaker(sc, snd, made_sound, frame, length_frames, name, location, speaker_values)
    else:
        out = _add_strip(sc, snd, made_sound, ed is None, path, frame, name, channel, volume, keys)
    out.update({"mode": mode, "scene": sc.name, "sceneSource": scene_source, "frame": frame, "sound": sound_row})
    return out


def _forget_sound(snd, made_sound):
    """Remove a sound datablock this call loaded, once nothing uses it."""
    if made_sound and snd is not None and snd.users == 0:
        bpy.data.sounds.remove(snd)


def _add_speaker(sc, snd, made_sound, frame, length_frames, name, location, values):
    """The speaker half of add_sound: operator, name, sound, settings, relations rebuild, and the loaded-sound check."""
    snap = selection_snapshot()
    before = set(bpy.data.objects.keys())
    old_frame = sc.frame_current
    obj = None
    try:
        # THE OPERATOR, because it is the only thing that makes a speaker's NLA SOUND strip - and it makes it at the
        # scene's current frame. frame_current is the property, not frame_set(): nothing needs evaluating here.
        sc.frame_current = frame
        try:
            bpy.ops.object.speaker_add(enter_editmode=False, align="WORLD", location=location,
                                       rotation=(0.0, 0.0, 0.0))
        except (RuntimeError, TypeError) as exc:
            _forget_sound(snd, made_sound)
            raise MifOpError("Blender's speaker operator refused: %s. Nothing was left behind." % exc) from None
    finally:
        sc.frame_current = old_frame
        selection_restore(snap)
    made = [n for n in bpy.data.objects.keys() if n not in before]
    if len(made) != 1 or bpy.data.objects[made[0]].type != "SPEAKER":
        for n in made:
            remove_objects([bpy.data.objects[n]])
        _forget_sound(snd, made_sound)
        raise MifOpError("expected the speaker operator to make exactly one SPEAKER object, got %d (%s); they were "
                         "removed again." % (len(made), ", ".join(made)))
    obj = bpy.data.objects[made[0]]
    spk = obj.data
    if name:
        # THROUGH assign_id_name, because 3.6 displaces an incumbent instead of suffixing the new block (ledger B12).
        assign_id_name(obj, name, bpy.data.objects)
    assign_id_name(spk, obj.name, bpy.data.speakers)
    spk.sound = snd
    for attr, value in values.items():
        setattr(spk, attr, value)
    # THE RELATIONS REBUILD. Re-assigning the object's data rebuilds the dependency graph's relations, so the sound
    # gets a node and is LOADED; the tag re-copies the speaker so its evaluated copy points at that loaded sound.
    # Without both the speaker is silent in every mixdown - see the module docstring. Renaming the speaker's DATA loads
    # the sound as well (renaming only the object does not), so a named speaker, whose data is renamed above, is loaded
    # either way; an unnamed one, whose data already carries the object's name, is loaded only by this - measured on
    # 4.4 by deleting these two lines, which only test_blender_audio BA102's unnamed speaker then caught.
    obj.data = spk
    spk.update_tag()

    ad = obj.animation_data
    strips = [s for t in (ad.nla_tracks if ad else ()) for s in t.strips if s.type == "SOUND"]
    for s in strips:
        # THE STRIP'S DISPLAYED LENGTH - Blender's own operator makes it ceil-free: start + round(seconds x fps) when
        # the speaker already has a sound, start + 10 when it does not, which is the state speaker_add leaves it in.
        # Playback ignores the end either way (sound.cc adds the sound with no end), so this is for the NLA editor.
        s.frame_end = float(frame + length_frames)
    ev = _evaluated_sound(obj)
    if not strips or ev is None or not ev.is_evaluated:
        data = obj.data
        remove_objects([obj])
        if data is not None and data.users == 0:
            bpy.data.speakers.remove(data)
        _forget_sound(snd, made_sound)
        raise MifOpError("the speaker was made but would be SILENT - %s. It was removed again."
                         % ("the operator made no NLA sound strip" if not strips
                            else "Blender did not load its sound for playback"))
    out = _speaker_readback(obj)
    out.update({
        "name": obj.name,
        "requestedName": name,
        "nameWasSuffixed": bool(name and obj.name != name),
        "frameEnd": int(frame + length_frames - 1),
    })
    if name and obj.name != name:
        out["nameNote"] = ("Blender named this '%s' because '%s' was taken - it suffixes rather than failing."
                           % (obj.name, name))
    if sc.camera is None:
        out["listenerNote"] = ("this scene has no camera, and Blender puts the 3D-audio listener at the scene camera - "
                               "a speaker is heard relative to it.")
    return out


def _add_strip(sc, snd, made_sound, made_editor, path, frame, name, channel, volume, keys):
    """The strip half of add_sound: new_sound, the sound pointer, volume, and the volume keys."""
    ed = sc.sequence_editor_create()
    coll = _top_strips(ed)
    wanted = name or os.path.basename(path)
    try:
        st = coll.new_sound(wanted, path, channel, frame)
    except (RuntimeError, TypeError) as exc:
        _forget_sound(snd, made_sound)
        # THE EMPTY EDITOR STAYS (2026-10-08). Scene.sequence_editor_clear() followed by any change of the scene's frame
        # rate crashes Blender in its sequencer evaluation (blender::seq::strip_for_each_recursive, measured on 4.4.0,
        # 5.0.1 and 5.2.1 - even right after sequence_editor_create in the same call), so a refusal leaves the new
        # editor in place, empty, as every 4.4+ default scene already has one.
        raise MifOpError("Blender would not make the sound strip: %s. Nothing was left behind." % exc) from None
    # ONE SOUND DATABLOCK PER FILE. new_sound loads its own copy on 3.6-5.0 ("tone.wav.001") and reuses the loaded one
    # on 5.2 (measured), so the strip is pointed at the datablock loaded above and the copy it made is dropped.
    dup = st.sound
    if dup is not None and dup != snd:
        st.sound = snd
        if dup.users == 0:
            bpy.data.sounds.remove(dup)
    if volume is not None:
        st.volume = volume
    key_rows, interps = [], []
    if keys:
        for kf, kv in keys:
            st.volume = kv
            st.keyframe_insert("volume", frame=kf)
        path_id = st.path_from_id("volume")
        curve = next((fc for fc in _fcurves(sc) if fc.data_path == path_id), None)
        if curve is not None:
            # LINEAR between the given keys - a level list is automation, and a Bezier curve would ease each change in
            # and out. The mixer still samples the curve once a frame (see the module docstring for what that costs).
            for kp in curve.keyframe_points:
                kp.interpolation = "LINEAR"
            curve.update()
            key_rows = [[round(float(kp.co[0]), 4), round(float(kp.co[1]), 6)] for kp in curve.keyframe_points]
            # READ BACK off the keys, not restated from the loop above.
            interps = sorted({kp.interpolation for kp in curve.keyframe_points})
    first, end = _span(st)
    if first != frame or int(st.channel) != channel:
        raise MifOpError("Blender placed strip '%s' at frame %d on channel %d, not frame %d on channel %d as asked - "
                         "the strip was kept; move it or remove it." % (st.name, first, st.channel, frame, channel))
    out = {
        "name": st.name,
        "strip": st.name,
        "requestedName": name,
        "nameWasSuffixed": bool(name and st.name != name),
        "channel": int(st.channel),
        "frameStart": first,
        "frameEnd": end - 1,
        "durationFrames": end - first,
        # THE STATIC VOLUME ONLY WHEN NOTHING KEYS IT - with keys it is whatever the last key wrote, and the keys, not
        # it, set the level on every frame.
        "volume": None if key_rows else round(float(st.volume), 6),
        "soundDatablock": st.sound.name if st.sound else None,
        "volumeKeys": key_rows,
        "volumeKeysInterpolation": "/".join(interps) if interps else None,
    }
    if keys and len(key_rows) != len(keys):
        out["volumeKeysNote"] = ("%d volume key(s) were asked for and the strip's curve holds %d - the difference is "
                                 "keys that were already on it." % (len(keys), len(key_rows)))
    if name and st.name != name:
        out["nameNote"] = ("Blender named this strip '%s' because '%s' was taken in this scene's sequencer."
                           % (st.name, name))
    return out


# ---------------------------------------------------------------------------------------------------------------
# mixdown_audio
# ---------------------------------------------------------------------------------------------------------------

_MIXDOWN_KEYS = {"filepath", "container", "sampleRate", "bitFormat", "frameStart", "frameEnd", "channels", "scene"}
# WHERE A LAYOUT'S SOUND LANDS, measured 2026-09-23 on 3.6.23, 4.2.17, 4.4.0, 5.0.1 and 5.2.1 - one mono tone strip at
# each pan, mixed down in every layout the build offers, each channel's level read out of the file. Identical on every
# build. MONO, STEREO, STEREO_LFE, SURROUND4, SURROUND5 and SURROUND51 put a source where their WAV channel mask says.
# SURROUND61 and SURROUND71 do not, and the reply says so rather than hand over a file a player will misplace.
_LAYOUT_WARNINGS = {
    "SURROUND61": ("a source behind the listener (strip pan 2 or -2) lands wholly in channel 5, which this file's "
                   "channel mask (0x13f) names back left; hard left (pan -1) splits between front left and channel 6 "
                   "(mask: back right), hard right between front right and channel 7 (mask: back center). A player "
                   "that honors the mask misplaces them - measured on 4.4.0, 5.0.1 and 5.2.1"),
    "SURROUND71": ("a source behind the listener (strip pan 2 or -2) lands equally in channels 7 and 8, which this "
                   "file's channel mask (0x63f) names side left and side right; hard left (pan -1) splits between "
                   "front left and channel 5, which the mask names back left. A player that honors the mask plays the "
                   "back pair from the sides and the sides from the back - measured on 3.6.23 to 5.2.1"),
}
_CONTAINER_CODEC = {"WAV": "PCM", "FLAC": "FLAC"}
_CONTAINER_EXT = {"WAV": ".wav", "FLAC": ".flac"}
# WHAT FLAC WRITES FOR A FORMAT IT CANNOT HOLD, measured from the file's STREAMINFO on all five builds: U8 comes out
# 16-bit, and S32, F32 and F64 come out 24-bit. S16 and S24 are written as asked.
_FLAC_REWRITES = {"U8": 16, "S32": 24, "F32": 24, "F64": 24}
_MIX_START_WAIT = 3.0
# HOW THIS SESSION'S MIXDOWNS HAVE RUN: written by a job after the operator returned, or before it returned. Nothing
# in RNA says which a build does, so it is measured on each call and remembered to explain a job that never starts.
_MIX_STATE = {"jobs": 0, "inline": 0}


def _complete(reader, path):
    """True once the writer has finished the file's header - False while it is absent or still being written."""
    try:
        return os.path.isfile(path) and reader(path)[0]
    except OSError:
        return False


def _wav_header(path):
    """(complete, info) off a WAV file's RIFF header - complete once the writer has filled the sizes in."""
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        head = fh.read(12)
        if len(head) < 12 or head[:4] != b"RIFF" or head[8:12] != b"WAVE":
            return False, None
        riff = struct.unpack("<I", head[4:8])[0]
        info = {}
        while True:
            hdr = fh.read(8)
            if len(hdr) < 8:
                return False, None
            cid, csz = hdr[:4], struct.unpack("<I", hdr[4:])[0]
            if cid == b"fmt ":
                body = fh.read(csz)
                tag = struct.unpack("<H", body[0:2])[0]
                if tag == 0xFFFE and csz >= 26:
                    info["channelMask"] = "0x%x" % struct.unpack("<I", body[20:24])[0]
                    tag = struct.unpack("<H", body[24:26])[0]
                info.update(channels=struct.unpack("<H", body[2:4])[0], rate=struct.unpack("<I", body[4:8])[0],
                            bits=struct.unpack("<H", body[14:16])[0], sampleFormat="float" if tag == 3 else "int")
            elif cid == b"data":
                if not info or csz in (0, 0xFFFFFFFF):
                    return False, None
                info["samples"] = csz // max(1, info["channels"] * info["bits"] // 8)
                return riff == size - 8, info
            else:
                fh.seek(csz + (csz & 1), 1)


def _flac_header(path):
    """(complete, info) off a FLAC file's STREAMINFO - complete once the writer has filled the sample count in."""
    with open(path, "rb") as fh:
        data = fh.read(42)
    if len(data) < 42 or data[:4] != b"fLaC":
        return False, None
    si = data[8:42]
    total = ((si[13] & 0xF) << 32) | struct.unpack(">I", si[14:18])[0]
    info = {"rate": (si[10] << 12) | (si[11] << 4) | (si[12] >> 4), "channels": ((si[12] >> 1) & 0x7) + 1,
            "bits": (((si[12] & 0x1) << 4) | (si[13] >> 4)) + 1, "samples": total, "sampleFormat": "int"}
    return total > 0, info


def _scene_sounds(sc):
    """Absolute paths of every sound the scene plays - strips and speakers."""
    out = set()
    snds = []
    if sc.sequence_editor is not None:
        snds += [getattr(s, "sound", None) for s in _all_strips(sc.sequence_editor)]
    snds += [o.data.sound for o in sc.objects if o.type == "SPEAKER"]
    for snd in snds:
        if snd is not None and snd.filepath:
            out.add(os.path.normcase(os.path.normpath(bpy.path.abspath(snd.filepath, library=snd.library))))
    return out


def op_mixdown_audio(params):
    """Render the active scene's audio - every sequencer sound strip and every speaker - to a WAV or FLAC file, and
    measure the file that was written.

    params:
      filepath (str, required)   the file to write; its extension must match the container
      container (str)            WAV or FLAC; default taken from the extension
      sampleRate (int)           default 48000
      bitFormat (str)            U8, S16, S24, S32, F32 or F64 for WAV; S16 or S24 for FLAC (default S16)
      frameStart / frameEnd (int)  default the scene's range
      scene (str)                the scene to mix. Default: on 5.0+ the workspace's sequencer scene when one is set
                                 (the edit the Sequencer shows), else the window's scene; the reply names it and says
                                 which (sceneSource). The mixdown runs with that scene as the context's.

    channels (str)             STEREO by default, or any layout this build's mixdown offers - MONO, STEREO,
                               SURROUND4, SURROUND51, SURROUND71 on every build; STEREO_LFE, SURROUND5 and SURROUND61
                               from 4.4. The reply names the WAV's channel mask; for SURROUND61 and SURROUND71 it warns
                               that the mask misnames where the sound is (measured: _LAYOUT_WARNINGS).

    Sample 0 of the file is frameStart, so a sound placed at frame F starts at sample
    (F - frameStart) x rate / fps. The reply is measured off the file: wroteFile, fileBytes, bitsPerSample,
    sampleRate, channels and durationSeconds come from its header, not from the request. A WAV runs a few hundred
    samples past the range; a FLAC does not.

    Blender steps through every frame of the range first, to capture animated volume and moving speakers, so a heavy
    scene mixes slowly. It writes a temporary file beside the target and renames it over the target once complete,
    so a failure never leaves half a file where the old one was. A speaker Blender has not loaded its sound for
    (its sound set after the dependency graph was built) is reloaded first and named in speakersReloaded; a speaker
    that still cannot play is named in silentSpeakers with the reason.
    On Blender 5.2 the mixdown runs as a job; a background Blender (-b) runs one mixdown job per session and never
    starts another, which is refused with that reason.
    """
    reject_unknown(params, _MIXDOWN_KEYS, "mixdown_audio", notes={
        "path": "spelled filepath",
        "output": "spelled filepath",
        "codec": "follows the container: WAV is written as PCM, FLAC as FLAC",
        "format": "spelled bitFormat",
        "rate": "spelled sampleRate",
        "normalize": ("this op writes the mix as Blender renders it; loudness is measured outside Blender, by the "
                      "MCP server's mif_measure_audio"),
    })
    tail = "NOTHING was written."
    sc = _seq_scene(params, tail)
    scene_source = ("named" if params.get("scene") is not None else
                    "sequencerScene" if sc is not bpy.context.scene else "window")
    try:
        raw = take(params, "filepath", required=True, kind=str)
        container = take(params, "container", default=None, kind=str)
        bit_format = take(params, "bitFormat", default="S16", kind=str)
        rate = take_int(params, "sampleRate", default=48000)
        layout = take(params, "channels", default="STEREO", kind=str)
        start = take_int(params, "frameStart", default=None)
        end = take_int(params, "frameEnd", default=None)
    except MifOpError as exc:
        raise MifOpError("%s %s" % (str(exc).rstrip("."), tail)) from None
    path = bpy.path.abspath(raw)
    check_output_path(raw, path, "written")
    if os.path.isdir(path):
        # REFUSED HERE, not discovered at the rename after the mix - on 5.2 that mix is the session's only one.
        raise MifOpError("'%s' is a directory - pass the path of the file to write inside it. %s" % (path, tail))
    ext = os.path.splitext(path)[1].lower()
    if container is None:
        container = next((c for c, e in _CONTAINER_EXT.items() if e == ext), None)
        if container is None:
            raise MifOpError("'%s' ends in neither .wav nor .flac, and no container was given - name the file .wav or "
                             ".flac. %s" % (path, tail))
    container = container.strip().upper()
    op_rna = bpy.ops.sound.mixdown.get_rna_type().properties
    offered = [i.identifier for i in op_rna["container"].enum_items]
    if container not in _CONTAINER_CODEC:
        raise MifOpError("container must be WAV or FLAC - the lossless formats this op writes and reads back - got "
                         "'%s'. This Blender's mixdown also offers %s. %s"
                         % (container, ", ".join(c for c in offered if c not in _CONTAINER_CODEC), tail))
    codecs = [i.identifier for i in op_rna["codec"].enum_items]
    if container not in offered or _CONTAINER_CODEC[container] not in codecs:
        raise MifOpError("this Blender's mixdown (%s) does not offer container %s with codec %s. Containers: %s. %s"
                         % (bpy.app.version_string, container, _CONTAINER_CODEC[container], ", ".join(offered), tail))
    if ext != _CONTAINER_EXT[container]:
        raise MifOpError("'%s' ends in '%s' but the container is %s - Blender writes the container under whatever name "
                         "it is given, so the file would lie about its format. Name it %s. %s"
                         % (path, ext, container, _CONTAINER_EXT[container], tail))
    formats = [i.identifier for i in op_rna["format"].enum_items]
    bit_format = bit_format.strip().upper()
    if bit_format not in formats:
        raise MifOpError("bitFormat '%s' is not one this Blender offers. Valid: %s. %s"
                         % (bit_format, ", ".join(formats), tail))
    if container == "FLAC" and bit_format in _FLAC_REWRITES:
        raise MifOpError("FLAC holds 16- and 24-bit integers - Blender writes a %s FLAC as %d-bit without a word "
                         "(measured on every build from 3.6 to 5.2). Use S16 or S24, or a WAV. %s"
                         % (bit_format, _FLAC_REWRITES[bit_format], tail))
    # THE LAYOUT'S HOME MOVED WITH THE RATE'S: 4.4+ take it as the operator's `channels` (eight layouts), 3.6 and 4.2 mix in
    # the scene's FFmpeg audio_channels (five). Each build's own list decides.
    has_channels = "channels" in op_rna
    layouts = [i.identifier for i in (op_rna["channels"] if has_channels
                                      else sc.render.ffmpeg.bl_rna.properties["audio_channels"]).enum_items]
    layout = layout.strip().upper()
    if layout not in layouts:
        later = " %s is in 4.4's mixdown and later." % layout if layout in ("STEREO_LFE", "SURROUND5", "SURROUND61") else ""
        raise MifOpError("channels '%s' is not a layout this Blender (%s) mixes down to. Valid: %s.%s %s"
                         % (layout, bpy.app.version_string, ", ".join(layouts), later, tail))
    # THE SAMPLE RATE'S HOME MOVED. 4.4+ take it as the operator's `mixrate`; 3.6 and 4.2 have no such property and
    # mix at the scene's FFmpeg audio rate (measured: 22050 there gives a 22050 Hz file). Same range on both.
    has_rate = "mixrate" in op_rna
    ff = sc.render.ffmpeg
    rate_rna = op_rna["mixrate"] if has_rate else ff.bl_rna.properties["audio_mixrate"]
    _in_range(rate, rate_rna.hard_min, rate_rna.hard_max, "sampleRate", tail)
    start = int(sc.frame_start) if start is None else start
    end = int(sc.frame_end) if end is None else end
    _in_range(start, *_hard_range(sc, "frame_start"), key="frameStart", tail=tail)
    _in_range(end, *_hard_range(sc, "frame_end"), key="frameEnd", tail=tail)
    if end < start:
        raise MifOpError("frameEnd (%d) is before frameStart (%d). %s" % (end, start, tail))
    # A SOURCE OVERWRITTEN WHILE IT IS READ. Mixing down onto a file a strip or speaker plays is measured to leave the
    # scene's sound state broken for later mixdowns in the session - and it destroys the source.
    if os.path.normcase(os.path.normpath(path)) in _scene_sounds(sc):
        raise MifOpError("'%s' is a sound this scene plays - mixing down onto it would overwrite a source while "
                         "Blender reads it. Write somewhere else. %s" % (path, tail))
    parent = os.path.dirname(path) or "."
    try:
        check_writable_dir(parent)
    except OSError as exc:
        raise MifOpError("cannot write in '%s' (%s). %s" % (parent, exc, tail)) from None

    # ------------------------------------------------------------------ speakers, before anything is mixed
    reloaded, silent = [], []
    for obj in [o for o in sc.objects if o.type == "SPEAKER"]:
        plays, cause, why = _speaker_plays(obj)
        if cause == "not_loaded":
            # THE REPAIR add_sound applies to its own speakers: a relations rebuild and a re-copy (see the docstring).
            obj.data = obj.data
            obj.data.update_tag()
            reloaded.append(obj.name)
            plays, cause, why = _speaker_plays(obj)
        if not plays and cause != "no_sound":
            silent.append({"speaker": obj.name, "reason": why})
    strips = [s for s in (_all_strips(sc.sequence_editor) if sc.sequence_editor is not None else ())
              if s.type == "SOUND" and getattr(s, "sound", None) is not None and not s.mute]
    speakers = [o.name for o in sc.objects if o.type == "SPEAKER" and _speaker_plays(o)[0]]

    fps = _fps(sc)
    tmp = os.path.join(parent, ".mif_mixdown_%s%s" % (uuid.uuid4().hex[:12], ext))
    replaced = os.path.isfile(path)
    old = (int(sc.frame_start), int(sc.frame_end))
    old_ff = (ff.audio_mixrate, ff.audio_channels) if not (has_rate and has_channels) else None
    args = {"filepath": tmp, "container": container, "codec": _CONTAINER_CODEC[container], "format": bit_format}
    if has_rate:
        args["mixrate"] = rate
    if has_channels:
        args["channels"] = layout
    reader = _wav_header if container == "WAV" else _flac_header
    requested = (end - start + 1) / fps
    t0 = time.time()
    job, info, error, late = None, None, None, False
    try:
        # WIDEN FIRST, THEN NARROW, so the frame range never clamps against itself on the way (set_frame_range's rule).
        if end >= sc.frame_end:
            sc.frame_end, sc.frame_start = end, start
        else:
            sc.frame_start, sc.frame_end = start, end
        if old_ff is not None:
            if not has_rate:
                ff.audio_mixrate = rate
            if not has_channels:
                ff.audio_channels = layout
        try:
            # THROUGH ops_undo's window override: a GUI Blender runs this op from an app timer, where there is no
            # window in the context, and 5.2 hands the job to one. -b has a window, so there it is a plain call.
            _mix_in_scene(sc, args)
        except RuntimeError as exc:
            error = "Blender's mixdown refused: %s" % str(exc).strip()
        returned = time.time()
        if error is None:
            done_at_return = _complete(reader, tmp)
            # WAITED FOR, NOT TRUSTED. On 5.2 the file appears ~20 ms after FINISHED; a job that never started never
            # makes one at all. Completion is the header the writer fills in last, not the size standing still.
            deadline_start = returned + _MIX_START_WAIT
            deadline_done = returned + 60.0 + 2.0 * requested
            while not done_at_return:
                exists = os.path.isfile(tmp)
                if exists and _complete(reader, tmp):
                    break
                now = time.time()
                if (not exists and now > deadline_start) or now > deadline_done:
                    break
                time.sleep(0.02)
            if _complete(reader, tmp):
                info = reader(tmp)[1]
                # MEASURED, NOT ASSUMED FROM THE VERSION: a file that was not finished when the operator returned was
                # written by a job. Remembered, because it is what explains a later mixdown that never starts.
                job = not done_at_return
                _MIX_STATE["jobs" if job else "inline"] += 1
            elif not os.path.isfile(tmp):
                if _MIX_STATE["jobs"] and bpy.app.background:
                    why = ("this Blender mixes down in a job, and a background Blender (-b) retires a finished job "
                           "only through the UI's job timer, which it never runs - so after the first mixdown of a "
                           "session (%d have run here) every later one is suspended and never writes (measured on "
                           "5.2.1). Mix down in a fresh Blender, or in one with a window" % _MIX_STATE["jobs"])
                elif _MIX_STATE["inline"]:
                    why = ("earlier mixdowns in this session wrote before returning, so this is the writer failing "
                           "without a message")
                else:
                    # NOT RULED OUT HERE: a job that starts after this op has given up. A background Blender never
                    # starts a suspended job, but one with a window starts it once the job ahead of it is retired, and
                    # it then writes the temp file. Reasoned from how window-manager jobs resume, not measured - this
                    # op's tests run headless - so the refusal below says it may happen rather than that it cannot.
                    late = True
                    why = ("either Blender's writer failed without a message, or - on a build that mixes down in a "
                           "job, 5.2 among them - the job is waiting behind an earlier one that has not been retired")
                error = ("Blender answered FINISHED and no file appeared within %.0f s: %s"
                         % (_MIX_START_WAIT, why))
            else:
                error = "the mixdown did not finish writing within %.0f s" % (deadline_done - returned)
    finally:
        if sc.frame_start != old[0] or sc.frame_end != old[1]:
            if old[1] >= sc.frame_end:
                sc.frame_end, sc.frame_start = old[1], old[0]
            else:
                sc.frame_start, sc.frame_end = old[0], old[1]
        if old_ff is not None:
            ff.audio_mixrate, ff.audio_channels = old_ff
    if error is not None:
        try:
            if os.path.isfile(tmp):
                os.remove(tmp)
        except OSError:
            pass
        # "NOTHING was written" ONLY WHEN IT IS TRUE. A temp file the writer still holds open (Windows will not delete
        # it) is named instead, and so is the one a late-starting job may still write.
        if os.path.isfile(tmp):
            raise MifOpError("%s. '%s' is unchanged, but the mixdown still holds its unfinished '%s' - delete it once "
                             "Blender lets go of it." % (error.rstrip("."), path, tmp))
        if late:
            raise MifOpError("%s. '%s' is unchanged. If Blender starts this mixdown later, it writes '%s' beside it - "
                             "safe to delete." % (error.rstrip("."), path, tmp))
        raise MifOpError("%s. '%s' is unchanged. NOTHING was written." % (error.rstrip("."), path))
    # THE RENAME, retried while the writer's handle closes - Windows will not replace a file still open for writing.
    for attempt in range(100):
        try:
            os.replace(tmp, path)
            break
        except PermissionError:
            if attempt == 99:
                raise MifOpError("the mix was written to '%s' and could not be renamed over '%s' - something holds "
                                 "the target open." % (tmp, path)) from None
            time.sleep(0.02)
    size = os.path.getsize(path)
    out = {
        "filepath": path,
        "wroteFile": size > 0,
        "fileBytes": size,
        "replacedExisting": replaced,
        "container": container,
        "codec": _CONTAINER_CODEC[container],
        "bitFormat": bit_format,
        "bitsPerSample": info.get("bits"),
        "sampleFormat": info.get("sampleFormat"),
        "sampleRate": info.get("rate"),
        "channels": info.get("channels"),
        "layout": layout,
        "channelMask": info.get("channelMask"),
        "samples": info.get("samples"),
        "durationSeconds": round(info["samples"] / float(info["rate"]), 6) if info.get("rate") else None,
        "requestedSeconds": round(requested, 6),
        "frameStart": start,
        "frameEnd": end,
        "fps": round(fps, 6),
        "scene": sc.name,
        "sceneSource": scene_source,
        "ranAsJob": job,
        "elapsedSeconds": round(time.time() - t0, 3),
        "soundStrips": len(strips),
        "speakers": speakers,
        "speakersReloaded": reloaded,
        "silentSpeakers": silent,
        "sizeNote": "every number here is read off the written file's header, not taken from the request.",
    }
    if layout in _LAYOUT_WARNINGS:
        out["layoutWarning"] = _LAYOUT_WARNINGS[layout]
    if not strips and not speakers:
        out["silentNote"] = ("the scene has no unmuted sound strip and no speaker that plays, so the file holds "
                             "silence.")
    if _scene_muted(sc):
        out["mutedNote"] = ("the scene's audio is muted (Scene.use_audio), so the file holds silence - measured on "
                            "every build.")
    return out


def _mix_in_scene(sc, args):
    """sound.mixdown with sc as the context's scene - the operator mixes CTX_data_scene, and a temp_override of scene
    reaches it (measured 2026-10-08 on 4.2.17, 4.4.0 and 5.2.1: a second scene's strip mixed while the window showed
    another)."""
    if sc is bpy.context.scene or not hasattr(bpy.context, "temp_override"):
        return _run_with_window(bpy.ops.sound.mixdown, **args)
    from .ops_undo import _window_override   # noqa: PLC0415
    over = dict(_window_override() or {})
    over["scene"] = sc
    with bpy.context.temp_override(**over):
        return bpy.ops.sound.mixdown(**args)


def _scene_muted(sc):
    """Is the scene's audio muted? Scene.use_audio MEANS OPPOSITE THINGS BY BUILD: on 3.6 it is "Audio Muted" (True
    mutes, default False) and from 4.2 "Play Audio" (False mutes, default True) - same identifier, measured by mixdown
    on 3.6.23, 4.2.17 and 5.2.1. So the property's own RNA name decides which way to read it, not a version number."""
    name = sc.bl_rna.properties["use_audio"].name.lower()
    return bool(sc.use_audio) if "mute" in name else not sc.use_audio


# ---------------------------------------------------------------------------------------------------------------
# bake_sound_keys
# ---------------------------------------------------------------------------------------------------------------

_BAKE_KEYS = {"file", "object", "dataPath", "index", "frameStart", "minValue", "maxValue", "attackMs", "releaseMs",
              "mode"}
_BAKE_MODES = ("rms", "peak")
# A GUARD ON MEMORY, not on length for its own sake: the samples are read as 32-bit floats, so this is ~800 MB -
# about 35 minutes of stereo at 48 kHz.
_BAKE_MAX_VALUES = 200000000


def _bake_target(obj, path, tail):
    """(owner ID, path from it, the struct holding the property, property name) for a path given from the object."""
    holder, leaf = obj, path
    if "." in path:
        head, leaf = path.rsplit(".", 1)
        try:
            holder = obj.path_resolve(head)
        except (ValueError, AttributeError, TypeError) as exc:
            raise MifOpError("could not resolve '%s' on '%s' (%s) - give the path from the object, e.g. scale, "
                             "location or data.energy. %s" % (head, obj.name, exc, tail)) from None
        if holder is None or not hasattr(holder, "bl_rna"):
            raise MifOpError("'%s' on '%s' is %r, not something with properties. %s" % (head, obj.name, holder, tail))
    owner = getattr(holder, "id_data", None) or obj
    prop = holder.bl_rna.properties.get(leaf)
    if prop is None:
        numeric = sorted(p.identifier for p in holder.bl_rna.properties
                         if p.type in ("FLOAT", "INT") and p.is_animatable and not p.is_readonly)
        raise MifOpError("'%s' has no property '%s'. Its animatable numbers: %s. %s"
                         % (holder.bl_rna.identifier, leaf, ", ".join(numeric[:30]) or "(none)", tail))
    # KEYED ON THE DATABLOCK THAT OWNS IT. Blender keys a property only on its own ID - data.energy lives on the Light,
    # not the object - so the path is re-rooted there with Blender's own path_from_id.
    try:
        owner_path = holder.path_from_id(leaf)
    except (ValueError, TypeError) as exc:
        raise MifOpError("Blender cannot name a path to '%s' from the datablock that owns it (%s), so it cannot be "
                         "keyed. %s" % (path, exc, tail)) from None
    return owner, owner_path, holder, prop


def _envelope(data, rate, fps, mode, np):
    """One level per 1/fps slice of the samples: RMS over every channel, or the peak. data is (samples, channels)."""
    n = int(data.shape[0])
    frames = int(math.ceil(n * fps / rate))
    # FLOOR, so every slice but possibly the last holds floor or ceil of rate / fps samples and none is empty.
    edges = np.floor(np.arange(frames + 1) * (rate / fps)).astype(np.int64)
    edges[-1] = n
    starts = edges[:-1]
    counts = np.diff(edges)
    if mode == "rms":
        power = np.einsum("ij,ij->i", data, data)
        sums = np.add.reduceat(power, starts, dtype=np.float64)
        level = np.sqrt(sums / (counts * data.shape[1]))
    else:
        level = np.maximum.reduceat(np.abs(data).max(axis=1), starts).astype(np.float64)
    return level


def _smooth(level, fps, attack_ms, release_ms):
    """A one-pole follower at the frame rate: rises with the attack time, falls with the release time, from silence."""
    dt = 1000.0 / fps
    a_up = math.exp(-dt / attack_ms) if attack_ms > 0 else 0.0
    a_down = math.exp(-dt / release_ms) if release_ms > 0 else 0.0
    out, y = [], 0.0
    for v in level:
        v = float(v)
        a = a_up if v > y else a_down
        y = a * y + (1.0 - a) * v
        out.append(y)
    return out


def op_bake_sound_keys(params):
    """Drive a property from a sound's loudness: one keyframe per scene frame, mapped to [minValue, maxValue].

    params:
      file (str, required)      the sound; read with Blender's own audio module, aud
      object (str, required)    the object whose property is keyed
      dataPath (str, required)  the property, from the object: scale, location, rotation_euler, data.energy ...
      index (int)               one array element, or -1 (default) for every element
      frameStart (int)          the frame the sound starts at; default the scene's start
      minValue / maxValue (float)  what silence and the loudest frame map to; default 0 and 1 (min above max inverts)
      attackMs / releaseMs (float) smoothing: how fast the level may rise and fall, in ms; default 0 (none)
      mode (str)                rms (default) - the level over each frame; or peak - the largest sample in it

    Frame frameStart + k holds the level of the sound's k-th 1/fps slice (the scene's own fps), after smoothing,
    scaled so the loudest frame maps to maxValue - a silent file keys minValue throughout and says so. The keys are
    LINEAR and written in one pass (keyframe_points.add + foreach_set), replacing any keys the curve already held in
    that frame range and keeping the rest. The path is keyed on the datablock that owns it - data.energy on the
    light - on every build, Blender 5.0's layered actions included.
    """
    reject_unknown(params, _BAKE_KEYS, "bake_sound_keys", notes={
        "smoothing": "two settings: attackMs (how fast the level may rise) and releaseMs (how fast it may fall), in ms",
        "frame": "spelled frameStart - the frame the sound starts at",
        "path": "spelled dataPath",
        "name": "spelled object",
        "normalize": "always on: the loudest frame maps to maxValue and silence to minValue",
        "interpolation": "baked keys are LINEAR - one per frame, so there is nothing to ease between",
    })
    tail = "NOTHING was keyed."
    sc = bpy.context.scene
    try:
        raw_file = take(params, "file", required=True, kind=str)
        obj = get_object(take(params, "object", required=True, kind=str))
        path = take(params, "dataPath", required=True, kind=str).strip()
        index = take_int(params, "index", default=-1)
        start = take_int(params, "frameStart", default=None)
        lo_v = take_float(params, "minValue", default=0.0)
        hi_v = take_float(params, "maxValue", default=1.0)
        attack = take_float(params, "attackMs", default=0.0)
        release = take_float(params, "releaseMs", default=0.0)
        mode = str(take(params, "mode", default="rms", kind=str)).strip().lower()
    except MifOpError as exc:
        text = str(exc)
        raise MifOpError(text if text.endswith(tail) else "%s %s" % (text.rstrip("."), tail)) from None
    require_editable(obj, "key")
    if not path:
        raise MifOpError("'dataPath' is blank - name the property to key, e.g. scale. %s" % tail)
    if mode not in _BAKE_MODES:
        raise MifOpError("mode must be rms (the level over each frame) or peak (the largest sample in it), got %r. %s"
                         % (mode, tail))
    for key, value in (("attackMs", attack), ("releaseMs", release)):
        _in_range(value, 0.0, 60000.0, key, tail)
    if lo_v == hi_v:
        raise MifOpError("minValue and maxValue are both %g - every key would hold the same value. %s" % (lo_v, tail))
    if index < -1:
        raise MifOpError("index must be -1 (every element) or an element number, got %d. %s" % (index, tail))
    start = int(sc.frame_start) if start is None else start
    _in_range(start, *_hard_range(sc, "frame_current"), key="frameStart", tail=tail)

    owner, owner_path, holder, prop = _bake_target(obj, path, tail)
    require_editable(owner, "key")
    if not prop.is_animatable:
        raise MifOpError("'%s' can be set but not animated - Blender does not key %s.%s. %s"
                         % (path, holder.bl_rna.identifier, prop.identifier, tail))
    if prop.type not in ("FLOAT", "INT"):
        raise MifOpError("'%s' is a %s property, and a sound's level is a number - key a FLOAT or INT property. %s"
                         % (path, prop.type, tail))
    size = int(getattr(prop, "array_length", 0) or 0)
    if index == -1:
        indices = list(range(size)) if size else [0]
    elif (size and index >= size) or (not size and index != 0):
        raise MifOpError("'%s' has %s, so index %d does not exist - use -1 for every element. %s"
                         % (path, ("%d elements" % size) if size else "one value", index, tail))
    else:
        indices = [index]
    _in_range(min(lo_v, hi_v), prop.hard_min, prop.hard_max, "minValue" if lo_v < hi_v else "maxValue", tail)
    _in_range(max(lo_v, hi_v), prop.hard_min, prop.hard_max, "maxValue" if lo_v < hi_v else "minValue", tail)

    sound_path, rate, channels, samples = _read_sound(raw_file, tail)
    if samples * channels > _BAKE_MAX_VALUES:
        raise MifOpError("'%s' holds %d samples across %d channel(s) - past the %d this op reads in one piece (about "
                         "35 minutes of stereo at 48 kHz). Trim the file. %s"
                         % (sound_path, samples, channels, _BAKE_MAX_VALUES, tail))
    try:
        import numpy as np
    except ImportError:
        raise MifOpError("this Blender's Python has no numpy, which the envelope needs. %s" % tail) from None
    import aud
    try:
        data = np.asarray(aud.Sound(sound_path).data(), dtype=np.float32)
    except (aud.error, MemoryError) as exc:
        raise MifOpError("could not read the samples of '%s' (%s). %s" % (sound_path, exc, tail)) from None
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    fps = _fps(sc)
    level = _envelope(data, rate, fps, mode, np)
    smoothed = _smooth(level, fps, attack, release)
    peak = max(smoothed) if smoothed else 0.0
    silent = peak <= 0.0
    values = [lo_v if silent else lo_v + (hi_v - lo_v) * (v / peak) for v in smoothed]
    count = len(values)
    frames = [float(start + k) for k in range(count)]
    first, last = frames[0], frames[-1]

    # ------------------------------------------------------------------ the writes: one pass per curve
    written, replaced = [], 0
    for i in indices:
        curve = next((fc for fc in _fcurves(owner) if fc.data_path == owner_path and fc.array_index == i), None)
        if curve is None:
            # THE CURVE, THROUGH keyframe_insert, because it builds whatever the build needs around it - an action on
            # 4.4 and earlier, and on 5.0 a layered action with its slot, layer, strip and channelbag. The key it
            # inserts is inside the range, so the pass below replaces it with the baked one. A SCALAR takes index -1:
            # keyframe_insert raises for index 0 on a property that is not an array (measured on 4.4 with a light's
            # energy), while its one curve still reads back as array_index 0.
            try:
                owner.keyframe_insert(data_path=owner_path, index=i if size else -1, frame=first)
            except (TypeError, RuntimeError) as exc:
                raise MifOpError(
                    "Blender would not start an F-curve for %s[%d] on %s '%s': %s%s"
                    % (owner_path, i, type(owner).__name__, owner.name, exc,
                       " - the curves before it were keyed" if written else ". NOTHING was keyed.")) from None
            curve = next((fc for fc in _fcurves(owner) if fc.data_path == owner_path and fc.array_index == i), None)
        if curve is None:
            raise MifOpError("Blender made no F-curve for %s[%d] on %s '%s' - keys were not written there."
                             % (owner_path, i, type(owner).__name__, owner.name))
        points = curve.keyframe_points
        for kp in reversed(list(points)):
            if first - 0.5 <= kp.co[0] <= last + 0.5:
                points.remove(kp, fast=True)
                replaced += 1
        kept = len(points)
        co = [0.0] * (2 * kept)
        points.foreach_get("co", co)
        points.add(count)
        for f, v in zip(frames, values, strict=True):
            co.extend((f, v))
        points.foreach_set("co", co)
        points.foreach_set("handle_left", co)
        points.foreach_set("handle_right", co)
        for kp in points[kept:]:
            kp.interpolation = "LINEAR"
        curve.update()
        in_range = [kp for kp in curve.keyframe_points if first - 0.5 <= kp.co[0] <= last + 0.5]
        written.append({"dataPath": owner_path, "index": i, "keys": len(in_range),
                        "firstKey": [round(float(in_range[0].co[0]), 4), round(float(in_range[0].co[1]), 6)]
                        if in_range else None,
                        "lastKey": [round(float(in_range[-1].co[0]), 4), round(float(in_range[-1].co[1]), 6)]
                        if in_range else None})
    short = [w for w in written if w["keys"] != count]
    if short:
        raise MifOpError("the keys were written and %s read back %s of the %d asked for - the curve holds what it "
                         "reads back now, not the bake." % (", ".join("%s[%d]" % (w["dataPath"], w["index"])
                                                                      for w in short),
                                                           "/".join(str(w["keys"]) for w in short), count))
    peak_at = max(range(count), key=lambda k: smoothed[k]) if count else None
    out = {
        "object": obj.name,
        "owner": owner.name,
        "ownerType": type(owner).__name__,
        "dataPath": owner_path,
        "requestedPath": path,
        "indices": indices,
        "curves": written,
        "keyCount": sum(w["keys"] for w in written),
        "keysPerCurve": count,
        "frameStart": int(first),
        "frameEnd": int(last),
        "fps": round(fps, 6),
        "mode": mode,
        "attackMs": attack,
        "releaseMs": release,
        "minValue": lo_v,
        "maxValue": hi_v,
        "envelopeMin": round(min(smoothed), 6) if smoothed else None,
        "envelopeMax": round(peak, 6),
        "envelopePeakFrame": int(start + peak_at) if peak_at is not None else None,
        "silent": silent,
        "replacedKeys": replaced,
        "interpolation": "LINEAR",
        "propertyType": prop.type,
        "sound": {"filepath": sound_path, "sampleRate": rate, "channels": channels,
                  "durationSeconds": round(samples / rate, 6)},
    }
    if silent:
        out["silentNote"] = "the file is silent, so every key holds minValue."
    if last > sc.frame_end:
        out["rangeNote"] = ("the keys run to frame %d, past the scene's end (%d) - the sound is longer than what is "
                            "left of the scene." % (int(last), sc.frame_end))
    if prop.type == "INT":
        out["valueNote"] = "an INT property evaluates to the nearest whole number, so the level moves in steps."
    return out


# ---------------------------------------------------------------------------------------------------------------
# extract_audio
# ---------------------------------------------------------------------------------------------------------------

_EXTRACT_KEYS = ("filepath", "outPath", "gainDb", "bitFormat")
_EXTRACT_WIDTH = {"S16": 2, "S24": 3, "S32": 4}
_EXTRACT_CHUNK_S = 10       # whole seconds per read, so every chunk boundary falls on a sample
_EXTRACT_GAIN_DB = (-120.0, 60.0)


def op_extract_audio(params):
    """Write the audio of a file - a video's soundtrack, or any sound file Blender reads - to a WAV, with an optional
    gain. The read half of a mux: mixdown_audio renders the scene's audio, render_animation carries it into a video
    (set_render_settings' ffmpeg group), and this reads it back out of the video so it can be measured - and, with a
    gain, it is how a normalized copy of a mixdown is written.

    params:
      filepath (str, required)  the source, decoded by Blender's own audio library (aud) through its FFmpeg: a
                                video's soundtrack (MP4, MKV, MOV, WebM...), WAV, FLAC, OGG, MP3
      outPath (str, required)   the WAV to write; its extension must be .wav, and it cannot be the source
      gainDb (float)            added to every sample, default 0, from -120 to +60. A sample the gain pushes past full
                                scale is clipped, and clippedSamples counts them
      bitFormat (str)           S16 (default), S24 or S32 integer PCM

    MEASURED 2026-09-23 on 3.6.23, 4.2.17, 4.4.0, 5.0.1 and 5.2.1: an AAC track read back out of a one-second MP4 as
    48 kHz stereo at the level the scene rendered it. aud's own Sound.write failed on that track on every build
    ("filling the audio frame failed with ffmpeg"), so the samples are read with Sound.data() and written by
    Python's wave module. The file is read in whole-second chunks - a long soundtrack never sits in memory at once -
    written beside the target and renamed over it once complete, and the reply is read off the written header.
    """
    reject_unknown(params, _EXTRACT_KEYS, "extract_audio", notes={
        "path": "spelled filepath (the source) - the WAV written is outPath",
        "file": "spelled filepath",
        "output": "spelled outPath",
        "volume": "spelled gainDb, in dB",
        "normalize": ("this op applies the gain it is given; bl_normalize_audio measures a mixdown and computes the "
                      "gain for a named loudness target"),
        "sampleRate": "the source's own rate is kept - Blender resamples in mixdown_audio, not here",
        "channels": "the source's own channel count is kept",
    })
    tail = "NOTHING was written."
    try:
        src_raw = take(params, "filepath", required=True, kind=str)
        out_raw = take(params, "outPath", required=True, kind=str)
        gain_db = take_float(params, "gainDb", default=0.0)
        bit_format = str(take(params, "bitFormat", default="S16", kind=str)).upper()
    except MifOpError as exc:
        raise MifOpError("%s %s" % (exc, tail))
    src = bpy.path.abspath(str(src_raw))
    if not os.path.isfile(src):
        raise MifOpError("no file at '%s'. %s" % (src_raw, tail))
    out = bpy.path.abspath(str(out_raw))
    check_output_path(out_raw, out, "written")
    if os.path.splitext(out)[1].lower() != ".wav":
        raise MifOpError("outPath must end in .wav - this op writes WAV, got '%s'. %s" % (out_raw, tail))
    if os.path.normcase(os.path.abspath(out)) == os.path.normcase(os.path.abspath(src)):
        raise MifOpError("outPath is the source file itself - it would be written while it is read. Name another "
                         "file. %s" % tail)
    if bit_format not in _EXTRACT_WIDTH:
        raise MifOpError("bitFormat is one of %s, got '%s'. %s" % (", ".join(sorted(_EXTRACT_WIDTH)), bit_format, tail))
    _in_range(gain_db, _EXTRACT_GAIN_DB[0], _EXTRACT_GAIN_DB[1], "gainDb", tail)
    check_writable_dir(os.path.dirname(out) or ".")

    import aud
    import numpy as np
    import wave
    try:
        snd = aud.Sound(src)
        rate, channels = snd.specs
        frames = int(snd.length)
    except Exception as exc:                                       # noqa: BLE001 - aud raises its own error type
        raise MifOpError("Blender's audio library could not read sound from '%s' (%s) - it is not audio, or it is a "
                         "video with no soundtrack. %s" % (src_raw, exc, tail))
    if frames <= 0 or channels < 1 or rate <= 0:
        raise MifOpError("'%s' holds no sound (%d frames, %d channels at %s Hz) - a video rendered with ffmpeg "
                         "audio_codec NONE has no soundtrack. %s" % (src_raw, frames, channels, rate, tail))
    width = _EXTRACT_WIDTH[bit_format]
    full = float(2 ** (8 * width - 1) - 1)
    gain = 10.0 ** (gain_db / 20.0)
    tmp = "%s.%s.tmp" % (out, uuid.uuid4().hex[:8])
    written = clipped = 0
    peak = 0.0
    try:
        with wave.open(tmp, "wb") as w:
            w.setnchannels(int(channels))
            w.setsampwidth(width)
            w.setframerate(int(round(rate)))
            start = 0
            while written < frames:
                part = snd.limit(float(start), float(start + _EXTRACT_CHUNK_S)).data()
                if part is None or len(part) == 0:
                    break
                x = np.asarray(part, dtype=np.float64) * gain
                over = np.abs(x) > 1.0
                clipped += int(over.sum())
                x = np.clip(x, -1.0, 1.0)
                peak = max(peak, float(np.abs(x).max()))
                ints = np.round(x * full).astype("<i4")
                if width == 2:
                    raw = ints.astype("<i2").tobytes()
                elif width == 3:
                    raw = ints.view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
                else:
                    raw = ints.tobytes()
                w.writeframes(raw)
                written += len(part)
                start += _EXTRACT_CHUNK_S
        os.replace(tmp, out)
    except Exception as exc:                                       # noqa: BLE001 - nothing half-written is left
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise MifOpError("writing '%s' failed (%s). %s" % (out_raw, exc, tail))
    complete, info = _wav_header(out)
    if not complete or not info:
        raise MifOpError("'%s' was written but its WAV header does not read back complete. The file is there; do not "
                         "trust it." % out_raw)
    reply = {
        "wroteFile": out, "fileBytes": os.path.getsize(out),
        "sampleRate": info["rate"], "channels": info["channels"], "bitsPerSample": info["bits"],
        "frames": info["samples"], "durationSeconds": round(info["samples"] / float(info["rate"]), 6),
        "source": {"filepath": src, "sampleRate": rate, "channels": int(channels), "frames": frames},
        "gainDb": round(gain_db, 6),
        "peakDbfs": round(20.0 * math.log10(peak), 3) if peak > 0 else None,
        "clippedSamples": clipped,
    }
    if info["samples"] != frames:
        reply["framesNote"] = ("wrote %d frames of the source's %d - its decoder reported a length it did not deliver"
                               % (info["samples"], frames))
    if clipped:
        reply["clipNote"] = ("%d sample(s) went past full scale at %+.2f dB and were clipped - that is distortion. "
                             "Normalize to a true-peak ceiling (bl_normalize_audio) instead of a gain that clips."
                             % (clipped, gain_db))
    return reply


OPS = {
    "set_scene_audio": op_set_scene_audio,
    "add_sound": op_add_sound,
    "mixdown_audio": op_mixdown_audio,
    "bake_sound_keys": op_bake_sound_keys,
    "extract_audio": op_extract_audio,
}
