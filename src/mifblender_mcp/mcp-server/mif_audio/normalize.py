"""Loudness normalization to a named target, for the Blender scene's mixdown and the video it is muxed into.

THE TARGETS, read in the standards themselves on 2026-09-23 (LUFS and LKFS are one unit):
  ebu_r128        EBU R 128 (V5, November 2023): programme loudness normalised to a Target Level of -23.0 LUFS,
                  a True Peak Level that does not exceed -1 dBTP; +/-0.2 LU allowed for measurement in quality control
  aswg_home       ASWG-R001: home-based platforms normalized to a Target Level of -24 (+/-2) LKFS; maximum True Peak
                  Level -1 dBTP
  aswg_portable   ASWG-R001: portable platforms normalized to a Target Level of -18 (+/-2) LKFS; the same -1 dBTP
ASWG-R001 also asks for at least 30 minutes of representative gameplay to be measured; a scene mixdown is whatever
the scene holds, so its compliance is only as good as what was measured, and the answer says what that was.

THE GAIN IS LINEAR: target minus measured integrated loudness - unless that would lift the true peak past the target's
ceiling, in which case the ceiling wins, the result is quieter than the target, and the answer says so (peakLimited).
Nothing here compresses or limits. That is a mix decision, not a normalization.

ONE MIXDOWN, THEN ARITHMETIC. On Blender 5.2 a background session writes only its first mixdown (ops_audio's
docstring), so the gain is computed from one mixdown and applied twice without mixing again: to a file, through
extract_audio's gainDb, and to the scene's master volume (set_scene_audio volume), which a video render reads -
measured linear on 3.6.23-5.2.1 (scene volume 0.5 put a rendered AAC track 6.02 dB down).

Transport-free: run() takes the caller's blender(op, **params).
"""
import math
import os
import shutil
import tempfile
import time

from . import measure as M

TARGETS = {
    "ebu_r128": {"lufs": -23.0, "truePeakDbtp": -1.0, "toleranceLu": 0.2,
                 "standard": "EBU R 128 (2023): -23.0 LUFS, true peak at most -1 dBTP"},
    "aswg_home": {"lufs": -24.0, "truePeakDbtp": -1.0, "toleranceLu": 2.0,
                  "standard": "ASWG-R001 home platforms: -24 (+/-2) LKFS, true peak at most -1 dBTP"},
    "aswg_portable": {"lufs": -18.0, "truePeakDbtp": -1.0, "toleranceLu": 2.0,
                      "standard": "ASWG-R001 portable platforms: -18 (+/-2) LKFS, true peak at most -1 dBTP"},
}
# how close a normalized result must measure to what the arithmetic predicts: EBU R 128's quality-control tolerance
CHECK_LU = 0.2
# a gain is written as 16- or 24-bit samples; rounding them moves a true peak by far less than this
CHECK_TP_DB = 0.05
SCENE_VOLUME_RANGE = (0.0, 100.0)       # Scene.audio_volume's RNA range on 3.6.23-5.2.1


def _fail(stage, msg, **extra):
    out = {"ok": False, "stage": stage, "error": msg}
    out.update(extra)
    return out


def resolve_target(target, true_peak=None):
    """A target name or a number of LUFS -> {lufs, truePeakDbtp, toleranceLu, standard, name}, or raises ValueError."""
    if isinstance(target, str) and target.strip().lower() not in TARGETS:
        try:
            target = float(target)             # "-20" arrives as text over MCP
        except ValueError:
            pass
    if isinstance(target, str) and target.strip().lower() in TARGETS:
        name = target.strip().lower()
        t = dict(TARGETS[name], name=name)
    elif isinstance(target, (int, float)) and not isinstance(target, bool) and math.isfinite(target):
        if not -70.0 < float(target) < 0.0:
            raise ValueError("a target in LUFS lies between -70 and 0, got %r" % (target,))
        t = {"lufs": float(target), "truePeakDbtp": -1.0, "toleranceLu": CHECK_LU, "name": "custom",
             "standard": "a custom target of %g LUFS" % float(target)}
    else:
        raise ValueError("target is one of %s or a number of LUFS, got %r" % (", ".join(sorted(TARGETS)), target))
    if true_peak is not None:
        if isinstance(true_peak, bool) or not isinstance(true_peak, (int, float)) or not -20.0 <= true_peak <= 0.0:
            raise ValueError("true_peak is a ceiling in dBTP from -20 to 0, got %r" % (true_peak,))
        t["truePeakDbtp"] = float(true_peak)
    return t


def gain_for(measured, target):
    """The gain that puts `measured` (a measure.measure() answer) on `target` -> {gainDb, peakLimited, expectedLufs,
    expectedTruePeakDbtp}, or {error} for a file with no integrated loudness (silent or shorter than 0.4 s)."""
    lufs, tp = measured.get("integratedLufs"), measured.get("truePeakDbtp")
    if lufs is None or tp is None:
        return {"error": "the file has no integrated loudness - it is silent or shorter than 0.4 s - so there is "
                         "nothing to normalize"}
    want = target["lufs"] - lufs
    room = target["truePeakDbtp"] - tp
    gain = min(want, room)
    return {"gainDb": round(gain, 3), "peakLimited": gain < want - 1e-9,
            "expectedLufs": round(lufs + gain, 2), "expectedTruePeakDbtp": round(tp + gain, 2),
            "wantedGainDb": round(want, 3)}


def _measured(m):
    return {k: m.get(k) for k in ("integratedLufs", "truePeakDbtp", "samplePeakDbfs", "loudnessRangeLu",
                                  "duration_s")}


def run(blender, target="ebu_r128", filepath=None, out_path=None, set_scene_volume=None, true_peak=None,
        bit_format="S24"):
    """Measure, compute the gain, and apply it - to out_path (a normalized WAV) and/or the scene's master volume.
    With no filepath the scene is mixed down first (its one mixdown on a 5.2 background session), and the scene's
    volume is set by default; with a filepath only a file is written unless set_scene_volume is asked for."""
    try:
        t = resolve_target(target, true_peak)
    except ValueError as exc:
        return _fail("plan", "%s. NOTHING was changed." % exc)
    if set_scene_volume is None:
        set_scene_volume = filepath is None
    if not out_path and not set_scene_volume:
        return _fail("plan", "nothing to apply the gain to: pass out_path (a normalized WAV), or leave "
                             "set_scene_volume on so the scene's master volume carries it into the next render. "
                             "NOTHING was changed.")
    if out_path and os.path.splitext(str(out_path))[1].lower() != ".wav":
        return _fail("plan", "out_path must end in .wav, got %r. NOTHING was changed." % (out_path,))
    # THE MIXDOWN AND A DECODED SOURCE LIVE HERE AND GO WITH IT, whatever the outcome: a folder per call left in
    # the temp directory is the leak the Blender version matrix had (4672fdc7)
    work = tempfile.mkdtemp(prefix="mif_normalize_")
    try:
        return _apply(blender, t, filepath, out_path, set_scene_volume, bit_format, work)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _apply(blender, t, filepath, out_path, set_scene_volume, bit_format, work):
    """run()'s body, with its scratch folder made and removed by run()."""
    steps = {}
    source = filepath
    if source is None:
        mix = os.path.join(work, "mixdown.wav")
        r = blender("mixdown_audio", filepath=mix, container="WAV", bitFormat="S24")
        steps["mixdown"] = {k: r.get(k) for k in ("ok", "error", "wroteFile", "durationSeconds", "silentSpeakers")}
        if r.get("ok") is not True or not os.path.isfile(mix):
            return _fail("mixdown", r.get("error") or "mixdown_audio wrote no file", steps=steps)
        source = mix
    wav = source
    m = M.measure(wav) if os.path.splitext(wav)[1].lower() == ".wav" else {"ok": False}
    if not m.get("ok"):
        # not a WAV this meter reads (FLAC, a video's soundtrack...): Blender decodes it first, at no gain
        wav = os.path.join(work, "source.wav")
        r = blender("extract_audio", filepath=source, outPath=wav, bitFormat="S24")
        steps["decode"] = {k: r.get(k) for k in ("ok", "error", "wroteFile", "durationSeconds")}
        if r.get("ok") is not True or not os.path.isfile(wav):
            return _fail("decode", r.get("error") or "extract_audio wrote no file", steps=steps)
        m = M.measure(wav)
        if not m.get("ok"):
            return _fail("measure", m.get("error"), steps=steps)
    g = gain_for(m, t)
    if g.get("error"):
        return _fail("measure", g["error"] + ". NOTHING was changed.", measured=_measured(m), steps=steps)
    out = {"ok": True, "target": {k: t[k] for k in ("name", "lufs", "truePeakDbtp", "standard")},
           "source": filepath or "the scene's mixdown", "measured": _measured(m), "gainDb": g["gainDb"], "peakLimited": g["peakLimited"],
           "expected": {"integratedLufs": g["expectedLufs"], "truePeakDbtp": g["expectedTruePeakDbtp"]},
           "steps": steps}
    if g["peakLimited"]:
        out["peakLimitedNote"] = ("the full %+.2f dB would lift the true peak past %g dBTP, so the gain stops at "
                                  "%+.2f dB and the result is %.2f LU under the target. Reaching it takes a limiter or "
                                  "a mix with less crest - a choice this tool does not make."
                                  % (g["wantedGainDb"], t["truePeakDbtp"], g["gainDb"], g["wantedGainDb"] - g["gainDb"]))
    if out_path:
        r = blender("extract_audio", filepath=source, outPath=out_path, gainDb=g["gainDb"], bitFormat=bit_format)
        steps["write"] = {k: r.get(k) for k in ("ok", "error", "wroteFile", "clippedSamples", "peakDbfs")}
        if r.get("ok") is not True or not os.path.isfile(str(out_path)):
            return dict(out, ok=False, stage="write", error=r.get("error") or "extract_audio wrote no file")
        n = M.measure(str(out_path))
        got = _measured(n) if n.get("ok") else {}
        lufs_ok = got.get("integratedLufs") is not None and abs(got["integratedLufs"] - g["expectedLufs"]) <= CHECK_LU
        tp_ok = got.get("truePeakDbtp") is not None and got["truePeakDbtp"] <= t["truePeakDbtp"] + CHECK_TP_DB
        out["output"] = {"file": str(out_path), "measured": got, "verified": bool(lufs_ok and tp_ok),
                         "withinStandard": (got.get("integratedLufs") is not None
                                            and abs(got["integratedLufs"] - t["lufs"]) <= t["toleranceLu"]
                                            and tp_ok)}
        if not out["output"]["verified"]:
            out.update(ok=False, stage="verify",
                       error="the normalized file measures %s LUFS / %s dBTP, not the %s LUFS / at most %g dBTP the "
                             "gain predicts" % (got.get("integratedLufs"), got.get("truePeakDbtp"), g["expectedLufs"],
                                                t["truePeakDbtp"]))
            return out
    if set_scene_volume:
        # a bare set_scene_audio reads; its reply carries the scene's settings at the top level (ops_audio)
        cur = blender("set_scene_audio")
        before = cur.get("volume") if isinstance(cur, dict) and cur.get("ok") is True else None
        if not isinstance(before, (int, float)):
            return dict(out, ok=False, stage="scene", error="set_scene_audio did not report the scene's volume: %s"
                        % (cur.get("error") if isinstance(cur, dict) else cur))
        after = before * 10.0 ** (g["gainDb"] / 20.0)
        if not SCENE_VOLUME_RANGE[0] < after <= SCENE_VOLUME_RANGE[1]:
            return dict(out, ok=False, stage="scene",
                        error="the scene volume would have to be %.4g (from %.4g), outside Blender's %g-%g - "
                              "normalize the file (out_path) and use it in the render instead"
                              % (after, before, SCENE_VOLUME_RANGE[0], SCENE_VOLUME_RANGE[1]))
        w = blender("set_scene_audio", volume=round(after, 6))
        read = w.get("volume") if isinstance(w, dict) and w.get("ok") is True else None
        if not isinstance(read, (int, float)) or abs(read - after) > 1e-4 * max(1.0, after):
            return dict(out, ok=False, stage="scene", error="the scene volume reads %r after writing %.6g: %s"
                        % (read, after, w.get("error") if isinstance(w, dict) else w))
        out["scene"] = {"volumeBefore": before, "volumeAfter": read,
                        "renderNote": ("the scene now renders at the target: set_render_settings {fileFormat: FFMPEG, "
                                       "ffmpeg: {format, codec, audio_codec}}, save_file, render_animation - then "
                                       "bl_extract_audio the video and mif_measure_audio it to check")}
    out["measuredSpan"] = ("%.1f s of audio measured%s" % (m.get("duration_s") or 0.0,
                           "; ASWG-R001 asks for at least 30 minutes of representative gameplay" if
                           t["name"].startswith("aswg") and (m.get("duration_s") or 0) < 1800 else ""))
    out["took"] = time.strftime("%H:%M:%S")
    return out
