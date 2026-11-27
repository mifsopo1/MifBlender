"""Video and picture reads through the user's own ffprobe and ffmpeg: a clip's facts, frames from it, and a GIF or WebP
turned into a PNG this server can show.

WHY (FOUND 2026-11-16 from the 5.9 extract): the native media toolset reads image files, a video's container facts and
frames over its length; ours viewed PNG, BMP and JPG only and ran ffmpeg inside start_mocap_performance alone. A user
gains a way to look at a render or a reference clip - how long, what frame rate, what it shows at five points - without
opening a player. ffprobe and ffmpeg, not Windows Media Foundation, so Linux and macOS work too.

Every run is a heavy job of mif_childproc (a slot of MIF_MAX_CHILD_JOBS, the memory ceiling, no window). Nothing is
written but PNG frames in a new folder under the system temp folder, named in each reply; folders of earlier calls older
than a day are removed when a new one is made.
"""
import json
import os
import shutil
import subprocess
import tempfile
import time

MAX_FRAMES = 16
TEMP_PREFIX = "mif_video_frames_"
_TIMEOUT = 120


class Refusal(Exception):
    pass


def tools(ffmpeg):
    """(ffmpeg, ffprobe) paths, ffprobe beside ffmpeg first, then on PATH. Refusal naming what is missing."""
    if not ffmpeg:
        raise Refusal("ffmpeg was not found on PATH (or MIF_FFMPEG) - install it to read videos, GIFs and WebP here")
    probe = os.path.join(os.path.dirname(ffmpeg), "ffprobe" + (".exe" if ffmpeg.lower().endswith(".exe") else ""))
    if not os.path.isfile(probe):
        probe = shutil.which("ffprobe")
    if not probe:
        raise Refusal("ffprobe was not found beside %s or on PATH - it ships with ffmpeg; install the full build" % ffmpeg)
    return ffmpeg, probe


# Every ffmpeg/ffprobe call passes creationflags CREATE_NO_WINDOW: a console program started from a windowless MCP
# server would otherwise open a console window on the person's desktop. The getattr is 0 off Windows, as it must be.


def _rate(text):
    try:
        num, _, den = str(text).partition("/")
        r = float(num) / float(den or 1)
        return round(r, 3) if r > 0 else None
    except (ValueError, ZeroDivisionError):
        return None


def info(path, ffprobe):
    """{durationSeconds, width, height, frameRate, frames, codec, container, sizeBytes, hasAudio}. Refusal when ffprobe
    cannot read it as a video."""
    if not isinstance(path, str) or not os.path.isfile(path):
        raise Refusal("%r is not a file this server can read" % (path,))
    try:
        # window: none - _hidden() passes CREATE_NO_WINDOW
        r = subprocess.run([ffprobe, "-v", "error", "-show_entries",
                            "format=duration,format_name,size:stream=codec_type,codec_name,width,height,avg_frame_rate,"
                            "r_frame_rate,nb_frames,duration", "-of", "json", path],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=_TIMEOUT,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise Refusal("ffprobe could not run on %s: %s" % (path, exc))
    try:
        j = json.loads(r.stdout or "{}")
    except ValueError:
        j = {}
    streams = j.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if r.returncode != 0 or not video:
        raise Refusal("%s is not a video ffprobe can read%s" % (path, (": " + (r.stderr or "").strip()[-300:]) if r.stderr else ""))
    fmt = j.get("format") or {}
    try:
        duration = float(fmt.get("duration") or video.get("duration") or 0)
    except ValueError:
        duration = 0.0
    try:
        frames = int(video.get("nb_frames")) if video.get("nb_frames") not in (None, "N/A") else None
    except ValueError:
        frames = None
    return {"durationSeconds": round(duration, 3), "width": video.get("width"), "height": video.get("height"),
            "frameRate": _rate(video.get("avg_frame_rate")) or _rate(video.get("r_frame_rate")), "frames": frames,
            "codec": video.get("codec_name"), "container": fmt.get("format_name"),
            "sizeBytes": int(fmt.get("size") or os.path.getsize(path)),
            "hasAudio": any(s.get("codec_type") == "audio" for s in streams)}


def times(duration, mode, count, t_from=None, t_to=None):
    """The times to sample: count midpoints of equal spans over the whole clip (skim) or over [t_from, t_to] (range)."""
    if mode == "skim":
        lo, hi = 0.0, float(duration)
    else:
        lo = 0.0 if t_from is None else float(t_from)
        hi = float(duration) if t_to is None else float(t_to)
    if not hi > lo:
        raise Refusal("the range %.3f to %.3f s holds no time (the clip is %.3f s)" % (lo, hi, duration))
    step = (hi - lo) / count
    return [round(lo + step * (i + 0.5), 4) for i in range(count)]


def _sweep_old(root):
    cutoff = time.time() - 86400
    for name in os.listdir(root):
        if name.startswith(TEMP_PREFIX):
            p = os.path.join(root, name)
            try:
                if os.path.getmtime(p) < cutoff:
                    shutil.rmtree(p, ignore_errors=True)
            except OSError:
                pass


def new_folder():
    root = tempfile.gettempdir()
    _sweep_old(root)
    return tempfile.mkdtemp(prefix=TEMP_PREFIX, dir=root)


def frame(run, ffmpeg, path, t, out, max_px):
    """One frame at t seconds as a PNG at out, its long edge at most max_px. run is mif_childproc.run."""
    # fit inside max_px x max_px, never larger than the source
    scale = "scale='min(iw,%d)':'min(ih,%d)':force_original_aspect_ratio=decrease" % (max_px, max_px)
    cmd = [ffmpeg, "-v", "error", "-y", "-ss", "%.4f" % t, "-i", path, "-frames:v", "1", "-vf", scale, out]
    done, job, refusal = run(cmd, "ffmpeg", os.path.basename(path), timeout=_TIMEOUT)
    if refusal:
        raise Refusal(refusal)
    if job and (job.get("timedOut") or job.get("killedForMemory")):
        raise Refusal("ffmpeg did not finish a frame of %s (%s)" % (path, "timed out" if job.get("timedOut") else "memory"))
    if done.returncode != 0 or not os.path.isfile(out):
        raise Refusal("ffmpeg could not read a frame at %.3f s of %s: %s" % (t, path, (done.stderr or "").strip()[-300:]))
    return out


def still_png(run, ffmpeg, path, out):
    """A GIF's or WebP's first frame as a PNG, at its own size."""
    cmd = [ffmpeg, "-v", "error", "-y", "-i", path, "-frames:v", "1", out]
    done, job, refusal = run(cmd, "ffmpeg", os.path.basename(path), timeout=_TIMEOUT)
    if refusal:
        raise Refusal(refusal)
    if done.returncode != 0 or not os.path.isfile(out):
        raise Refusal("ffmpeg could not read %s: %s" % (path, (done.stderr or "").strip()[-300:]))
    return out


def sniff(path):
    """'gif', 'webp' or None, by the file's first bytes - never by its name."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(12)
    except OSError:
        return None
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    return None
