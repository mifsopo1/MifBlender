"""A spectrum over time for mif_measure_audio: the dominant frequency, a pitch track, harmonic levels, the pitch's low,
high, rise and cycle. Standard library only; every transform is mif_audio.fft.

  frames      the channels' average, cut into frames of the power of two at or above 80 ms (4096 samples at 44.1 and
              48 kHz: 11.7 Hz bins), periodic Hann, one frame every stepMs. A frame's t is its centre.
  pitch       the strongest partial between minHz and maxHz, its frequency refined by a parabola through the log power
              of the peak bin and its neighbours. It is the DOMINANT partial: a sound whose fundamental is weaker than
              its second harmonic tracks the harmonic. A frame is pitched when that partial (+/-3 bins) holds a quarter
              or more of the frame's power between minHz and maxHz and sits within 50 dB of the loudest pitched frame
              (ours, both: noise spreads its power over every bin and never holds a quarter in seven).
  harmonics   multiples n = 1 .. N of each pitched frame's pitch, each read as a sine's level in dBFS (a full-scale
              sine is 0 dB) off the bins round n x pitch (mif_audio.fft.sine_power - Parseval, so it does not depend on
              where between bins the partial falls), widened by how far that partial sweeps across the frame, and
              power-averaged over the pitched frames.
  low, high   the lowest and highest pitch of the pitched frames after a three-frame median (one stray frame cannot
              set them).
  riseFraction  the share of the pitch's frame-to-frame moves that go up - a siren rising 2.3 s of a 4.8 s cycle
              reads 0.48. Null when the pitch does not move.
  cycleSeconds  the lag of the first peak (correlation 0.5 or more) of the pitch track's autocorrelation after it
              first goes negative - the repeat period of a siren's wail or a vibrato. Null when nothing repeats within
              the file, which needs about one and a half cycles.
"""
import math

from mif_audio import fft as F

TONAL_SHARE = 0.25        # ours: a pitched frame's strongest partial holds this share of the band's power
RANGE_DB = 50.0           # ours: and sits within this many dB of the loudest pitched frame
LOBE = 3                  # Hann's main lobe is +/-2 bins; one more catches the parabola's spill
MAX_SPREAD = 16           # widest sweep (bins) a harmonic is followed over within one frame
MAX_POINTS = 3000


def _parabola(p, k):
    """Peak offset in bins from a parabola through ln p[k-1], ln p[k], ln p[k+1]."""
    if k <= 0 or k >= len(p) - 1 or min(p[k - 1], p[k], p[k + 1]) <= 0:
        return 0.0
    a, b, c = math.log(p[k - 1]), math.log(p[k]), math.log(p[k + 1])
    den = a - 2.0 * b + c
    return 0.5 * (a - c) / den if den < 0 else 0.0


def _median3(vals):
    out = list(vals)
    for i in range(1, len(vals) - 1):
        trio = [vals[i - 1], vals[i], vals[i + 1]]
        if None not in trio:
            out[i] = sorted(trio)[1]
    return out


def _options(spec, rate):
    if spec is True:
        spec = {}
    if not isinstance(spec, dict):
        return None, "spectrum must be true or an object {stepMs, minHz, maxHz, harmonics, maxPoints}, got %r." % (
            spec,)
    unknown = sorted(set(spec) - {"stepMs", "minHz", "maxHz", "harmonics", "maxPoints"})
    if unknown:
        return None, ("spectrum has no field %s - it takes stepMs, minHz, maxHz, harmonics and maxPoints."
                      % ", ".join(unknown))
    o = {"stepMs": spec.get("stepMs", 50), "minHz": spec.get("minHz", 40.0),
         "maxHz": spec.get("maxHz", min(16000.0, 0.45 * rate)), "harmonics": spec.get("harmonics", 6),
         "maxPoints": spec.get("maxPoints", MAX_POINTS)}
    for k in ("stepMs", "minHz", "maxHz"):
        if isinstance(o[k], bool) or not isinstance(o[k], (int, float)):
            return None, "spectrum.%s must be a number, got %r." % (k, o[k])
    if not 10 <= o["stepMs"] <= 1000:
        return None, "spectrum.stepMs must be 10-1000 (ms between frames), got %r." % (o["stepMs"],)
    if o["minHz"] < 0 or o["maxHz"] > rate / 2.0 or o["maxHz"] <= o["minHz"]:
        return None, ("spectrum.maxHz must be over minHz, minHz 0 or more and maxHz at most half the sample rate "
                      "(%g Hz) - got minHz %r, maxHz %r." % (rate / 2.0, o["minHz"], o["maxHz"]))
    for k, hi in (("harmonics", 16), ("maxPoints", MAX_POINTS)):
        if isinstance(o[k], bool) or not isinstance(o[k], int) or not 1 <= o[k] <= hi:
            return None, "spectrum.%s must be an integer 1-%d, got %r." % (k, hi, o[k])
    return o, None


def analyze(rate, channels, spec):
    """-> (spectrum dict, None) or (None, error)."""
    o, err = _options(spec, rate)
    if err:
        return None, err
    n = len(channels[0])
    nch = float(len(channels))
    mono = [sum(ch[i] for ch in channels) / nch for i in range(n)] if len(channels) > 1 else list(channels[0])
    size = F.next_pow2(int(math.ceil(rate * 0.08)))
    if n < size:
        return None, ("the file (%d samples) is shorter than one analysis frame (%d samples, %.0f ms)."
                      % (n, size, 1000.0 * size / rate))
    hop = max(1, int(round(o["stepMs"] * rate / 1000.0)))
    starts = list(range(0, n - size + 1, hop))
    if len(starts) > o["maxPoints"]:
        need = int(math.ceil(1000.0 * (n - size) / rate / max(1, o["maxPoints"] - 1)))
        return None, ("a spectrum at stepMs %g has %d frames, over maxPoints %d - give stepMs %d or more."
                      % (o["stepMs"], len(starts), o["maxPoints"], need))
    if not any(v != 0.0 for v in mono):
        return None, "the file is silent - every sample is zero, so it has no spectrum and no pitch."
    win = F.hann(size)
    wsq = sum(v * v for v in win)
    bin_hz = rate / float(size)
    kmin = max(1, int(math.ceil(o["minHz"] / bin_hz)))
    kmax = min(size // 2 - 1, int(math.floor(o["maxHz"] / bin_hz)))
    avg = [0.0] * (size // 2 + 1)
    rows = []                                     # (f0 or None, level dB, {n: (k0, slice)})
    reach = LOBE + MAX_SPREAD
    for s in starts:
        p = [abs(v) ** 2 for v in F.rfft([mono[s + i] * win[i] for i in range(size)])]
        for i, v in enumerate(p):
            avg[i] += v
        band = p[kmin:kmax + 1]
        total = sum(band)
        k = kmin + max(range(len(band)), key=band.__getitem__)
        lobe = sum(p[max(0, k - LOBE):k + LOBE + 1])
        if total <= 0 or lobe < TONAL_SHARE * total:
            rows.append((None, None, {}))
            continue
        f0 = (k + _parabola(p, k)) * bin_hz
        keep = {}
        for h in range(1, o["harmonics"] + 1):
            c = int(round(h * f0 / bin_hz))
            if c + LOBE + 1 > size // 2:
                break
            keep[h] = (max(0, c - reach), p[max(0, c - reach):c + reach + 1])
        rows.append((f0, 10.0 * math.log10(max(1e-30, 2.0 * F.sine_power(p, k - LOBE, k + LOBE, size, wsq))), keep))
    loudest = max((r[1] for r in rows if r[0] is not None), default=None)
    pitched = [r[0] is not None and r[1] >= loudest - RANGE_DB for r in rows] if loudest is not None else \
        [False] * len(rows)
    track = [r[0] if ok else None for r, ok in zip(rows, pitched, strict=True)]
    step_s = hop / float(rate)
    out = {"frameSamples": size, "frameMs": round(1000.0 * size / rate, 2), "binHz": round(bin_hz, 3),
           "stepMs": round(1000.0 * step_s, 3), "minHz": o["minHz"], "maxHz": o["maxHz"], "frames": len(rows),
           "voicedFrames": sum(1 for v in track if v is not None)}
    ka = kmin + max(range(kmax - kmin + 1), key=lambda i: avg[kmin + i])
    out["dominantHz"] = round((ka + _parabola(avg, ka)) * bin_hz, 2)
    # harmonics, each widened by how far it sweeps across its frame
    frame_s = size / float(rate)
    acc = {}
    for i, f0 in enumerate(track):
        if f0 is None:
            continue
        nb = [track[j] for j in (i - 1, i + 1) if 0 <= j < len(track) and track[j] is not None]
        if len(nb) == 2:
            slope = abs(nb[1] - nb[0]) / (2.0 * step_s)          # Hz per second through this frame
        else:
            slope = abs(nb[0] - f0) / step_s if nb else 0.0
        cap = max(1, int(f0 / bin_hz / 2.0))
        for h, (k0, sl) in rows[i][2].items():
            w = min(cap, LOBE + int(math.ceil(h * slope * frame_s / bin_hz / 2.0)), LOBE + MAX_SPREAD)
            c = int(round(h * f0 / bin_hz)) - k0
            power = 2.0 * sum(sl[max(0, c - w):c + w + 1]) / (size * wsq)
            a = acc.setdefault(h, [0.0, 0.0, 0])
            a[0] += power
            a[1] += h * f0
            a[2] += 1
    out["harmonics"] = [{"n": h, "hz": round(a[1] / a[2], 2), "db": round(10.0 * math.log10(max(1e-30, 2.0 * a[0] /
                                                                                              a[2])), 2)}
                        for h, a in sorted(acc.items())]
    smooth = _median3(track)
    voiced = [i for i, v in enumerate(smooth) if v is not None]
    out["lowHz"] = out["highHz"] = None
    if voiced:
        lo = min(voiced, key=lambda i: smooth[i])
        hi = max(voiced, key=lambda i: smooth[i])
        out["lowHz"] = round(_refine(mono, rate, starts[lo] + size // 2, hop, smooth[lo], -1), 2)
        out["highHz"] = round(_refine(mono, rate, starts[hi] + size // 2, hop, smooth[hi], 1), 2)
    ups = downs = 0
    for a, b in zip(smooth, smooth[1:], strict=False):
        if a is not None and b is not None and abs(b - a) > 0.1:
            ups += b > a
            downs += b < a
    out["riseFraction"] = round(ups / float(ups + downs), 3) if ups + downs >= 2 else None
    out["cycleSeconds"], why = _cycle(smooth, step_s, bin_hz)
    if why:
        out["cycleNote"] = why
    out["pitchTrack"] = [{"t": round((s + size / 2.0) / rate, 4), "hz": round(v, 1) if v is not None else None}
                         for s, v in zip(starts, track, strict=True)]
    out["about"] = ("pitch is the strongest partial between minHz and maxHz, null where a frame has none; harmonics "
                    "are multiples of it, levels in dBFS of a sine")
    return out, None


def _refine(mono, rate, centre, span, coarse, sign):
    """The pitch's extreme near `centre`, read with ~20 ms frames slid 1 ms at a time over +/- span samples. A long
    frame reads the average pitch across it, so at a sharp turn - a siren's top - it falls short of the apex by the
    sweep across a quarter frame (11 Hz on a 970-1630 Hz wail); a short frame hugs it. Kept to the partial the long
    frame found (+/-10 %), and to partials at least four short bins up, where a short frame still resolves them."""
    size = F.next_pow2(int(math.ceil(rate * 0.02)))
    bin_hz = rate / float(size)
    if coarse < 4 * bin_hz:
        return coarse
    win = F.hann(size)
    k_lo = max(1, int(coarse * 0.9 / bin_hz))
    k_hi = min(size // 2 - 1, int(math.ceil(coarse * 1.1 / bin_hz)))
    best = coarse
    for c in range(centre - span, centre + span + 1, max(1, rate // 1000)):
        s = c - size // 2
        if s < 0 or s + size > len(mono):
            continue
        p = [abs(v) ** 2 for v in F.rfft([mono[s + i] * win[i] for i in range(size)])]
        k = max(range(k_lo, k_hi + 1), key=p.__getitem__)
        f = (k + _parabola(p, k)) * bin_hz
        if (f - best) * sign > 0:
            best = f
    return best


def _cycle(track, step_s, bin_hz):
    voiced = [v for v in track if v is not None]
    if len(voiced) < 8:
        return None, "too few pitched frames to look for a cycle."
    mean = sum(voiced) / len(voiced)
    d = [(v - mean) if v is not None else 0.0 for v in track]
    if max(abs(v) for v in d) < 0.25 * bin_hz:
        return None, "the pitch holds steady, so it has no cycle."
    n = len(d)
    r = []
    for lag in range(0, (2 * n) // 3 + 1):
        a, b = d[:n - lag], d[lag:]
        den = math.sqrt(sum(v * v for v in a) * sum(v * v for v in b))
        r.append(sum(x * y for x, y in zip(a, b, strict=True)) / den if den > 0 else 0.0)
    gone = False
    for lag in range(1, len(r) - 1):
        if r[lag] < 0:
            gone = True
        if gone and r[lag] >= 0.5 and r[lag] >= r[lag - 1] and r[lag] >= r[lag + 1]:
            den = r[lag - 1] - 2.0 * r[lag] + r[lag + 1]
            off = 0.5 * (r[lag - 1] - r[lag + 1]) / den if den < 0 else 0.0
            return round((lag + off) * step_s, 3), None
    return None, ("no repeating cycle in the pitch within the file - a cycle shows only when about one and a half of "
                  "them are in it.")
