"""Measure a WAV file: loudness per ITU-R BS.1770-4 / EBU R 128, loudness range per EBU Tech 3342, sample peak, RMS,
crest, and where the energy sits (lows < 250 Hz, mids, highs > 4 kHz). Standard library only - the MCP server
declares no numpy - so every filter is a biquad run in plain Python, on array('d') buffers.

  integrated loudness  K-weighting (the BS.1770 pre-filter shelf and RLB high-pass, coefficients for ANY sample rate
                       as libebur128 derives them), 400 ms blocks every 100 ms, absolute gate -70 LUFS, relative gate
                       -10 LU. Channel weights 1.0 for up to three channels, 1.41 for surrounds.
  loudness range       3 s short-term windows every 100 ms, gates -70 LUFS and -20 LU, 95th minus 10th percentile.
  over time            EBU Tech 3341's momentary (0.4 s) and short-term (3 s) loudness, ungated, on the same 100 ms
                       sub-blocks (the Tech's minimum 10 Hz update): their maxima always, and a curve of either on
                       request, each point at the END of its window, as a meter shows it then.
  gated                the integrated loudness over the 400 ms blocks a gate keeps - blocks wholly inside given
                       regions, or blocks where a stem (the dialogue alone) passes its own BS.1770 gates - so a
                       speech-gated reading of a full mix needs no edit of the mix.
  spectrum             on request, mif_audio.spectrum: a pitch track, harmonic levels, the pitch's low, high, rise and
                       cycle, through the one shared FFT (mif_audio.fft).
  a mono file          is ONE channel, as BS.1770 defines it: a 0 dBFS 997 Hz sine reads -3.01 LUFS, not 0. Some meters
                       treat mono as dual-mono and read 3 dB higher; this one says which it did (monoAsDualMono false).
  true peak            BS.1770-4 Annex 2: the largest magnitude of the signal oversampled 4x below 96 kHz, 2x below
                       192 kHz. The interpolator is derived here, not copied: a 24-tap Kaiser-windowed sinc (beta 6)
                       per fractional position, each phase scaled to unity gain - within 0.015 dB of a perfect
                       interpolator from DC to 20 kHz at 48 kHz, computed from its own taps (16 taps at beta 8 were
                       1.2 dB out at 20 kHz, past EBU R 128's +/-0.3 dB meter tolerance). A 12 kHz sine sampled at 45
                       degrees - every sample at 0.707 of the crest - reads its true crest, 3.01 dB over its sample peak.

self_test() checks the math on signals with known answers; tools that report loudness run it once first.
"""
import array
import math
import os
import struct
import sys

LOW_HZ, HIGH_HZ = 250.0, 4000.0


# ---------------------------------------------------------------------------------------------------------------------
# reading

def read_wav(path):
    """-> (rate, [array('d') per channel], info). PCM 8/16/24/32-bit integer and 32/64-bit float, plain or
    WAVE_FORMAT_EXTENSIBLE. Raises ValueError naming what it met."""
    with open(path, "rb") as fh:
        data = fh.read()
    if len(data) < 12 or data[:4] not in (b"RIFF", b"RF64") or data[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")
    pos, fmt, raw = 12, None, None
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], struct.unpack_from("<I", data, pos + 4)[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            fmt = body
        elif cid == b"data":
            raw = body
        pos += 8 + size + (size & 1)
    if fmt is None or raw is None:
        raise ValueError("no fmt or data chunk")
    tag, channels, rate, _, align, bits = struct.unpack_from("<HHIIHH", fmt, 0)
    if tag == 0xFFFE and len(fmt) >= 26:
        tag = struct.unpack_from("<H", fmt, 24)[0]
    if tag not in (1, 3) or channels < 1:
        raise ValueError("unsupported WAV encoding (format tag %d, %d channels)" % (tag, channels))
    width = bits // 8
    frames = len(raw) // (width * channels)
    if tag == 3:
        code = {4: "f", 8: "d"}.get(width)
        if not code:
            raise ValueError("unsupported float width %d bits" % bits)
        flat = array.array(code, raw[:frames * width * channels])
        scale = 1.0
    elif width == 3:
        flat = array.array("d", [0.0]) * (frames * channels)
        mv = memoryview(raw)
        for i in range(frames * channels):
            b = mv[3 * i:3 * i + 3]
            v = b[0] | (b[1] << 8) | (b[2] << 16)
            flat[i] = v - 16777216 if v & 0x800000 else v
        scale = 1.0 / 8388608.0
    else:
        code = {1: "B", 2: "h", 4: "i"}.get(width)
        if not code:
            raise ValueError("unsupported PCM width %d bits" % bits)
        flat = array.array(code, raw[:frames * width * channels])
        scale = 1.0 / {1: 128.0, 2: 32768.0, 4: 2147483648.0}[width]
    out = []
    for c in range(channels):
        ch = array.array("d", flat[c::channels])
        if width == 1 and tag == 1:
            ch = array.array("d", ((v - 128) * scale for v in ch))
        elif scale != 1.0:
            ch = array.array("d", (v * scale for v in ch))
        out.append(ch)
    return rate, out, {"format": "float" if tag == 3 else "pcm", "bits": bits, "channels": channels, "frames": frames}


# ---------------------------------------------------------------------------------------------------------------------
# filters

def biquad(x, b, a):
    """Direct form II transposed. b = (b0, b1, b2), a = (1, a1, a2)."""
    b0, b1, b2 = b
    a1, a2 = a[1], a[2]
    z1 = z2 = 0.0
    y = array.array("d", bytes(8 * len(x)))
    for i, v in enumerate(x):
        o = b0 * v + z1
        z1 = b1 * v - a1 * o + z2
        z2 = b2 * v - a2 * o
        y[i] = o
    return y


def k_weighting(rate):
    """The two BS.1770 stages for any sample rate, derived as libebur128 does (it reproduces the standard's 48 kHz
    table): a high-shelf pre-filter near 1.68 kHz, +4 dB, and the RLB high-pass near 38 Hz."""
    f0, gain, q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    k = math.tan(math.pi * f0 / rate)
    vh = 10.0 ** (gain / 20.0)
    vb = vh ** 0.4996667741545416
    a0 = 1.0 + k / q + k * k
    shelf = ((vh + vb * k / q + k * k) / a0, 2.0 * (k * k - vh) / a0, (vh - vb * k / q + k * k) / a0), \
        (1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0)
    f0, q = 38.13547087602444, 0.5003270373238773
    k = math.tan(math.pi * f0 / rate)
    a0 = 1.0 + k / q + k * k
    highpass = (1.0, -2.0, 1.0), (1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0)
    return shelf, highpass


def butter2(kind, fc, rate):
    """A 2nd-order Butterworth low- or high-pass (RBJ cookbook, Q = 1/sqrt 2)."""
    w = 2.0 * math.pi * fc / rate
    cw, alpha = math.cos(w), math.sin(w) / math.sqrt(2.0)
    a0 = 1.0 + alpha
    if kind == "low":
        b = ((1 - cw) / 2 / a0, (1 - cw) / a0, (1 - cw) / 2 / a0)
    else:
        b = ((1 + cw) / 2 / a0, -(1 + cw) / a0, (1 + cw) / 2 / a0)
    return b, (1.0, -2.0 * cw / a0, (1 - alpha) / a0)


# ---------------------------------------------------------------------------------------------------------------------
# measures

def _db(v, floor=-200.0):
    return 10.0 * math.log10(v) if v > 0 else floor


def _sub_block_energy(weighted, rate, hop):
    """Mean square per 100 ms sub-block, per channel."""
    n = len(weighted[0]) // hop
    return [[sum(v * v for v in ch[i * hop:(i + 1) * hop]) / hop for i in range(n)] for ch in weighted]


def _gated(loudness_blocks, energy_blocks, relative_lu):
    above = [e for lu, e in zip(loudness_blocks, energy_blocks, strict=True) if lu > -70.0]
    if not above:
        return None, []
    rel = _db(sum(above) / len(above)) - 0.691 - relative_lu
    kept = [(lu, e) for lu, e in zip(loudness_blocks, energy_blocks, strict=True) if lu > -70.0 and lu > rel]
    return rel, kept


CHANNEL_WEIGHTS = (1.0, 1.0, 1.0, 1.41, 1.41)   # BS.1770-4 Table 3: L, R, C 1.0; Ls, Rs 1.41
MOMENTARY_SUB, SHORT_TERM_SUB = 4, 30            # EBU Tech 3341: M over 0.4 s, S over 3 s, in 100 ms sub-blocks
CURVE_MAX_POINTS = 3000


def sub_blocks(rate, channels):
    """The channel-weighted mean square of the K-weighted signal per 100 ms sub-block - the quantity every BS.1770 and
    Tech 3341 window averages (a window's loudness is -0.691 + 10 log10 of the mean of its sub-blocks)."""
    shelf, highpass = k_weighting(rate)
    weighted = [biquad(biquad(ch, *shelf), *highpass) for ch in channels]
    weights = list(CHANNEL_WEIGHTS[:len(channels)]) + [1.0] * max(0, len(channels) - 5)
    hop = int(round(rate * 0.1))
    sub = _sub_block_energy(weighted, rate, hop)
    n = len(sub[0]) if sub else 0
    return [sum(weights[c] * sub[c][i] for c in range(len(sub))) for i in range(n)]


def windows(z, size):
    """-> (loudness per window, energy per window) for every window of `size` sub-blocks, one per 100 ms step."""
    run = [0.0]
    for v in z:
        run.append(run[-1] + v)
    e = [max(0.0, (run[i + size] - run[i]) / size) for i in range(0, len(z) - size + 1)]
    return [(-0.691 + _db(v)) for v in e], e


def integrated(lk, zk):
    """BS.1770-4 gated integration over the given 400 ms blocks: -70 LUFS absolute, then -10 LU relative."""
    _, kept = _gated(lk, zk, 10.0)
    return round(-0.691 + _db(sum(e for _, e in kept) / len(kept)), 2) if kept else None


def _max_lufs(ek):
    best = max(ek, default=0.0)
    return round(-0.691 + _db(best), 2) if best > 0 else None


def loudness(rate, channels, z=None):
    """-> {integratedLufs, loudnessRangeLu, blocks, maxMomentaryLufs, maxShortTermLufs} per BS.1770-4, Tech 3342 and
    Tech 3341. None where a gate leaves nothing, or where the file is shorter than the window."""
    z = sub_blocks(rate, channels) if z is None else z
    nsub = len(z)
    out = {"integratedLufs": None, "loudnessRangeLu": None, "blocks": 0, "maxMomentaryLufs": None,
           "maxShortTermLufs": None}
    if nsub >= MOMENTARY_SUB:
        lk, zk = windows(z, MOMENTARY_SUB)
        out["blocks"] = len(lk)
        out["integratedLufs"] = integrated(lk, zk)
        out["maxMomentaryLufs"] = _max_lufs(zk)
    if nsub >= SHORT_TERM_SUB:
        ls, zs = windows(z, SHORT_TERM_SUB)
        out["maxShortTermLufs"] = _max_lufs(zs)
        _, kept = _gated(ls, zs, 20.0)
        if len(kept) >= 2:
            vals = sorted(lu for lu, _ in kept)
            lo = vals[int(round(0.10 * (len(vals) - 1)))]
            hi = vals[int(round(0.95 * (len(vals) - 1)))]
            out["loudnessRangeLu"] = round(hi - lo, 2)
    else:
        out["shortTermNote"] = ("the file is shorter than the 3 s short-term window, so it has no short-term loudness "
                                "and no loudness range.")
    return out


def _bad(msg):
    return {"ok": False, "error": msg + " Nothing was measured."}


def loudness_curve(z, curve):
    """curve {window MOMENTARY|SHORT_TERM, stepMs (a multiple of 100, default 100), maxPoints} -> (dict, None) or
    (None, error). Each point's t is the END of its window, as a meter shows it at that moment; lufs null = silence."""
    if not isinstance(curve, dict):
        return None, "curve must be an object {window MOMENTARY|SHORT_TERM, stepMs}, got %r." % (curve,)
    unknown = sorted(set(curve) - {"window", "stepMs", "maxPoints"})
    if unknown:
        return None, "curve has no field %s - it takes window, stepMs and maxPoints." % ", ".join(unknown)
    window = curve.get("window", "MOMENTARY")
    size = {"MOMENTARY": MOMENTARY_SUB, "SHORT_TERM": SHORT_TERM_SUB}.get(window)
    if size is None:
        return None, "curve.window must be MOMENTARY (400 ms) or SHORT_TERM (3 s), got %r." % (window,)
    step = curve.get("stepMs", 100)
    if isinstance(step, bool) or not isinstance(step, (int, float)) or step < 100 or step % 100:
        return None, ("curve.stepMs must be a multiple of 100 (ms) - the meter's sub-block is 100 ms, Tech 3341's "
                      "10 Hz update - got %r." % (step,))
    cap = curve.get("maxPoints", CURVE_MAX_POINTS)
    if isinstance(cap, bool) or not isinstance(cap, int) or not 1 <= cap <= CURVE_MAX_POINTS:
        return None, "curve.maxPoints must be an integer 1-%d, got %r." % (CURVE_MAX_POINTS, cap)
    k = int(step) // 100
    if len(z) < size:
        return None, "the file is shorter than the %d ms %s window, so it has no curve of it." % (size * 100, window)
    lk, ek = windows(z, size)
    starts = list(range(0, len(lk), k))
    if len(starts) > cap:
        need = 100 * -(-len(lk) // cap)
        return None, ("a %s curve at stepMs %d has %d points, over maxPoints %d - give stepMs %d or more."
                      % (window, step, len(starts), cap, need))
    pts = [{"t": round((i + size) / 10.0, 3), "lufs": round(lk[i], 2) if ek[i] > 0 else None} for i in starts]
    return {"window": window, "windowMs": size * 100, "stepMs": int(step), "points": pts}, None


def _stem_activity(stem_path, rate, nblocks, threshold):
    """-> (a bool per 400 ms block, the threshold used, None) or (None, None, error). Without thresholdLufs a block
    is active when the stem's own block passes the stem's BS.1770 gates (-70 LUFS, then -10 LU under its mean)."""
    if not isinstance(stem_path, str) or not os.path.isfile(stem_path):
        return None, None, "gate.stem: no file at %r." % (stem_path,)
    try:
        srate, schans, _ = read_wav(stem_path)
    except (ValueError, struct.error, OSError) as exc:
        return None, None, "gate.stem: could not read %s as WAV: %s." % (stem_path, exc)
    if srate != rate:
        return None, None, ("gate.stem is at %d Hz and the file at %d Hz - the stem must be the same mix's stem, "
                            "at its rate." % (srate, rate))
    zs = sub_blocks(srate, schans)
    if len(zs) < MOMENTARY_SUB:
        return None, None, "gate.stem is shorter than one 400 ms block."
    ls, es = windows(zs, MOMENTARY_SUB)
    if threshold is None:
        rel, kept = _gated(ls, es, 10.0)
        if not kept:
            return None, None, ("gate.stem is silent - no 400 ms block of it rose above -70 LUFS, so it marks no "
                                "span to measure.")
        threshold = max(-70.0, rel)
    elif isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        return None, None, "gate.thresholdLufs must be a number, got %r." % (threshold,)
    active = [i < len(ls) and ls[i] > threshold for i in range(nblocks)]
    if not any(active):
        return None, None, ("gate.stem is silent above thresholdLufs %.1f - no block of it rises past it, so it "
                            "marks no span to measure." % threshold)
    return active, threshold, None


def gated_measure(z, rate, gate):
    """gate {regions [[start, end]...] | stem path (+ thresholdLufs)} -> (dict, None) or (None, error): the BS.1770
    integrated loudness and the maximum momentary over the 400 ms blocks the gate keeps."""
    if not isinstance(gate, dict):
        return None, "gate must be an object {regions [[start, end]...]} or {stem: path}, got %r." % (gate,)
    unknown = sorted(set(gate) - {"regions", "stem", "thresholdLufs"})
    if unknown:
        return None, "gate has no field %s - it takes regions, or stem and thresholdLufs." % ", ".join(unknown)
    has_r, has_s = gate.get("regions") is not None, gate.get("stem") is not None
    if has_r and has_s:
        return None, "gate takes regions or stem, not both."
    if not has_r and not has_s:
        return None, "gate needs regions [[start, end]...] (seconds) or stem (a WAV of the part to measure)."
    if len(z) < MOMENTARY_SUB:
        return None, "the file is shorter than one 400 ms block, so there is nothing to gate."
    lk, ek = windows(z, MOMENTARY_SUB)
    n = len(lk)
    out = {}
    if has_r:
        regions = gate["regions"]
        if not isinstance(regions, list) or not regions:
            return None, "gate.regions must be a non-empty list of [start, end] in seconds, got %r." % (regions,)
        spans = []
        for r in regions:
            if (not isinstance(r, (list, tuple)) or len(r) != 2
                    or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in r)):
                return None, "each gate region is [start, end] in seconds, got %r." % (r,)
            if r[0] < 0 or r[1] <= r[0]:
                return None, "a gate region's end must be after its start, and its start 0 or more - got %r." % (r,)
            spans.append((float(r[0]), float(r[1])))
        # a block starting at i / 10 s covers [i / 10, i / 10 + 0.4]; it counts when one region holds all of it
        keep = [any(a - 1e-9 <= i / 10.0 and i / 10.0 + 0.4 <= b + 1e-9 for a, b in spans) for i in range(n)]
        if not any(keep):
            return None, ("no 400 ms block lies wholly inside the regions %s (the file runs %.1f s) - a region "
                          "shorter than 400 ms, or past the end, holds none." % (spans, (n + 3) / 10.0))
        out["by"] = "regions"
    else:
        keep, thr, err = _stem_activity(gate["stem"], rate, n, gate.get("thresholdLufs"))
        if err:
            return None, err
        out["by"] = "stem"
        out["stemThresholdLufs"] = round(thr, 2)
    lk2 = [v for v, k in zip(lk, keep, strict=True) if k]
    ek2 = [v for v, k in zip(ek, keep, strict=True) if k]
    out.update({"integratedLufs": integrated(lk2, ek2), "maxMomentaryLufs": _max_lufs(ek2),
                "blocks": len(lk2), "blocksTotal": n})
    if out["integratedLufs"] is None:
        return None, ("the %d blocks the gate kept are all under -70 LUFS - the gated part of the file is silent."
                      % len(lk2))
    return out, None


TP_HALF_TAPS = 12         # each interpolated value reads the 12 samples either side of its gap
TP_BETA = 6.0             # the Kaiser window's shape (see the module docstring for the error it leaves)
_TP_PHASES = {}


def _bessel_i0(x):
    s, term, k = 1.0, 1.0, 1
    while term > 1e-15 * s:
        term *= (x / (2.0 * k)) ** 2
        s += term
        k += 1
    return s


def tp_factor(rate):
    """The oversampling BS.1770-4 Annex 2 asks for: 4x at 48 kHz, 2x at 96 kHz, none from 192 kHz."""
    return 4 if rate < 96000 else 2 if rate < 192000 else 1


def tp_phases(factor):
    """The interpolator for each fractional position p / factor between two samples, p = 1 .. factor - 1: taps for
    the samples i-11 .. i+12 around the gap (i, i+1), a Kaiser-windowed sinc, each phase scaled to sum to 1."""
    if factor not in _TP_PHASES:
        i0b = _bessel_i0(TP_BETA)
        phases = []
        for p in range(1, factor):
            f = p / float(factor)
            taps = []
            for k in range(-TP_HALF_TAPS + 1, TP_HALF_TAPS + 1):
                t = f - k
                sinc = math.sin(math.pi * t) / (math.pi * t)
                r = t / TP_HALF_TAPS
                taps.append(sinc * _bessel_i0(TP_BETA * math.sqrt(max(0.0, 1.0 - r * r))) / i0b)
            s = sum(taps)
            phases.append([c / s for c in taps])
        _TP_PHASES[factor] = phases
    return _TP_PHASES[factor]


def true_peak(rate, channels):
    """-> (true peak as a linear magnitude, 0.0 for silence). Only the gaps that CAN beat the sample peak are
    interpolated: a value between two samples is at most sum|taps| times the largest sample in its 24-sample window,
    so a window whose samples all sit below sample_peak / max(sum|taps|) cannot beat the sample peak and is skipped.
    That bound is exact, so skipping changes no answer - it only keeps plain Python fast on real audio."""
    peak = max((max(abs(v) for v in ch) for ch in channels), default=0.0)
    factor = tp_factor(rate)
    if peak <= 0.0 or factor == 1:
        return peak
    phases = tp_phases(factor)
    floor = peak / max(sum(abs(c) for c in ph) for ph in phases)
    best = peak
    h = TP_HALF_TAPS
    for ch in channels:
        n = len(ch)
        gaps = bytearray(n)
        for j, v in enumerate(ch):
            if v >= floor or v <= -floor:
                # the gaps whose window (i-11 .. i+12) holds sample j
                lo, hi = max(0, j - h), min(n - 1, j + h)
                gaps[lo:hi] = b"\x01" * (hi - lo)
        for i in (k for k in range(n - 1) if gaps[k]):
            lo = i - h + 1
            window = [ch[j] if 0 <= j < n else 0.0 for j in range(lo, lo + 2 * h)]
            for ph in phases:
                y = abs(sum(c * x for c, x in zip(ph, window)))
                if y > best:
                    best = y
    return best


def band_shares(rate, channels):
    """Energy shares below LOW_HZ, between, above HIGH_HZ - two cascaded Butterworth stages (4th order) each side,
    on the channel average. Shares of a filter bank, so they sum to about 100, not exactly."""
    n = len(channels[0])
    mono = array.array("d", (sum(ch[i] for ch in channels) / len(channels) for i in range(n)))
    total = sum(v * v for v in mono)
    if total <= 0:
        return None
    lp = butter2("low", LOW_HZ, rate)
    hp = butter2("high", HIGH_HZ, rate)
    low = biquad(biquad(mono, *lp), *lp)
    high = biquad(biquad(mono, *hp), *hp)
    lows = 100.0 * sum(v * v for v in low) / total
    highs = 100.0 * sum(v * v for v in high) / total
    return {"lows": round(lows, 2), "mids": round(max(0.0, 100.0 - lows - highs), 2), "highs": round(highs, 2)}


def measure(path, curve=None, gate=None, spectrum=None):
    """Everything above for one file, as a dict with ok:True, or ok:False and the reason. curve adds a loudness curve
    (loudness_curve), gate a loudness over the gate's spans (gated_measure), spectrum a pitch track, harmonics and
    cycle (mif_audio.spectrum); a refusal of any of them measures nothing."""
    if not os.path.isfile(path):
        return {"ok": False, "error": "no file at %s" % path}
    try:
        rate, channels, info = read_wav(path)
    except (ValueError, struct.error, OSError) as exc:
        return {"ok": False, "error": "could not read %s as WAV: %s. FLAC and compressed files are not read here - "
                                      "render or export a WAV first." % (path, exc)}
    n = len(channels[0])
    if n == 0:
        return {"ok": False, "error": "%s holds no samples" % path}
    z = sub_blocks(rate, channels)
    extra = {}
    if curve is not None:
        extra["curve"], err = loudness_curve(z, curve)
        if err:
            return _bad(err)
    if gate is not None:
        extra["gated"], err = gated_measure(z, rate, gate)
        if err:
            return _bad(err)
    if spectrum is not None and spectrum is not False:
        from mif_audio import spectrum as SPECTRUM
        extra["spectrum"], err = SPECTRUM.analyze(rate, channels, spectrum)
        if err:
            return _bad(err)
    peak = max(max(abs(v) for v in ch) for ch in channels)
    ms = sum(sum(v * v for v in ch) for ch in channels) / (n * len(channels))
    tp = true_peak(rate, channels)
    out = {"ok": True, "path": path, "rate": rate, "channels": info["channels"], "bits": info["bits"],
           "encoding": info["format"], "duration_s": round(n / float(rate), 4),
           "samplePeakDbfs": round(20.0 * math.log10(peak), 2) if peak > 0 else None,
           "truePeakDbtp": round(20.0 * math.log10(tp), 2) if tp > 0 else None,
           "rmsDbfs": round(_db(ms), 2) if ms > 0 else None,
           "crestDb": round(20.0 * math.log10(peak) - _db(ms), 2) if peak > 0 and ms > 0 else None,
           "monoAsDualMono": False}
    out.update(loudness(rate, channels, z))
    out.update(extra)
    out["bands"] = band_shares(rate, channels)
    if out["integratedLufs"] is None:
        out["note"] = ("no 400 ms block rose above the -70 LUFS gate (or the file is shorter than 0.4 s), so there "
                       "is no integrated loudness - the file is silent or too short, not quiet.")
    return out


def write_wav(path, rate, channels, bits=16):
    """Integer PCM, 16 or 24 bits, samples clipped to [-1, 1]; or bits 32, IEEE float (WAVE_FORMAT_IEEE_FLOAT with its
    fact chunk), not clipped. Used by the tests and the renderers."""
    import wave
    n = len(channels[0])
    nch = len(channels)
    if bits == 32:
        buf = array.array("f", bytes(4 * n * nch))
        for c, ch in enumerate(channels):
            for i, v in enumerate(ch):
                buf[i * nch + c] = v
        if sys.byteorder != "little":
            buf.byteswap()
        data = buf.tobytes()
        fmt = struct.pack("<HHIIHHH", 3, nch, rate, rate * nch * 4, nch * 4, 32, 0)
        body = (b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"fact" + struct.pack("<II", 4, n)
                + b"data" + struct.pack("<I", len(data)) + data + (b"\0" if len(data) & 1 else b""))
        with open(path, "wb") as fh:
            fh.write(b"RIFF" + struct.pack("<I", len(body)) + body)
        return
    if bits == 16:
        buf = array.array("h", bytes(2 * n * nch))
        for c, ch in enumerate(channels):
            for i, v in enumerate(ch):
                buf[i * nch + c] = int(max(-1.0, min(1.0, v)) * 32767.0)
        payload = buf.tobytes()
    elif bits == 24:
        out = bytearray(3 * n * nch)
        for c, ch in enumerate(channels):
            for i, v in enumerate(ch):
                s = int(max(-1.0, min(1.0, v)) * 8388607.0) & 0xFFFFFF
                j = 3 * (i * nch + c)
                out[j:j + 3] = bytes((s & 0xFF, (s >> 8) & 0xFF, (s >> 16) & 0xFF))
        payload = bytes(out)
    else:
        raise ValueError("bits must be 16, 24 or 32 (float)")
    with wave.open(path, "wb") as w:
        w.setnchannels(nch)
        w.setsampwidth(bits // 8)
        w.setframerate(rate)
        w.writeframes(payload)


def self_test(tmp_dir):
    """Known answers. Raises AssertionError naming the first that fails; returns the measured values."""
    rate = 48000
    tone = array.array("d", (0.1 * math.sin(2 * math.pi * 997 * i / rate) for i in range(rate * 5)))
    got = {}
    p = os.path.join(tmp_dir, "_mif_measure_selftest_stereo.wav")
    write_wav(p, rate, [tone, tone], bits=24)
    m = measure(p)
    got["stereo_tone_-20dBFS_peak"] = m["integratedLufs"]
    assert m["ok"] and abs(m["integratedLufs"] - (-20.0)) < 0.1, m        # BS.1770: both channels, -20 LUFS
    assert abs(m["samplePeakDbfs"] - (-20.0)) < 0.05 and abs(m["crestDb"] - 3.01) < 0.05, m
    assert m["bands"]["mids"] > 95, m                                    # 997 Hz is a mid
    p1 = os.path.join(tmp_dir, "_mif_measure_selftest_mono.wav")
    write_wav(p1, rate, [tone], bits=16)
    m1 = measure(p1)
    got["mono_tone_-20dBFS_peak"] = m1["integratedLufs"]
    assert abs(m1["integratedLufs"] - (-23.01)) < 0.12, m1              # one channel: 3.01 dB lower
    quiet = array.array("d", bytes(8 * rate))
    p2 = os.path.join(tmp_dir, "_mif_measure_selftest_silent.wav")
    write_wav(p2, rate, [quiet], bits=16)
    m2 = measure(p2)
    assert m2["ok"] and m2["integratedLufs"] is None and "note" in m2, m2  # silence has NO loudness, not a number
    low = array.array("d", (0.3 * math.sin(2 * math.pi * 80 * i / rate) for i in range(rate)))
    p3 = os.path.join(tmp_dir, "_mif_measure_selftest_low.wav")
    write_wav(p3, rate, [low], bits=16)
    assert measure(p3)["bands"]["lows"] > 90                               # 80 Hz is a low
    # TRUE PEAK: a 12 kHz sine (fs / 4) sampled at 45 degrees puts every sample at 0.707 of its crest, so its sample
    # peak is 3.01 dB under its true peak. Crest 0.5: sample peak -9.03 dBFS, true peak -6.02 dBTP. FADED IN AND OUT
    # over 100 ms: a sine that starts on a step is not band-limited, and every interpolator rings over that step
    # (measured: +0.11 dB at the edge of an unfaded one).
    ramp = rate // 10

    def fade(i):
        return min(1.0, i / float(ramp), (rate - 1 - i) / float(ramp))
    quarter = array.array("d", (0.5 * fade(i) * math.sin(2 * math.pi * i / 4 + math.pi / 4) for i in range(rate)))
    tp = 20.0 * math.log10(true_peak(rate, [quarter]))
    sp = 20.0 * math.log10(max(abs(v) for v in quarter))
    got["quarter_rate_true_peak"] = round(tp, 3)
    assert abs(sp - (-9.03)) < 0.01 and abs(tp - (-6.02)) < 0.05, (sp, tp)
    assert abs(m["truePeakDbtp"] - (-20.0)) < 0.05, m                        # a 997 Hz tone: true peak = its crest
    for q in (p, p1, p2, p3):
        os.remove(q)
    return got
