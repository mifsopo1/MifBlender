"""One pure-Python FFT for every mif_audio module that needs a spectrum - the MCP server declares no numpy.

  fft / ifft      iterative radix-2 Cooley-Tukey on a list of complex numbers, length a power of two; ifft is
                  conj(fft(conj X)) / n, so X -> ifft(fft(X)) returns X.
  rfft / irfft    a real frame of n samples as one complex FFT of n / 2 (even samples real, odd samples imaginary),
                  then split - the standard packing, about twice as fast as a complex FFT of the real frame. rfft
                  returns bins 0 .. n / 2; irfft takes them back to n real samples.
  hann            the periodic Hann window, the one whose shifted copies at a half or a quarter frame sum to a
                  constant (the property an overlap-add resynthesis needs).
  sine_power      the mean square of a sinusoid read off the bins around its peak: 2 sum |X_k|^2 / (n sum w^2), by
                  Parseval - a full-scale sine reads 0.5 (0 dB as 10 log10 of twice it) wherever it falls between
                  bins, which a peak bin's height alone does not (the Hann window's scalloping is 1.42 dB).

self_test() checks fft and rfft against a direct DFT, the round trips, Parseval and the window's overlap sums.
"""
import cmath
import math

_PLANS = {}


def _plan(n):
    if n < 1 or n & (n - 1):
        raise ValueError("FFT length must be a power of two, got %d" % n)
    if n not in _PLANS:
        bits = n.bit_length() - 1
        rev = [int(format(i, "0%db" % bits)[::-1], 2) if bits else 0 for i in range(n)]
        tw = [cmath.exp(-2j * math.pi * k / n) for k in range(max(1, n // 2))]
        _PLANS[n] = (rev, tw)
    return _PLANS[n]


def fft(x):
    """-> list of n complex bins of the sequence x (any numbers), n a power of two."""
    n = len(x)
    rev, tw = _plan(n)
    a = [complex(x[r]) for r in rev]
    size = 2
    while size <= n:
        half, step = size // 2, n // size
        for k in range(half):
            w = tw[k * step]
            for j in range(k, n, size):
                t = w * a[j + half]
                a[j + half] = a[j] - t
                a[j] += t
        size *= 2
    return a


def ifft(x):
    n = len(x)
    return [v.conjugate() / n for v in fft([complex(v).conjugate() for v in x])]


def rfft(x):
    """-> bins 0 .. n / 2 of a real sequence of n samples (n a power of two, 2 or more)."""
    n = len(x)
    m = n // 2
    if m < 1:
        raise ValueError("rfft needs at least 2 samples")
    z = fft([complex(x[2 * i], x[2 * i + 1]) for i in range(m)])
    tw = _plan(n)[1]
    out = [0j] * (m + 1)
    for k in range(m + 1):
        zk, zc = z[k % m], z[(m - k) % m].conjugate()
        w = tw[k] if k < m else -1.0
        out[k] = 0.5 * (zk + zc) - 0.5j * w * (zk - zc)
    return out


def irfft(spec, n):
    """bins 0 .. n / 2 -> n real samples (the inverse of rfft)."""
    m = n // 2
    if len(spec) != m + 1:
        raise ValueError("irfft of %d samples takes %d bins, got %d" % (n, m + 1, len(spec)))
    tw = _plan(n)[1]
    z = []
    for k in range(m):
        xk, xc = spec[k], spec[m - k].conjugate()
        e = 0.5 * (xk + xc)
        o = 0.5 * (xk - xc) / tw[k]
        z.append(e + 1j * o)
    y = ifft(z)
    out = [0.0] * n
    for i, v in enumerate(y):
        out[2 * i], out[2 * i + 1] = v.real, v.imag
    return out


def hann(n):
    """Periodic Hann: 0.5 - 0.5 cos(2 pi i / n)."""
    return [0.5 - 0.5 * math.cos(2.0 * math.pi * i / n) for i in range(n)]


def sine_power(power_bins, k_lo, k_hi, n, window_sq_sum):
    """Mean square of the sinusoid whose energy lies in bins k_lo .. k_hi of an rfft's |X|^2 (see the docstring)."""
    lo, hi = max(0, k_lo), min(len(power_bins) - 1, k_hi)
    return 2.0 * sum(power_bins[lo:hi + 1]) / (n * window_sq_sum) if hi >= lo else 0.0


def next_pow2(v):
    n = 1
    while n < v:
        n *= 2
    return n


def self_test():
    """Known answers; raises AssertionError naming the first that fails."""
    import random
    rng = random.Random(7)
    for n in (1, 2, 8, 64):
        x = [complex(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(n)]
        direct = [sum(x[t] * cmath.exp(-2j * math.pi * k * t / n) for t in range(n)) for k in range(n)]
        got = fft(x)
        assert max(abs(a - b) for a, b in zip(got, direct, strict=True)) < 1e-9, ("fft vs DFT", n)
        assert max(abs(a - b) for a, b in zip(ifft(got), x, strict=True)) < 1e-12, ("ifft round trip", n)
    for n in (2, 16, 256):
        r = [rng.uniform(-1, 1) for _ in range(n)]
        full = fft(r)
        half = rfft(r)
        assert max(abs(a - b) for a, b in zip(half, full[:n // 2 + 1], strict=True)) < 1e-9, ("rfft vs fft", n)
        back = irfft(half, n)
        assert max(abs(a - b) for a, b in zip(back, r, strict=True)) < 1e-12, ("irfft round trip", n)
        assert abs(sum(abs(v) ** 2 for v in full) - n * sum(v * v for v in r)) < 1e-9 * n, ("Parseval", n)
    n = 1024
    w = hann(n)
    for hop in (n // 2, n // 4):
        sums = [sum(w[(i + j * hop) % n] for j in range(n // hop)) for i in range(hop)]
        assert max(sums) - min(sums) < 1e-12, ("Hann overlap-add", hop)
    wsq = sum(v * v for v in w)
    for frac in (0.0, 0.5, 0.25):                     # on a bin, half-way, a quarter: the same power
        f = (100 + frac) / n
        x = [math.sin(2 * math.pi * f * i) * w[i] for i in range(n)]
        p = [abs(v) ** 2 for v in rfft(x)]
        ms = sine_power(p, 97, 104, n, wsq)
        assert abs(10 * math.log10(2 * ms)) < 0.01, ("sine_power", frac, ms)
    return True
