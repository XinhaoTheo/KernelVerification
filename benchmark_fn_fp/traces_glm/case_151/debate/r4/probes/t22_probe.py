import torch, sys, json
import numpy as np
sys.path.insert(0, "/root/cases/case_151")
import kernel

Q = 3329
def mul(v, c):
    product = int(v) * int(c)
    correction = (product * 3327) & 65535
    reduced = (product + correction * Q) >> 16
    if reduced >= Q:
        reduced -= Q
    return reduced

def simulate(xrow, width, perm):
    # perm: function lane -> source index
    LOG_N = width.bit_length() - 1
    N = width
    coeffs = kernel._make_coefficients(width)
    lane = np.arange(N)
    value = np.array([int(xrow[perm(l)]) for l in lane], dtype=np.int64)
    for stage in range(LOG_N // 2):
        half = 1 << (2 * stage)
        local = lane % half
        base = (lane // (4 * half)) * (4 * half) + local
        quadrant = (lane // half) & 3
        a = value[base]; b = value[base + half]; c = value[base + 2*half]; d = value[base + 3*half]
        start = 5 * (half - 1) // 3
        first = np.array([coeffs[start + int(l)] for l in local])
        second = np.array([coeffs[start + half + int(l)] for l in local])
        upper = np.array([coeffs[start + 2*half + int(l)] for l in local])
        cross_low = np.array([coeffs[start + 3*half + int(l)] for l in local])
        cross_high = np.array([coeffs[start + 4*half + int(l)] for l in local])
        p = np.array([mul(v, co) for v, co in zip(b, first)])
        c_low = np.array([mul(v, co) for v, co in zip(c, second)])
        c_high = np.array([mul(v, co) for v, co in zip(c, upper)])
        d_low = np.array([mul(v, co) for v, co in zip(d, cross_low)])
        d_high = np.array([mul(v, co) for v, co in zip(d, cross_high)])
        ls = a + p + c_low + d_low
        ld = a + p - c_low - d_low
        us = a - p + c_high - d_high
        ud = a - p - c_high + d_high
        value = np.where(quadrant == 0, ls, np.where(quadrant == 1, us, np.where(quadrant == 2, ld, ud)))
        value = (value + 2 * Q) % Q
    if LOG_N % 2:
        half = N // 2
        local = lane % half
        left = value[local]; right = value[local + half]
        start = 5 * ((1 << (2 * (LOG_N // 2))) - 1) // 3
        coeff = np.array([coeffs[start + int(l)] for l in local])
        weighted = np.array([mul(v, co) for v, co in zip(right, coeff)])
        value = np.where(lane < half, left + weighted, left - weighted)
        value = (value + Q) % Q
    return value.astype(np.int64)

def bit_mirror(l, LOG_N):
    r = 0
    for bit in range(LOG_N):
        r |= ((l >> bit) & 1) << (LOG_N - 1 - bit)
    return r

def base4_rev(l, width):
    LOG_N = width.bit_length() - 1
    ndig = LOG_N // 2
    digits = [(l >> (2*i)) & 3 for i in range(ndig)]
    if LOG_N % 2:
        digits.append((l >> (2*ndig)) & 1)
    r = 0
    for i, dgi in enumerate(reversed(digits)):
        r |= dgi << (2 * i)
    return r

def refdft(xrow, width):
    return np.array([sum(int(xrow[j]) * pow(17, (256//width)*k*j, Q) for j in range(width)) % Q for k in range(width)])

results = {}
for width in (64, 128):
    LOG_N = width.bit_length() - 1
    bm_eq = b4_eq = True
    rng = np.random.default_rng(3)
    for t in range(4):
        xrow = rng.integers(0, Q, width)
        ref = refdft(xrow, width)
        bm_eq &= bool(np.array_equal(simulate(xrow, width, lambda l: bit_mirror(l, LOG_N)), ref))
        b4_eq &= bool(np.array_equal(simulate(xrow, width, lambda l: base4_rev(l, width)), ref))
    # spike row check for bit-mirror
    spike = np.zeros(width, dtype=np.int64); spike[1] = 1
    bm_spike = bool(np.array_equal(simulate(spike, width, lambda l: bit_mirror(l, LOG_N)), refdft(spike, width)))
    b4_spike = bool(np.array_equal(simulate(spike, width, lambda l: base4_rev(l, width)), refdft(spike, width)))
    results[f"N{width}"] = {"bitmirror_matches_ref": bm_eq, "base4_rev_matches_ref": b4_eq,
                            "bitmirror_spike_ok": bm_spike, "base4_spike_ok": b4_spike}
print(json.dumps(results))