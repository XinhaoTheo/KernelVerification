
import json, math, struct, torch
from mpmath import mp, mpf, exp as mpexp
mp.dps = 60

# all finite FP16 bit patterns
vals = []
patterns = []
for p in range(65536):
    b = struct.pack("<H", p)
    f = struct.unpack("<e", b)[0]
    if math.isfinite(f):
        vals.append(f)
        patterns.append(p)
print("finite count", len(vals))

# high-precision reference: correct RN rounding of real sigmoid to fp16
# Use Fraction/mpf to get exact-ish real value; round manually with round-half-even.
ref_bits = []
for f in vals:
    x = mpf(f)
    s = mpf(1)/(1 + mpexp(-x))
    # round to fp16 via struct with mpmath -> convert with careful tie handling:
    # easiest: use struct only if not a tie; detect ties by checking neighborhood.
    # Do manual rounding: represent s as mpf, find fp16 candidates via nextafter.
    lo = mpf(struct.unpack("<e", struct.pack("<e", float(s)))[0])
    # better: compute candidates by converting mpf to float, then check rounding directly.
    # Manual: get exponent, construct.
    # Simpler robust approach: use Python float(s) is double-rounded -> can be wrong at ties.
    # Use mpmath to pick between two nearest fp16 values:
    def fp16_candidates(v):
        fl = float(v)
        a = struct.unpack("<e", struct.pack("<e", fl))[0]
        # get neighbor below and above using bit manipulation on candidate
        import struct as st
        bits = st.unpack("<H", st.pack("<e", a))[0]
        if a == 0:
            below_bits, above_bits = 0x8000, 0x0001  # -0, smallest subnormal
        elif bits & 0x8000:  # negative
            if bits == 0x8000:
                below_bits, above_bits = 0x8001, 0x0000
            else:
                below_bits, above_bits = bits+1, bits-1
        else:
            below_bits, above_bits = bits-1, bits+1
        b = struct.unpack("<e", st.pack("<H", below_bits))[0]
        c = struct.unpack("<e", st.pack("<H", above_bits))[0]
        return b, a, c, below_bits, bits, above_bits
    b, mid, a, bb, mb, ab = fp16_candidates(s)
    if s == mpf(mid):
        chosen = mb if (struct.unpack("<H", struct.pack("<e", mid))[0] & 1) == 0 else (ab if False else None)
        # even tie: pick candidate with even last bit
        chosen = mb if (mb & 1)==0 else ab
    elif s < mpf(mid):
        # between below and mid
        if s - mpf(b) < mpf(mid) - s:
            chosen = bb
        elif s - mpf(b) > mpf(mid) - s:
            chosen = mb
        else:
            chosen = bb if (bb & 1)==0 else mb
    else:
        if s - mpf(mid) < mpf(a) - s:
            chosen = mb
        elif s - mpf(mid) > mpf(a) - s:
            chosen = ab
        else:
            chosen = mb if (mb & 1)==0 else ab
    ref_bits.append(chosen)

N = len(vals)
x_gpu = torch.tensor(vals, dtype=torch.float16, device="cuda")
import sys
sys.path.insert(0, "/root/cases/case_148")
import kernel
out = kernel.run(x_gpu)
out_bits = out.view(torch.int16).cpu().numpy().astype("int64") & 0xFFFF
ref_arr = [ (b if b < 32768 else b - 65536) for b in ref_bits ]
mism = []
for i in range(N):
    if out_bits[i] != ref_arr[i]:
        mism.append((vals[i], patterns[i], int(out_bits[i]), ref_bits[i]))
print("mismatches:", len(mism))
for m in mism[:20]:
    print(m)
print(json.dumps({"mismatch_count": len(mism), "tested": N}))
