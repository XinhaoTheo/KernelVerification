import torch, struct, sys, json
sys.path.insert(0, "/root/cases/case_149")
from kernel import run
import mpmath as mp
mp.mp.prec = 80

# Focus: large positive x producing subnormal FP16 outputs (x in ~[11, 18.5]) and the underflow boundary.
# Build all finite FP16 inputs with x >= 10.0 (covers subnormal-output and underflow-to-+0 regions), plus x=largest finite.
vals = [b for b in range(65536) if (b & 0x7F00) or True]
x_all = torch.frombuffer(bytearray(b.to_bytes(2, "little") for b in vals).to_bytes(0, "little")) if False else None
# simpler: construct via python floats
codes = []
for b in range(65536):
    v = struct.unpack("<e", b.to_bytes(2, "little"))[0]
    if v == v and abs(v) != float("inf") and v >= 10.0:
        codes.append(b)
payload = bytearray()
for b in codes:
    payload += b.to_bytes(2, "little")
x = torch.frombuffer(payload, dtype=torch.float16).clone().to("cuda")
y = run(x)
xl = x.cpu().tolist()
yb = y.cpu().view(torch.uint16).tolist()

def correct_fp16_bits(xv):
    xm = mp.mpf(float(xv))
    s = 1 / (1 + mp.exp(-xm))
    if s <= mp.mpf(2)**-25:  # rounds to +0
        return 0
    if s < mp.mpf(2)**-14:
        ulp = mp.mpf(2)**-24
        q = int(mp.floor(s / ulp))
        mid = (q + mp.mpf(1)/2) * ulp
        if s < mid: k = q
        elif s > mid: k = q + 1
        else: k = q if q % 2 == 0 else q + 1
        return k
    e = int(mp.floor(mp.log(s, 2))) + 1
    ulp = mp.mpf(2)**(e - 11)
    q = int(mp.floor(s / ulp))
    mid = (q + mp.mpf(1)/2) * ulp
    if s < mid: k = q
    elif s > mid: k = q + 1
    else: k = q if q % 2 == 0 else q + 1
    expfield = 15 + int(mp.floor(mp.log(s, 2)))
    return (expfield << 10) | (k - 1024)

mism = []
sub_mism = 0
zero_mism = 0
for i in range(len(codes)):
    exp = correct_fp16_bits(xl[i])
    got = yb[i]
    if exp != got:
        mism.append((xl[i], exp, got))
        if exp < 1024: sub_mism += 1
        if exp == 0: zero_mism += 1
        if len(mism) <= 20:
            print("MISMATCH x=%r expected_bits=0x%04x got_bits=0x%04x" % (xl[i], exp, got))
# identify boundary: smallest x giving nonzero output vs +0
nonzero_min = None; zero_max = None
for i in range(len(codes)):
    if yb[i] != 0 and nonzero_min is None: nonzero_min = xl[i]
    if yb[i] == 0: zero_max = xl[i]
print(json.dumps({
    "metric": "exact FP16 bit-pattern equality in subnormal/underflow output region (x>=10), vs mpmath(80-bit) correctly-rounded sigmoid",
    "n_tested": len(codes),
    "x_min": min(xl), "x_max": max(xl),
    "mismatch_count": len(mism),
    "subnormal_region_mismatches": sub_mism,
    "underflow_zero_mismatches": zero_mism,
    "smallest_x_nonzero_output": nonzero_min,
    "largest_x_zero_output": zero_max,
    "first_mismatches": [{"x": m[0], "expected_bits": m[1], "got_bits": m[2]} for m in mism[:10]],
}))
