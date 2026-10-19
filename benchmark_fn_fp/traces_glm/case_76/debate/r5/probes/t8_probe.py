import torch, struct, sys, json
sys.path.insert(0, "/root/cases/case_149")
from kernel import run
import mpmath as mp
mp.mp.prec = 80

# Build all finite FP16 encodings
vals = []
for bits in range(65536):
    v = struct.unpack("<e", bits.to_bytes(2, "little"))[0]
    if v == v and abs(v) != float("inf"):
        vals.append(bits)
N = len(vals)
x = torch.frombuffer(bytearray(b for bits in vals for b in bits.to_bytes(2, "little")), dtype=torch.float16).clone().to("cuda")
y = run(x)
yh = y.cpu()

def correct_fp16_bits(xv):
    # exact sigmoid via mpmath
    xm = mp.mpf(float(xv))
    if xm >= 0:
        s = 1 / (1 + mp.exp(-xm))
    else:
        e = mp.exp(xm)
        s = e / (1 + e)
    if s <= 0:
        return 0  # +0
    if s >= 1:
        return 0x3C00  # 1.0 (sigmoid(35)<1? sigmoid<1 always; sigmoid of 35.53? max x=65504 -> 1 exactly in mp? handle)
    # determine ulp region
    if s < mp.mpf(2)**-14:
        ulp = mp.mpf(2)**-24
        emax = 0
    else:
        e = int(mp.floor(mp.log(s, 2))) + 1
        ulp = mp.mpf(2)**(e - 11)
    q = int(mp.floor(s / ulp))
    mid = (q + mp.mpf(1)/2) * ulp
    if s < mid:
        k = q
    elif s > mid:
        k = q + 1
    else:
        k = q if q % 2 == 0 else q + 1
    # encode
    if s < mp.mpf(2)**-14:
        return k  # subnormal: fraction bits directly
    expfield = 15 + int(mp.floor(mp.log(s, 2)))
    if k >= 2048:
        k = 2047; expfield = 30  # overflow -> inf shouldn't happen for sigmoid
    return (expfield << 10) | (k - 1024)

mismatches = []
near_mid = []
x_list = x.cpu().tolist()
y_bits = yh.view(torch.uint16).tolist()
for i in range(N):
    expected = correct_fp16_bits(x_list[i])
    got = y_bits[i]
    if expected != got:
        mismatches.append((x_list[i], expected, got))
        if len(mismatches) <= 20:
            print("MISMATCH x=%r expected_bits=0x%04x got_bits=0x%04x" % (x_list[i], expected, got))

# also sanity: count how many exact values sit within 1e-3 ulp of an fp16 midpoint (double-rounding risk density)
print(json.dumps({
    "metric": "exact FP16 bit-pattern equality vs mpmath(80-bit) correctly-rounded sigmoid over all 65504 finite FP16 inputs",
    "n_tested": N,
    "mismatch_count": len(mismismatches) if False else len(mismatches),
    "first_mismatches": [{"x": m[0], "expected_bits": m[1], "got_bits": m[2]} for m in mismatches[:10]],
}))
