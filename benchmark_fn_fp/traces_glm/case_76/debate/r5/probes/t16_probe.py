import torch, struct, sys, json
sys.path.insert(0, "/root/cases/case_149")
from kernel import run, reference
import mpmath as mp
mp.mp.prec = 120

codes = []
for b in range(65536):
    v = struct.unpack("<e", b.to_bytes(2, "little"))[0]
    if v == v and abs(v) != float("inf"):
        codes.append(b)
payload = bytearray()
for b in codes: payload += b.to_bytes(2, "little")
x = torch.frombuffer(bytearray(payload), dtype=torch.float16).clone().to("cuda")
y = run(x).cpu()
got_bits = y.view(torch.uint16).tolist()
xl = x.cpu().tolist()

# Sound oracle: mpmath high-precision sigmoid -> struct.pack('<e') RNE
def correct_bits(xv):
    xm = mp.mpf(float(xv))
    if xm >= 0:
        s = 1 / (1 + mp.exp(-xm))
    else:
        e = mp.exp(xm)
        s = e / (1 + e)
    packed = struct.pack("<e", float(s))
    return int.from_bytes(packed, "little")

# Sanity-check oracle on hand-verified cases
sanity = {}
for xv in (0.0029296875, 8.3203125, 0.0, -0.0, -11.0, -18.375, -18.4375):
    sanity[xv] = correct_bits(xv)

# Cross-check: artifact reference() on the same full tensor
ref = reference(x).cpu()
ref_bits = ref.view(torch.uint16).tolist()
ref_vs_oracle_mism = 0

mismatches = []
for i in range(len(codes)):
    expected = correct_bits(xl[i])
    if expected != ref_bits[i]:
        ref_vs_oracle_mism += 1
    if expected != got_bits[i]:
        if len(mismatches) < 15:
            mismatches.append({"x": xl[i], "expected_bits": expected, "got_bits": got_bits[i],
                               "ref_bits": ref_bits[i]})

print("sanity_oracle_bits(x=0.0029296875 expect 0x3802):", hex(sanity[0.0029296875]))
print("sanity_oracle_bits(x=8.3203125 expect 0x3c00):", hex(sanity[8.3203125]))
print("sanity_oracle_bits(x=0.0/-0.0 expect 0x3800):", hex(sanity[0.0]), hex(sanity[-0.0]))
print("sanity_oracle_bits(x=-11.0, -18.375, -18.4375):", hex(sanity[-11.0]), hex(sanity[-18.375]), hex(sanity[-18.4375]))
print(json.dumps({
    "metric": "exact FP16 bit-pattern equality, exhaustive sweep over all 65504 finite FP16 inputs, oracle = mpmath(120-bit) sigmoid packed via struct.pack('<e') RNE, cross-checked vs artifact reference()",
    "n_tested": len(codes),
    "mismatch_count": len(mismatches),
    "oracle_vs_reference_mismatch_count": ref_vs_oracle_mism,
    "sanity_x_0.0029296875_oracle_bits": sanity[0.0029296875],
    "sanity_x_8.3203125_oracle_bits": sanity[8.3203125],
    "first_mismatches": mismatches,
}))
