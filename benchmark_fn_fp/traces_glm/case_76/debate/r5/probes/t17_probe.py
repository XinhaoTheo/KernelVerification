import torch, struct, sys, json
sys.path.insert(0, "/root/cases/case_149")
from kernel import run
import mpmath as mp
mp.mp.prec = 120

# Region: large NEGATIVE x, subnormal-output range and underflow cutoff.
codes = []
for b in range(65536):
    v = struct.unpack("<e", b.to_bytes(2, "little"))[0]
    if v == v and abs(v) != float("inf") and -20.0 <= v <= -9.0:
        codes.append((v, b))
codes.sort()
payload = bytearray()
for _, b in codes: payload += b.to_bytes(2, "little")
x = torch.frombuffer(bytearray(payload), dtype=torch.float16).clone().to("cuda")
y = run(x).cpu()
got_bits = y.view(torch.uint16).tolist()
xl = x.cpu().tolist()

def correct_bits(xv):
    xm = mp.mpf(float(xv))
    e = mp.exp(xm)
    s = e / (1 + e)
    return int.from_bytes(struct.pack("<e", float(s)), "little")

mism = []
sub_mism = 0; zero_mism = 0
for i in range(len(codes)):
    exp = correct_bits(xl[i])
    if exp != got_bits[i]:
        mism.append({"x": xl[i], "expected_bits": exp, "got_bits": got_bits[i]})
        if 0 < exp < 1024: sub_mism += 1
        if exp == 0 or got_bits[i] == 0: zero_mism += 1

# boundary census: smallest x with nonzero output, largest x with +0 output
smallest_nonzero = None; largest_zero = None
for i in range(len(codes)):
    if got_bits[i] != 0 and smallest_nonzero is None: smallest_nonzero = xl[i]
    if got_bits[i] == 0: largest_zero = xl[i]
# expected cutoff: smallest x where exact sigmoid >= 2^-25
exp_smallest_nonzero = None; exp_largest_zero = None
for i in range(len(codes)):
    xb = correct_bits(xl[i])
    if xb != 0 and exp_smallest_nonzero is None: exp_smallest_nonzero = xl[i]
    if xb == 0: exp_largest_zero = xl[i]

print(json.dumps({
    "metric": "exact FP16 bit-pattern equality in negative-x subnormal/underflow region [-20,-9], oracle = mpmath(120-bit) sigmoid via struct.pack('<e') RNE",
    "n_tested": len(codes),
    "mismatch_count": len(mism),
    "subnormal_region_mismatches": sub_mism,
    "underflow_zero_mismatches": zero_mism,
    "kernel_smallest_x_nonzero_output": smallest_nonzero,
    "kernel_largest_x_zero_output": largest_zero,
    "oracle_smallest_x_nonzero": exp_smallest_nonzero,
    "oracle_largest_x_zero": exp_largest_zero,
    "first_mismatches": mism[:10],
}))
