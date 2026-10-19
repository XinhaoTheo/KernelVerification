import torch, struct, json, math
import importlib.util
spec = importlib.util.spec_from_file_location("kernel_mod", "/root/cases/case_149/kernel.py")
km = importlib.util.module_from_spec(spec); spec.loader.exec_module(km)

# sigmoid(x) near smallest FP16 subnormal: smallest sub 2^-24 ~5.96e-8, tie at 2^-25.
# sigmoid(-x)=exp(-x) roughly; boundary x ~ -ln(2^-25) ~ 17.3. Enumerate all FP16 x in [16.5, 18.5] plus negatives.
from decimal import Decimal, getcontext
getcontext().prec = 60
import numpy as np

def exact_sigmoid_half(v):  # v: float16 value; return Decimal sigmoid
    x = Decimal(v)
    # sigmoid = 1/(1+exp(-x)); for negatives use exp(x)/(1+exp(x))
    if v >= 0:
        e = (-x).exp()
        return Decimal(1)/ (Decimal(1)+e)
    else:
        e = x.exp()  # exp(x), x<0, tiny
        return e / (Decimal(1)+e)

cand = []
all_bits = [b for b in range(0x10000) if math.isfinite(struct.unpack('<e', b.to_bytes(2,'little'))[0])]
xs_list = [struct.unpack('<e', b.to_bytes(2,'little'))[0] for b in all_bits]
xs = torch.tensor(xs_list, dtype=torch.float16)
# select |x| in [16,19]: sigmoid magnitude between ~1.1e-8 and ~1.1e-7 spans the 2^-25 tie
sel = [i for i, v in enumerate(xs_list) if 15.5 <= abs(v) <= 20.0]
sel_x = xs[torch.tensor(sel)].cuda()
out = km.run(sel_x)
out_bits = out.view(torch.int16).to(torch.int32).cpu().numpy()
mism = 0
examples = []
tie_dist_min = None
for j, i in enumerate(sel):
    v = xs_list[i]
    s = exact_sigmoid_half(v)
    # correctly rounded FP16 of exact decimal
    lo = s * 1  # use Decimal -> float via struct pack '<e' not decimal-aware; do manual RNE
    # compare decimal s against tie 2^-25 = 2.98023223876953125e-8
    half_sub = Decimal(2) ** Decimal(-25)
    full_sub = Decimal(2) ** Decimal(-24)
    if s >= full_sub + half_sub:  # won't happen here
        correct = "above-subnormal-range"
    elif s > half_sub:
        correct = 1  # smallest subnormal
    elif s < half_sub:
        correct = 0  # zero
    else:
        correct = "exact-tie"
    d = abs(s - half_sub)
    tie_dist_min = d if tie_dist_min is None else min(tie_dist_min, d)
    kb = int(out_bits[j]) & 0xFFFF
    kernel_sub = 1 if kb == 1 else (0 if kb == 0 else "other")
    if kernel_sub != correct:
        mism += 1
        examples.append({"x": v, "exact_sigmoid": str(s), "correct": str(correct),
                         "kernel_bits": kb, "kernel_val": float(out.cpu()[j])})
print(json.dumps({"inputs_tested": len(sel), "abs_x_range": "[15.5,20.0]",
                  "subnormal_boundary_mismatches": mism, "mismatch_examples": examples,
                  "min_distance_to_exact_tie": str(tie_dist_min),
                  "metric": "decimal-60 exact sigmoid vs FP16 subnormal tie 2^-25; kernel output classified as 0 / smallest-subnormal"}))