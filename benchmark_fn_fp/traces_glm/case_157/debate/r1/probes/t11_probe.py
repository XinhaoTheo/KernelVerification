
import torch, importlib.util, json
spec = importlib.util.spec_from_file_location("kernel", "/root/cases/case_157/kernel.py")
K = importlib.util.module_from_spec(spec); spec.loader.exec_module(K)
dev = "cuda"
C = 32  # r = 0..31
r = torch.arange(C, dtype=torch.int32, device=dev)
z = torch.zeros(C, dtype=torch.int32, device=dev)
g = torch.Generator("cpu").manual_seed(11)
rows = 64
# per channel choose multiplier near 2^31-1
m = (2**31 - 1 - (torch.arange(C) % 7)).to(torch.int32).to(dev)
mc = m.cpu().to(torch.int64)
# build accumulators so R = round(a*m/2^(31+r)) lands in nonsaturated [-100,100]
rows_list = []
for c in range(C):
    denom = 1 << (31 + c)
    vals = []
    for q in list(range(-100, 101, 10)):
        target = q * denom
        a_base = int(round(target / int(mc[c])))
        for d in (-2, -1, 0, 1, 2):
            vals.append(a_base + d)
    vals = [v if -(1<<31) <= v < (1<<31) else 0 for v in vals]
    while len(vals) < rows: vals.append(0)
    rows_list.append(vals[:rows])
a = torch.tensor(rows_list, dtype=torch.int64).t()  # rows x C
# add extreme rows: a = -2^31 and 2^31-1
a[0, :] = -(1<<31)
a[1, :] = (1<<31)-1
a = a.to(torch.int32).to(dev)
out = K.run(a, m, r, z)
ref = K.reference(a, m, r, z)
mism = (out != ref)
n = int(mism.sum())
per_r = {}
if n:
    for i, c in mism.nonzero().tolist():
        per_r.setdefault(c, []).append({"a": int(a[i,c]), "m": int(m[c]), "r": c,
                                        "expected": int(ref[i,c]), "got": int(out[i,c])})
# also check saturation-free coverage count
nonsat = int(((ref > -128) & (ref < 127) & mism).sum())
print(json.dumps({"claim":"c3","mismatch_count":n,"total":a.numel(),
                  "mismatches_in_nonsaturated_range":nonsat,
                  "per_channel_r_examples":{str(k):v[:4] for k,v in per_r.items()},
                  "extreme_row_a=-2^31_mismatches":int(mism[0].sum()),
                  "extreme_row_a=2^31-1_mismatches":int(mism[1].sum())}))
