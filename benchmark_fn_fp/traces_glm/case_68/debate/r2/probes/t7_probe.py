import torch, json
import importlib.util
spec = importlib.util.spec_from_file_location("kernel", "/root/cases/case_141/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

# targeted tail values ≡6 mod 7 and controls
xs = [3435973834, 3435973841, 3435973841, 4000000005, 4294967289, 4294967292, 4294967295, 0, 6, 4294967293, 4294967288]
x = torch.tensor(xs, dtype=torch.uint32, device="cuda")
out = k.run(x)
ref = k.reference(x.cpu()).cuda()
mism = (out != ref).any(dim=1).cpu()
results = []
for i, v in enumerate(xs):
    results.append({"x": int(v), "x%7": int(v)%7, "row": int(out[i,0].item()), "col": int(out[i,1].item()),
                    "ref_row": int(ref[i,0].item()), "ref_col": int(ref[i,1].item()),
                    "bad": bool(mism[i])})

# sweep the whole predicted failing set: all x ≡6 mod 7 with x in [3435973841, 2^32-1]
n = ( (1<<32) - 3435973841 + 6 ) // 7 + 1
base = 3435973841  # ≡ 6 mod 7? check: 3435973841 % 7
vals = torch.arange(0, n, dtype=torch.int64) * 7 + 3435973841
vals = vals[vals < (1<<32)].to(torch.uint32).cuda()
o = k.run(vals); r = k.reference(vals.cpu()).cuda()
bad = (o != r).any(dim=1)
inv = (o[:,0].to(torch.int64) * 7 + o[:,1].to(torch.int64)) != vals.to(torch.int64)
print(json.dumps({"targeted": results, "sweep_count": int(vals.numel()),
    "sweep_mismatch_count": int(bad.sum().item()),
    "sweep_invariant_violation_count": int(inv.sum().item()),
    "first_bad_x": int(vals[bad.argmax()].item()) if bad.any() else None,
    "last_good_x": int(vals[(~bad)][(~bad).nonzero()[-1]].item()) if (~bad).any() else None,
    "multiplier": k._MULTIPLIER, "shift": k._SHIFT}))