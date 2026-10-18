import torch, json
import importlib.util
spec = importlib.util.spec_from_file_location("kernel", "/root/cases/case_140/kernel.py")
k = importlib.util.spec_from_file_location
spec = importlib.util.spec_from_file_location("kernel_mod", "/root/cases/case_140/kernel.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

dev = "cuda"
cases = []
# boundary/extreme values
vals = set([0, 1, 6, 7, 8, 13, 14, 4294967290, 4294967295, 4294967289, 2**32-7, 2**32-8, 2**32-14])
# exhaustive-ish: every multiple of 7 +/-1 in a high range and near extrema
for q in [0,1,2,612712399,613566756,306783378,  (2**32-1)//7]:
    for v in [q*7-1, q*7, q*7+1, q*7+6, q*7+7]:
        if 0 <= v <= 4294967295: vals.add(v)
# large random sample
g = torch.Generator().manual_seed(0)
r = torch.randint(0, 2**32, (300000,), dtype=torch.int64, generator=g).tolist()
vals.update(r)
# all uint32 values in a slice near top and near 0
vals.update(range(2**32-1000, 2**32))
vals.update(range(0, 5000))
vals.update(range(7*(2**31//7)-5, 7*(2**31//7)+5))

arr = torch.tensor(sorted(vals), dtype=torch.int64).to(torch.uint32).to(dev)
out = mod.run(arr)
ref = mod.reference(arr.cpu()).to(dev)
exact = torch.equal(out, ref)
inv = (out[:,0]*7 + out[:,1]).to(torch.int64) == arr.to(torch.int64)
bad = (out != ref).any(dim=1)
nbad = int(bad.sum())
js = {"n_inputs": arr.numel(), "exact_match_reference": bool(exact),
      "invariant_7row_plus_col": bool(inv.all()),
      "mismatch_count": nbad,
      "first_bad": None}
if nbad:
    i = int(bad.nonzero()[0,0])
    js["first_bad"] = {"value": int(arr[i].item()), "kernel": out[i].cpu().tolist(), "ref": ref[i].cpu().tolist()}
print(json.dumps(js))
