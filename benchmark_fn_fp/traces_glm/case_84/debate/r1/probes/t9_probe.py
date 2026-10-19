
import torch, importlib.util, json
spec = importlib.util.spec_from_file_location("kernel", "/root/cases/case_157/kernel.py")
K = importlib.util.module_from_spec(spec); spec.loader.exec_module(K)
dev = "cuda"
g = torch.Generator("cpu").manual_seed(7)
rows, C = 4096, 8
# all channels r=0, multipliers near 2^31-1
m = (2**31 - 1 - torch.arange(C, dtype=torch.int64)).to(torch.int32).to(dev)
r = torch.zeros(C, dtype=torch.int32, device=dev)
z = torch.zeros(C, dtype=torch.int32, device=dev)
# extreme-ish accumulators, full int32 range plus engineered near-extremes
a = torch.randint(-(1<<31), 1<<31, (rows, C), generator=g, dtype=torch.int64).to(torch.int32)
a[0, :] = torch.tensor([(1<<31)-1, -(1<<31), (1<<31)-2, -(1<<31)+1, (1<<31)-3, -(1<<31)+2, (1<<31)-1000, -(1<<31)+1000], dtype=torch.int32)
a = a.to(dev)
out = K.run(a, m, r, z)
ref = K.reference(a, m, r, z)
mism = (out != ref)
n = int(mism.sum())
examples = []
if n:
    idx = mism.nonzero()[:10]
    for i, c in idx.tolist():
        examples.append({"a": int(a[i,c]), "m": int(m[c]), "r": 0, "z": 0,
                         "expected": int(ref[i,c]), "got": int(out[i,c])})
    # is mismatch a 1-unit error?
    diff = (out.to(torch.int64) - ref.to(torch.int64))[mism]
    print("max_abs_diff_mismatched:", int(diff.abs().max()))
print(json.dumps({"claim":"c1","mismatch_count":n,"total":rows*C,"examples":examples}))
