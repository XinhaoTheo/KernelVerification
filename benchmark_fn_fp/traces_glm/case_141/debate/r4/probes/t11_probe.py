import sys, json, torch
sys.path.insert(0, "/root/cases/case_141")
import kernel
g = torch.Generator().manual_seed(2)
results = {}
for n in [1, 2, 255, 256, 257, 512, 131072]:
    v = (torch.rand(n, generator=g) * 2**32).to(torch.int64).to(torch.uint32).cuda()
    out = kernel.run(v)
    ref = kernel.reference(v.cpu()).cuda()
    match = bool((out == ref).all())
    results[str(n)] = {"shape_ok": tuple(out.shape) == (n,2), "match": match}
print(json.dumps(results))
