import sys, torch, json
sys.path.insert(0, "/root/cases/case_140")
import kernel as k
dev = "cuda"
g = torch.Generator().manual_seed(7)
results = {}
for n in [1, 2, 255, 256, 257, 4093, 131071]:
    x = torch.randint(0, 1<<32, (n,), dtype=torch.int64, generator=g).to(torch.uint32).to(dev)
    out = k.run(x)
    ok_shape = tuple(out.shape) == (n,2) and out.dtype == torch.uint32 and out.is_cuda
    ref = k.reference(x)
    exact = torch.equal(out.to(torch.int64), ref.to(torch.int64))
    results[n] = {"shape_ok": ok_shape, "exact": bool(exact)}
print(json.dumps({"claim":"c3", "results": results,
 "all_pass": all(r["exact"] and r["shape_ok"] for r in results.values())}))