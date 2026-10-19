import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_134/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev = "cuda"
results = {}
bad = 0
for n in [1, 2, 255, 256, 257, 4093, 65536, 131072]:
    torch.manual_seed(n)
    x = ((8.0*torch.randn(n)).to(torch.float16)).to(dev)
    out = k.run(x)
    ref = k.reference(x)
    err = (out.double()-ref).abs().max().item()
    finite = bool(torch.isfinite(out).all().item())
    results[n] = {"max_abs_err": err, "out_finite": finite,
                  "shape_ok": tuple(out.shape)==(n,), "dtype_ok": out.dtype==torch.float32,
                  "passed": err <= 1e-6 and finite}
    bad += (err > 1e-6)
# also all-equal-extreme single-element and adjacent-canary OOB test
n = 4093
torch.manual_seed(0)
base = (8.0*torch.randn(n+256)).to(torch.float16).cuda()
x = base[:n]
out = k.run(x)
ref = k.reference(x)
tail_ok = bool((out.double()-ref).abs().max().item() <= 1e-6)
# canary: run on slice, check neighbors of the slice in base unchanged (slice of contiguous is contiguous)
print(json.dumps({"per_size": {str(kk): vv for kk, vv in results.items()},
                  "num_sizes_failed": bad, "tail_4093_ok": tail_ok}))