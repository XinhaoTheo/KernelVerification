import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_154/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev = "cuda"
res = {}

# case A: same tensor object as both args
t = torch.randn((1000, 2)).to(torch.float16).to(dev)
t[(t == 0).all(dim=1), 0] = 1
t0 = t.clone()
try:
    out = k.run(t, t)
    ref = k.reference(t0, t0)
    err = (out.to(torch.float64) - ref.to(torch.float64)).abs()
    tol = 1e-7 + 5e-7 * ref.to(torch.float64).abs()
    res["same_tensor"] = {"accepted": True, "max_ratio": float((err / tol).max()),
                          "inputs_preserved": bool(torch.equal(t, t0)),
                          "finite_out": bool(torch.isfinite(out).all())}
except Exception as e:
    res["same_tensor"] = {"accepted": False, "error": str(e)}

# case B: two overlapping views of one backing storage (offset views, both contiguous)
back = torch.randn((3000, 2)).to(torch.float16).to(dev)
back[(back == 0).all(dim=1)] = 1.0  # avoid all-zero rows anywhere
n_v = back[500:1500]          # contiguous view, nonzero offset
d_v = back[1000:2000]         # overlapping, contiguous, nonzero offset
# ensure no denominator row all zero
ref = k.reference(n_v, d_v)
out = k.run(n_v, d_v)
err = (out.to(torch.float64) - ref.to(torch.float64)).abs()
tol = 1e-7 + 5e-7 * ref.to(torch.float64).abs()
res["overlapping_views"] = {"max_ratio": float((err / tol).max()),
                            "finite_out": bool(torch.isfinite(out).all()),
                            "denominator_row_nonzero_ok": True}

print(json.dumps({"metric": "error_ratio_vs_tolerance_under_shared_backing_storage", **res}))
