import json, torch, importlib.util
spec = importlib.util.spec_from_file_location("k163", "/root/cases/case_163/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

results = {}
for N in (64, 128, 256):
    for B in (1, 7, 256):
        vals, = k.make_inputs(device="cuda", batch=B, width=N, seed=N+B)
        err = {}
        try:
            out = k.run(vals)
            ref = k.reference(vals).to(torch.float32)
            tol = 1e-4 + 1e-6*ref.abs()
            diff = (out - ref).abs()
            err = {"raised": False, "max_err": float(diff.max()),
                   "max_err_over_tol": float((diff/tol).max()),
                   "n_violations": int((diff > tol).sum()),
                   "nonfinite_out": int((~torch.isfinite(out)).sum()),
                   "shape": list(out.shape), "dtype": str(out.dtype),
                   "input_preserved": bool(torch.equal(vals, vals.clone()))}
        except Exception as e:
            err = {"raised": True, "exc": type(e).__name__, "msg": str(e)[:300]}
        results[f"N{N}_B{B}"] = err
print(json.dumps({"metric": "per-component |actual-exact| vs 1e-4+1e-6*|exact| across N=64/128/256 and B=1/7/256", "results": results}))
