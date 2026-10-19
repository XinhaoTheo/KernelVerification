import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_29/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
x = torch.tensor([460.0, 464.0, 500.0, 1000.0, 10000.0, -460.0, 449.0, 448.0], dtype=torch.float32, device="cuda")
out = k.fp8_roundtrip(x)
res = {float(v): float(o) for v, o in zip(x.tolist(), out.tolist())}
# true nearest-representable handling: e4m3 has no representation above 448
bad = {v: o for v, o in res.items() if abs(o) > 448.0 and abs(o) != 448.0}
print(json.dumps({"results": res, "outputs_above_448": bad, "n_above_448": len(bad)}))