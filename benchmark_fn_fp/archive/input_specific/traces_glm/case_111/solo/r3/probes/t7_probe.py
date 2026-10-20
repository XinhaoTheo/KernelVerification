import json, torch, numpy as np, importlib.util
spec = importlib.util.spec_from_file_location("kernel", "/root/numerical_cases/case_g/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
(values,) = m.make_inputs("cuda")
out = m.run(values).float().cpu()
ref = values.double().sum(dim=1).cpu()
E = (out - ref).norm() / max(ref.norm(), 0.008)
print(json.dumps({
 "metric": "E = ||out-ref||2 / max(||ref||2,0.008)",
 "E": float(E),
 "finite": bool(torch.isfinite(out).all()),
 "ref_norm": float(ref.norm()),
 "max_abs_err": float((out-ref).abs().max()),
 "max_rel_row_err": float(((out-ref).abs()/ref.abs().clamp(min=1e-9)).max()),
 "sample_out": out[:4].tolist(), "sample_ref": ref[:4].tolist()
}))
