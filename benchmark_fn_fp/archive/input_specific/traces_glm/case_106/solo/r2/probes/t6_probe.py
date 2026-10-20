
import json, numpy as np, torch, sys
sys.path.insert(0, "/root/numerical_cases/case_h")
import kernel as k

values, = k.make_inputs()
out = k.run(values)
ref = values.double().sum(dim=1)
E = torch.norm((out.double() - ref)) / max(torch.norm(ref).item(), 0.008)
finite = bool(torch.isfinite(out).all())
print(json.dumps({
  "metric": "E = ||output-ref||_2 / max(||ref||_2, 0.008), per contract",
  "E": float(E),
  "finite": finite,
  "max_abs_err": float((out.double()-ref).abs().max()),
  "ref_norm": float(torch.norm(ref)),
  "out_norm": float(torch.norm(out.double())),
  "sample_out": out[:4].tolist(),
  "sample_ref": ref[:4].tolist(),
}))
