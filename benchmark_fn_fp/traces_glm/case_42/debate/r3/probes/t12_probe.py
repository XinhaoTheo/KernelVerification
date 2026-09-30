import sys, json
sys.path.insert(0, "/root/numerical_cases/case_g")
import torch, numpy as np
from kernel import run, make_inputs

vals, = make_inputs("cuda")
out = run(vals)
torch.cuda.synchronize()
ref = vals.double().sum(dim=1)  # float64 per-row reference
err = out.double() - ref
E = float(err.norm() / max(float(ref.norm()), 0.008))
print(json.dumps({
    "E": E, "tolerance": 0.1, "passes": E <= 0.1,
    "all_finite": bool(torch.isfinite(out).all()),
    "ref_norm": float(ref.norm()),
    "err_norm": float(err.norm()),
    "max_abs_err": float(err.abs().max()),
    "max_abs_ref_row": float(ref.abs().max()),
}))