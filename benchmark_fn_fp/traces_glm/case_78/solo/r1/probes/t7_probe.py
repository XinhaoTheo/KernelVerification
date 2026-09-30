import numpy as np, json, sys
sys.path.insert(0, '/root/evidence_cases/case_e17')
import kernel

x, = kernel.make_inputs()
before = x.clone()
out = kernel.run(x)
vals = before.cpu().numpy().astype(np.float64)
target = vals.sum(axis=1)  # float64; exact enough for these values
# exact check with integer-free mathlib: also use Kahan on float64 (identical here)
actual = out.cpu().numpy().astype(np.float64)
rel = float(np.linalg.norm(actual-target)/max(np.linalg.norm(target),1e-12))
structural = (tuple(out.shape)==(4,) and out.dtype==torch.float32 if False else True)
import torch
structural = tuple(out.shape)==(4,) and out.dtype==torch.float32 and bool(np.isfinite(actual).all()) and bool(torch.equal(x,before))
print(json.dumps({"target":target.tolist(),"output":actual.tolist(),"relative_error":rel,"tolerance":1e-5,"structural":structural,"passed":bool(structural and rel<=1e-5)}))