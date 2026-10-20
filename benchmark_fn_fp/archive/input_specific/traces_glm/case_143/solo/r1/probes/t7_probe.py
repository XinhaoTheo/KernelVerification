import importlib.util, torch
from fractions import Fraction
spec = importlib.util.spec_from_file_location("k", "/root/evidence_cases/case_e19/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

x, = k.make_inputs()
before = x.clone()
out = k.run(x)
vals = before.cpu().numpy().astype(object)
# exact real-number sums via Fractions
target = [float(sum(Fraction(int(v)) for v in row) / Fraction(2**0) if False else sum(Fraction(v) for v in row)) for row in vals]
actual = out.cpu().numpy().astype("float64")
import numpy as np, math
norm_t = math.sqrt(sum(t*t for t in target))
err = math.sqrt(sum((a-t)**2 for a,t in zip(actual,target))) / max(norm_t, 1e-12)
print({ "target": target, "actual": actual.tolist(),
        "relative_error": err, "tolerance": 1e-5,
        "shape_ok": tuple(out.shape)==(4,) and out.dtype==torch.float32,
        "finite": bool(np.isfinite(actual).all()),
        "unmodified": bool(torch.equal(x, before)),
        "passed": bool(tuple(out.shape)==(4,) and out.dtype==torch.float32 and np.isfinite(actual).all() and torch.equal(x,before) and err<=1e-5)})