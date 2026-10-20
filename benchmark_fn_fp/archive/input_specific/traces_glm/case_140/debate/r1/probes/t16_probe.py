import json, os, numpy as np, torch, tempfile
src = '''
import numpy as np, torch, triton, triton.language as tl
SEED = 203600
@triton.jit
def _compensated_rows(X, Out, COLS: tl.constexpr):
    row = tl.program_id(0)
    total = tl.full((), 0.0, tl.float32)
    correction = tl.full((), 0.0, tl.float32)
    for column in tl.static_range(COLS):
        value = tl.load(X + row * COLS + column)
        updated = total + value
        lost = tl.where(tl.abs(total) >= tl.abs(value),
                        (total - updated) + value,
                        (value - updated) + total)
        correction = correction + lost
        total = updated
    tl.store(Out + row, total + correction)
def run(x):
    output = torch.empty((4,), device=x.device, dtype=torch.float32)
    _compensated_rows[(4,)](x, output, COLS=12, num_warps=1, enable_fp_fusion=False)
    return output
def make_inputs_numpy():
    rng = np.random.Generator(np.random.PCG64(SEED))
    low = rng.uniform(0.25, 1.75, (4, 8)).astype(np.float32)
    interior = np.concatenate((np.full((4, 1), 2.0**30, np.float32),
                               np.full((4, 1), -(2.0**30), np.float32), low), axis=1)
    order = rng.permutation(10)
    x = np.empty((4, 12), dtype=np.float32)
    x[:, 0] = np.float32(2.0**80)
    x[:, 1:-1] = interior[:, order]
    x[:, -1] = np.float32(-(2.0**80))
    return (x,)
def make_inputs(device="cuda"):
    return tuple(torch.from_numpy(array).to(device) for array in make_inputs_numpy())
'''
d = tempfile.mkdtemp()
p = os.path.join(d, "kmod_e16.py")
open(p, "w").write(src)
import importlib.util
spec = importlib.util.spec_from_file_location("kmod_e16", p)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
x, = m.make_inputs()
before = x.clone()
out = m.run(x)
from fractions import Fraction as Fr
xn = before.detach().cpu().numpy()
target = [float(sum((Fr(v) for v in xn[r]), Fr(0))) for r in range(4)]
act = out.detach().cpu().numpy().astype(np.float64)
rel = float(np.linalg.norm(act - np.array(target)) / max(np.linalg.norm(np.array(target)), 1e-12))
res = {"output": act.tolist(), "exact_target": target, "relative_l2_error": rel,
       "tolerance": 1e-5, "passed": bool(rel <= 1e-5),
       "finite": bool(np.isfinite(act).all()), "shape": list(out.shape),
       "dtype": str(out.dtype), "input_unmodified": bool(torch.equal(x, before))}
print(json.dumps(res))