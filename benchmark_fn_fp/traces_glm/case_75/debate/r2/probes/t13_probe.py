import numpy as np, torch, triton, triton.language as tl, json, math
from fractions import Fraction

SEED = 193602

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
    return x, order, low

x_np, order, low = make_inputs_numpy()
x = torch.from_numpy(x_np.copy()).to("cuda")
before = x.clone()
out = run(x)
actual = out.detach().cpu().numpy().astype(np.float64)

# exact real-number row sums of the STORED float32 values via Fraction
exact = []
for r in range(4):
    s = Fraction(0)
    for c in range(12):
        s += Fraction(float(x_np[r, c]))
    exact.append(float(s))

fsum_target = [math.fsum(x_np[r].astype(np.float64).tolist()) for r in range(4)]

rel = float(np.linalg.norm(np.array(actual) - np.array(exact)) / max(np.linalg.norm(np.array(exact)), 1e-12))
print(json.dumps({
    "metric": "kernel output vs exact real-number row sum (Fraction exact arithmetic)",
    "exact_real_target": exact,
    "fsum_target_crosscheck": fsum_target,
    "sum_of_low_values": low.astype(np.float64).sum(axis=1).tolist(),
    "kernel_output": actual.tolist(),
    "relative_error": rel,
    "tolerance": 1e-5,
    "permutation_order": order.tolist(),
    "inputs_unmodified": bool(torch.equal(x, before)),
    "finite": bool(np.isfinite(actual).all())
}))