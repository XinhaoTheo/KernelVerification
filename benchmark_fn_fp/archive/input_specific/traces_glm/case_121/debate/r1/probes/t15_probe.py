import numpy as np, torch, triton, triton.language as tl, json, sys
sys.path.insert(0, "/root/numerical_cases/case_w")
import kernel

# Get the exact pivot values of the fixed workload via fp64 elimination (mask-identical algorithm)
(m,) = kernel.make_inputs_numpy()
M = m.astype(np.float64)
pivots = []
A = M.copy()
N = 8
for k in range(N):
    p = A[k, k]; pivots.append(p)
    mult = A[:, k] / p
    upd = A - np.outer(mult, A[k, :])
    for i in range(k+1, N):
        for j in range(k+1, N):
            A[i, j] = upd[i, j]
piv = np.array(pivots, dtype=np.float32)

@triton.jit
def logk(X, O, N: tl.constexpr):
    i = tl.arange(0, N)
    x = tl.load(X + i)
    tl.store(O + i, tl.log(x))

xg = torch.from_numpy(piv).cuda()
og = torch.empty(8, device="cuda", dtype=torch.float32)
logk[(1,)](xg, og, N=8, enable_fp_fusion=False)
tri = og.cpu().numpy()
# correctly-rounded reference: fp64 log then round to fp32
ref = np.log(piv.astype(np.float64)).astype(np.float32)
ulp = (np.abs(tri.astype(np.float64) - ref.astype(np.float64)) /
       np.maximum(np.abs(ref.astype(np.float64)) * np.float64(2**-23), 1e-30))
# also effect on the sum
tri_sum = float(np.float32(np.sum(tri, dtype=np.float32)))
ref_sum = float(np.float32(np.sum(ref, dtype=np.float32)))
print(json.dumps({
    "claim": "c2", "pivots_f32": piv.tolist(),
    "triton_log": tri.tolist(), "correctly_rounded_f32_log": ref.tolist(),
    "max_abs_log_err": float(np.abs(tri.astype(np.float64)-ref.astype(np.float64)).max()),
    "max_relative_log_err": float(ulp.max()),
    "triton_sum_f32": tri_sum, "correctly_rounded_sum_f32": ref_sum,
    "sum_abs_diff": abs(tri_sum - ref_sum),
    "metric": "per-value and summed deviation of tl.log vs correctly-rounded fp32 log on actual pivots"
}))
