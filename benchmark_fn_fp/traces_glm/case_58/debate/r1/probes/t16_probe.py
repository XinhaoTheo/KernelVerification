import numpy as np, torch, triton, triton.language as tl, json, sys
sys.path.insert(0, "/root/numerical_cases/case_w")
import kernel

# Kernel variant identical to _logdet_kernel but stores the pivot sequence and final trailing entry
@triton.jit
def probe_kernel(Matrix, Piv, Fin, N: tl.constexpr):
    rows = tl.arange(0, N)
    columns = tl.arange(0, N)
    matrix = tl.load(Matrix + rows[:, None] * N + columns[None, :]).to(tl.float32)
    output = tl.full((), 0.0, tl.float32)
    for k in tl.static_range(0, N):
        pivot = tl.sum(tl.sum(tl.where((rows[:, None] == k) &
                                       (columns[None, :] == k), matrix, 0.0), 0), 0)
        column = tl.sum(tl.where(columns[None, :] == k, matrix, 0.0), 1)
        pivot_row = tl.sum(tl.where(rows[:, None] == k, matrix, 0.0), 0)
        multiplier = tl.div_rn(column, pivot)
        product = multiplier[:, None] * pivot_row[None, :]
        updated = matrix - product
        matrix = tl.where((rows[:, None] > k) & (columns[None, :] > k),
                          updated, matrix)
        output = output + tl.log(pivot)
        tl.store(Piv + k, pivot)
    tl.store(Fin, matrix[N-1, N-1])

(m,) = kernel.make_inputs()
piv = torch.empty(8, device="cuda", dtype=torch.float32)
fin = torch.empty(1, device="cuda", dtype=torch.float32)
probe_kernel[(1,)](m, piv, fin, N=8, num_warps=1, enable_fp_fusion=False)
kp = piv.cpu().numpy(); kf = fin.cpu().numpy()[0]

# numpy reference: identical masked elimination in fp64, and standard fp64 LU (no pivoting) pivots
A = m.cpu().numpy().astype(np.float64)
piv64 = []
B = A.copy()
for k in range(8):
    p = B[k, k]; piv64.append(p)
    mult = B[:, k] / p
    upd = B - np.outer(mult, B[k, :])
    for i in range(k+1, 8):
        for j in range(k+1, 8):
            B[i, j] = upd[i, j]
fin64 = B[7, 7]
piv64 = np.array(piv64)

# standard LU via scipy-free fallback: use np.linalg with no pivoting? use Cholesky-free approach: compare product of pivots to det
prod_piv = np.prod(piv64)
det = np.linalg.det(A)

rel = np.abs(kp.astype(np.float64) - piv64) / np.abs(piv64)
print(json.dumps({
    "claim": "c3",
    "kernel_pivots_f32": kp.tolist(),
    "numpy_fp64_masked_elim_pivots": piv64.tolist(),
    "max_pivot_rel_diff": float(rel.max()),
    "all_pivots_positive": bool((kp > 0).all()),
    "kernel_final_trailing_entry": float(kf), "numpy_fp64_final_entry": float(fin64),
    "prod_fp64_pivots": float(prod_pivots), "det_f64": float(det),
    "prod_vs_det_rel_diff": float(abs(prod_pivots - det)/abs(det)),
    "metric": "pivot sequence vs fp64 mask-identical elimination; product of pivots vs det (validates mask semantics)"
}))
