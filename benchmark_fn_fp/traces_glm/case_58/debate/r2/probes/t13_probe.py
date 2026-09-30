
import sys, json
import numpy as np
import torch, triton
import triton.language as tl
sys.path.insert(0, "/root/numerical_cases/case_w")
from kernel import make_inputs_numpy, run

m_np = make_inputs_numpy()[0]

# Triton kernel: replicate elimination, store pivots and per-pivot tl.log
@triton.jit
def _probe(Matrix, Piv, Logs, N: tl.constexpr):
    rows = tl.arange(0, N)
    columns = tl.arange(0, N)
    matrix = tl.load(Matrix + rows[:, None] * N + columns[None, :]).to(tl.float32)
    for k in tl.static_range(0, N):
        pivot = tl.sum(tl.sum(tl.where((rows[:, None] == k) & (columns[None, :] == k), matrix, 0.0), 0), 0)
        column = tl.sum(tl.where(columns[None, :] == k, matrix, 0.0), 1)
        pivot_row = tl.sum(tl.where(rows[:, None] == k, matrix, 0.0), 0)
        multiplier = tl.div_rn(column, pivot)
        product = multiplier[:, None] * pivot_row[None, :]
        updated = matrix - product
        matrix = tl.where((rows[:, None] > k) & (columns[None, :] > k), updated, matrix)
        tl.store(Piv + k, pivot)
        tl.store(Logs + k, tl.log(pivot))

m = torch.from_numpy(m_np).to("cuda")
piv_t = torch.empty((8,), device="cuda", dtype=torch.float32)
logs_t = torch.empty((8,), device="cuda", dtype=torch.float32)
_probe[(1,)](m, piv_t, logs_t, N=8, num_warps=1, enable_fp_fusion=False)
piv = piv_t.cpu().numpy().astype(np.float64)
logs = logs_t.cpu().numpy().astype(np.float64)

# correctly-rounded reference: float64 log rounded to float32
ref_logs = np.array([np.float32(np.log(np.float64(np.float32(p)))) for p in piv], dtype=np.float32).astype(np.float64)

sum_tl = float(np.float32(np.sum(logs_t.cpu().numpy().astype(np.float32), dtype=np.float32)))
sum_ref = float(np.sum(ref_logs))

# also full kernel output for context
ref_logdet = np.linalg.slogdet(m_np.astype(np.float64))[1]
out = run(m).cpu().numpy().astype(np.float64)
kernel_rel = abs(out[0]-ref_logdet)/max(abs(ref_logdet),1.0)

log_only_err = abs(sum_tl - sum_ref)/max(abs(sum_ref),1.0)
per_ulp = np.abs(logs - ref_logs)
print(json.dumps({
  "pivots_f32": [float(p) for p in piv],
  "tl_log_values": [float(x) for x in logs],
  "correctly_rounded_log_values": [float(x) for x in ref_logs],
  "max_abs_per_pivot_log_diff": float(np.max(per_ulp)),
  "sum_tl_log_f32": sum_tl,
  "sum_correctly_rounded_f32": sum_ref,
  "log_only_rel_err": float(log_only_err),
  "tolerance": 1e-4,
  "kernel_output_rel_err": float(kernel_rel),
  "reference_logdet_f64": float(ref_logdet)
}))
