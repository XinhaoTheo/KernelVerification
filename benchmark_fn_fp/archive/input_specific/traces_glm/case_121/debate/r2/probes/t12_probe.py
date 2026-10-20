
import sys, json
import numpy as np
sys.path.insert(0, "/root/numerical_cases/case_w")
from kernel import make_inputs_numpy, run
import torch

m_np = make_inputs_numpy()[0]
A = m_np.astype(np.float64)
# float64 reference on stored float32 entries
sign, logdet_ref = np.linalg.slogdet(A)

# run actual kernel
dev = "cuda" if torch.cuda.is_available() else "cpu"
m = torch.from_numpy(m_np).to(dev)
out = run(m).cpu().numpy().astype(np.float64)
kernel_rel = abs(out[0]-logdet_ref)/max(abs(logdet_ref),1.0)

# CPU float32 emulation of the exact kernel arithmetic (RN div, no fusion)
a = m_np.astype(np.float32).astype(np.float64)
# use np.float32 ops
m32 = m_np.astype(np.float32)
pivots32 = []
for k in range(8):
    pivot = np.float32(m32[k,k])
    column = m32[:,k].astype(np.float32)
    row = m32[k,:].astype(np.float32)
    mult = (column / pivot).astype(np.float32)
    prod = (mult[:,None]*row[None,:]).astype(np.float32)
    upd = (m32 - prod).astype(np.float32)
    mask = np.zeros((8,8), dtype=bool)
    for i in range(8):
        for j in range(8):
            mask[i,j] = (i>k) and (j>k)
    m32 = np.where(mask, upd, m32).astype(np.float32)
    pivots32.append(pivot)
piv = np.array(pivots32, dtype=np.float32)

# elimination-only error: float64 log of float32 pivots vs reference
elim_err = abs(np.sum(np.log(piv.astype(np.float64))) - logdet_ref)/max(abs(logdet_ref),1.0)
# full emulated float32 accumulation with exact float32 log (correctly rounded)
acc32 = np.float32(0.0)
for p in piv:
    acc32 = np.float32(acc32 + np.float32(np.log(np.float64(p))))
emul_rel = abs(float(acc32)-logdet_ref)/max(abs(logdet_ref),1.0)

# pivot-relative errors vs float64 elimination pivots
m64 = A.copy()
piv64 = []
for k in range(8):
    piv64.append(m64[k,k])
    m64[k+1:,k+1:] -= np.outer(m64[k+1:,k]/m64[k,k], m64[k,k+1:])
piv64 = np.array(piv64)
pivot_rel_err = np.abs(piv.astype(np.float64)-piv64)/np.abs(piv64)

print(json.dumps({
  "reference_logdet_f64": float(logdet_ref),
  "kernel_output": float(out[0]),
  "kernel_rel_err": float(kernel_rel),
  "tolerance": 1e-4,
  "elim_only_rel_err_f64log_of_f32_pivots": float(elim_err),
  "emulated_full_f32_rel_err": float(emul_rel),
  "pivots_f32": [float(p) for p in piv],
  "pivot_rel_err_vs_f64": [float(e) for e in pivot_rel_err],
  "min_pivot_f32": float(np.min(piv)),
  "min_pivot_positive": bool(np.all(piv>0))
}))
