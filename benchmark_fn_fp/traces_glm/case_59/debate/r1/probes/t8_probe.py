import torch, importlib.util, numpy as np
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_132/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev = "cuda" if torch.cuda.is_available() else "cpu"
# All finite FP16 values with |x| >= 1024 (both signs)
vals = []
for i in range(0, 65536):
    b = np.float16(0)
    x = np.frombuffer(np.uint16(i).tobytes(), dtype=np.float16)[0]
    if np.isfinite(x) and abs(float(x)) >= 1024:
        vals.append(float(x))
x = torch.tensor(vals, dtype=torch.float16, device=dev)
ref = torch.sin(x.to(torch.float64)).cpu()
if dev == "cuda":
    out = k.run(x).cpu().to(torch.float64)
else:
    out = k.reference(x).to(torch.float64)
err = (out - ref).abs()
# q computed as kernel does: FP32 multiply with FP32-rounded constant, add 0.5, floor
c = np.float32(0.63661977236758134308)
qq = np.floor(np.float32(np.float32(vals) * c).astype(np.float32) + np.float32(0.5))
q_true = np.floor(np.float64(np.float64(vals) * (2.0/np.pi)) + 0.5)
qmis = (qq != q_true)
i_worst = int(err.argmax())
print("final:" + str({
  "n_inputs": len(vals),
  "max_abs_err": float(err.max()),
  "tolerance": 1e-6,
  "n_violations": int((err > 1e-6).sum()),
  "n_q_mismatch": int(qmis.sum()),
  "err_at_q_mismatch_max": float(err[torch.tensor(qmis)].max()) if qmis.any() else 0.0,
  "n_violation_with_correct_q": int(((err > 1e-6).numpy() & ~qmis).sum()),
  "worst_x": vals[i_worst], "worst_err": float(err[i_worst]), "worst_q_kernel": float(qq[i_worst]), "worst_q_true": float(q_true[i_worst]),
}))