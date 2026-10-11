import torch, importlib.util, numpy as np
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_132/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev = "cuda" if torch.cuda.is_available() else "cpu"
vals = []
for i in range(0, 65536):
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
# exact q (FP64)
q_true = np.floor(np.float64(vals) * (2.0/np.pi) + 0.5)
okq = q_true == np.floor(np.float32(np.float32(vals) * np.float32(0.63661977236758134308)) + np.float32(0.5))
e = err.numpy()
# restrict to inputs where kernel q agrees with exact q: cancellation is the only remaining error source
e_ok = e[okq]
viols_ok = e_ok > 1e-6
i_worst = int(e_ok.argmax())
print("final:" + str({
  "n_inputs": len(vals),
  "n_with_correct_q": int(okq.sum()),
  "max_abs_err_correct_q": float(e_ok.max()),
  "tolerance": 1e-6,
  "n_violations_correct_q": int(viols_ok.sum()),
  "worst_x_correct_q": None if e_ok.size==0 else float(np.array(vals)[okq][i_worst]),
  "worst_err_correct_q": float(e_ok.max()),
  "n_magnitude_bins_over_1e6": int((e_ok > 1e-6).sum()),
}))