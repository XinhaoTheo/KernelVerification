import json, torch, numpy as np, sys
sys.path.insert(0, "/root/evidence_cases/case_e09")
import kernel

dev = "cuda"
w, v, G = kernel.make_inputs(dev)
w0 = w.double().cpu().numpy().copy(); v0 = v.double().cpu().numpy().copy()
Gd = G.double().cpu().numpy().copy(); G_bytes = G.cpu().numpy().tobytes()
G_backup = G.clone()

# reference in float64
ref_w = w0.copy(); ref_v = v0.copy()
ref_hist = []
for t in range(6):
    ref_v = 0.75*ref_v + 0.25*Gd[t]
    ref_w = ref_w - 0.03125*ref_v
    ref_hist.append(ref_w.copy())
H_target = np.stack(ref_hist)
w6, v6 = ref_w, ref_v

hist = kernel.run_sequence(w, v, G)   # retained, observed after
H = torch.stack(hist).double().cpu().numpy()
fw = w.double().cpu().numpy(); fv = v.double().cpu().numpy()

def relerr(a,b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),0.1))

hist_err = relerr(H, H_target)
w_err = relerr(fw, w6)
v_err = relerr(fv, v6)
G_unchanged = (G.cpu().numpy().tobytes() == G_bytes)
rows_equal_final = [bool(np.allclose(H[t], fw)) for t in range(6)]
alias = [bool(h.data_ptr()==w.data_ptr()) for h in hist]

print(json.dumps({
  "metric": "Frobenius/relative l2 vs float64 reference, plus aliasing ptr check",
  "history_relative_error": hist_err,
  "final_weights_relative_error": w_err,
  "final_velocity_relative_error": v_err,
  "tolerances": [0.025, 1e-5, 1e-5],
  "history_rows_equal_final_weights": rows_equal_final,
  "returned_tensors_alias_weights_storage": alias,
  "G_unchanged": G_unchanged,
  "shapes_ok": [list(H.shape)==[6,128], list(fw.shape)==[128], list(fv.shape)==[128]],
  "finite": bool(np.isfinite(H).all() and np.isfinite(fw).all() and np.isfinite(fv).all()),
  "acceptance_pass": bool(hist_err<=0.025 and w_err<=1e-5 and v_err<=1e-5 and G_unchanged),
}))
