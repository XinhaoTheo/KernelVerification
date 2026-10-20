import sys, json, torch
sys.path.insert(0, "/root/evidence_cases/case_e09")
import kernel

w, v, G = kernel.make_inputs()
w0 = w.double().clone(); v0 = v.double().clone(); G64 = G.double().clone()
w_ref, v_ref = w0, v0
for t in range(6):
    v_ref = 0.75 * v_ref + 0.25 * G64[t]
    w_ref = w_ref - 0.03125 * v_ref
kernel.run(w, v, G)
def rel(a, b):
    return float(torch.linalg.vector_norm(a.double() - b) / max(torch.linalg.vector_norm(b), 0.1))
we = rel(w, w_ref); ve = rel(v, v_ref)
finite = bool(torch.isfinite(w).all() and torch.isfinite(v).all())
print(json.dumps({"final_weights_relative_error": we, "final_velocity_relative_error": ve,
                  "state_tolerance": 1e-05, "exceeds": max(we, ve) > 1e-05,
                  "finite_float32": finite,
                  "weights_shape": list(w.shape), "velocity_shape": list(v.shape),
                  "dtypes": [str(w.dtype), str(v.dtype)]}))