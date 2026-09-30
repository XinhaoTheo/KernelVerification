import sys, json, torch, numpy as np
sys.path.insert(0, "/root/evidence_cases/case_e09")
import kernel

w, v, G = kernel.make_inputs()
w64 = w.double().clone(); v64 = v.double().clone(); G64 = G.double().clone()
hist = kernel.run_sequence(w, v, G)  # retained views, observed after full sequence
# float64 target history w[1..6]
targets = []
for t in range(6):
    v64 = 0.75 * v64 + 0.25 * G64[t]
    w64 = w64 - 0.03125 * v64
    targets.append(w64.clone())
H_target = torch.stack(targets)
H = torch.stack([h.double() for h in hist])
err = float(torch.linalg.norm(H - H_target) / max(torch.linalg.norm(H_target), 0.1))
rows_equal_w6 = [bool(torch.equal(h.double(), H[5].double())) for h in hist]
print(json.dumps({"history_relative_error": err, "tolerance": 0.025,
                  "exceeds_tolerance": err > 0.025,
                  "all_rows_equal_final_w6": all(rows_equal_w6),
                  "rows_equal_w6": rows_equal_w6,
                  "H_target_fro": float(torch.linalg.norm(H_target))}))