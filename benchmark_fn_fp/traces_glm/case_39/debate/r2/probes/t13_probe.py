import sys, json, numpy as np
sys.path.insert(0, "/root/numerical_cases/case_d")
import kernel

logits_t, values_t = kernel.make_inputs(device="cuda")
l = logits_t.detach().cpu().numpy().astype(np.float64)
v = values_t.detach().cpu().numpy().astype(np.float64)

scale = np.abs(l).max()/7.0
codes = np.clip(np.floor(l/scale + 0.5), -7.0, 7.0)
rounded = codes*scale

def softmax(x):
    w = np.exp(x - x.max()); return w/w.sum()
p_ref = softmax(l)
p_k = softmax(rounded)

top_idx = np.argsort(l)[::-1][:6]
info = []
for i in top_idx:
    info.append({
        "idx": int(i), "logit": float(l[i]), "rounded": float(rounded[i]),
        "round_direction": "down" if rounded[i] < l[i] else ("up" if rounded[i] > l[i] else "same"),
        "p_ref": float(p_ref[i]), "p_kernel": float(p_k[i]),
        "p_ratio": float(p_k[i]/p_ref[i]) if p_ref[i] > 0 else None,
    })

# probability redistribution metric (L1 shift among entries)
p_l1 = float(np.abs(p_k - p_ref).sum())
# output error attributable to probability shift alone (fp64 sim)
ref = p_ref @ v
sim = p_k @ v
p_only_rel = float(np.linalg.norm(sim - ref)/max(np.linalg.norm(ref), 0.004))
top_mass_ref = float(p_ref[top_idx].sum())
top_mass_k = float(p_k[top_idx].sum())

print(json.dumps({
  "metric": "top-logit rounding directions + probability redistribution (L1) + fp64 simulated output error from quantized-vs-raw logits",
  "max_abs_logits": float(np.abs(l).max()),
  "scale": float(scale),
  "distinct_codes": int(len(np.unique(codes))),
  "top6": info,
  "prob_l1_shift": p_l1,
  "prob_l1_shift_quantized_scale_pct": 100*p_l1/2,
  "p_only_relative_l2_error": p_only_rel,
  "top6_mass_ref": top_mass_ref,
  "top6_mass_kernel": top_mass_k,
}))