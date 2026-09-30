import json, sys
import numpy as np
import torch

sys.path.insert(0, "/root/numerical_cases/case_c")
import kernel

# --- exact in-scope inputs from the public generator ---
logits, values = kernel.make_inputs("cuda")
print("logits", tuple(logits.shape), logits.dtype, "values", tuple(values.shape), values.dtype,
      "values_contiguous", values.is_contiguous())

out = kernel.run(logits, values)
torch.cuda.synchronize()
print("output", tuple(out.shape), out.dtype)

o = out.detach().cpu().numpy().astype(np.float64)
l = logits.detach().cpu().numpy().astype(np.float64)
v = values.detach().cpu().numpy().astype(np.float64)

# --- contract reference: float64 softmax on ORIGINAL (unquantized) logits ---
m = l.max()
w = np.exp(l - m)
p = w / w.sum()
ref = p @ v  # shape (16,)

denom = max(np.linalg.norm(ref), 0.001 * np.sqrt(16))
rel_l2 = np.linalg.norm(o - ref) / denom

# --- what the kernel's own math (quantized logits) produces, in float64 ---
scale = np.abs(l).max() / 7.0
codes = np.clip(np.floor(l / scale + 0.5), -7.0, 7.0)
rounded = codes * scale
wq = np.exp(rounded - rounded.max())
pq = wq / wq.sum()
refq = pq @ v
rel_kernel_vs_quantized = np.linalg.norm(o - refq) / max(np.linalg.norm(refq), 0.001 * np.sqrt(16))
rel_quantized_vs_ref = np.linalg.norm(refq - ref) / denom

# --- plain fp32 softmax baseline (a non-quantizing implementation) ---
p32 = torch.softmax(logits, dim=0).detach().cpu().numpy().astype(np.float64)
ref32 = p32 @ v
rel_fp32_baseline = np.linalg.norm(ref32 - ref) / denom

result = {
    "metric": "relative L2 error per contract: ||output - reference||_2 / max(||reference||_2, 0.001*sqrt(16)), reference = float64 softmax on original logits",
    "reason_metric_matches_claim": "problem.txt defines exactly this error formula and requires it <= 0.02 on the fixed make_inputs() workload",
    "rel_l2_error": float(rel_l2),
    "threshold": 0.02,
    "passes_contract": bool(rel_l2 <= 0.02),
    "output_finite": bool(np.isfinite(o).all()),
    "output_shape": list(o.shape),
    "reference_norm": float(np.linalg.norm(ref)),
    "kernel_vs_its_own_quantized_ref_rel_l2": float(rel_kernel_vs_quantized),
    "quantized_ref_vs_true_ref_rel_l2": float(rel_quantized_vs_ref),
    "fp32_softmax_baseline_rel_l2": float(rel_fp32_baseline),
    "logits_absmax": float(np.abs(l).max()),
    "quantization_step": float(scale),
    "max_rounding_perturbation_nats": float(np.abs(rounded - l).max()),
    "tv_distance_quantized_vs_true_probs": float(0.5 * np.abs(pq - p).sum()),
    "output": o.tolist(),
    "reference": ref.tolist(),
}
print(json.dumps(result, indent=2))
