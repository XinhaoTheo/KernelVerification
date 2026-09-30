import numpy as np, torch, json, sys, os
sys.path.insert(0, "/root/numerical_cases/case_w")
import kernel

(m,) = kernel.make_inputs()
out = kernel.run(m)
out = out.cpu().numpy()
A64 = m.cpu().numpy().astype(np.float64)
sign, logdet = np.linalg.slogdet(A64)
ref = logdet
err = float(abs(out[0] - ref) / max(abs(ref), 1.0))
print(json.dumps({
    "claim": "c1", "kernel_output": out.tolist(), "reference_logdet_f64": ref,
    "relative_error": err, "tolerance": 1e-4, "passes": err <= 1e-4,
    "output_finite": bool(np.isfinite(out).all()), "shape": list(out.shape),
    "input_dtype": str(m.dtype), "metric": "||out-ref||/max(||ref||,1) per problem.txt"
}))
