import json
import numpy as np, torch, triton
import triton.language as tl
import importlib.util
spec = importlib.util.spec_from_file_location("kern", "/root/evidence_cases/case_e18/kernel.py")
kern = importlib.util.module_from_spec(spec); spec.loader.exec_module(kern)

xg, = kern.make_inputs()
out = kern.run(xg)
actual = out.detach().cpu().numpy().astype(np.float64)
xn, = kern.make_inputs_numpy()

# faithful fp32 Neumaier trace per row (strict fp32 rounding at each step, no fusion)
def neumaier_fp32(vals):
    t = np.float32(0.0); c = np.float32(0.0)
    for v in vals:
        v = np.float32(v)
        u = np.float32(t + v)
        lost = (t - u) + v if abs(t) >= abs(v) else (v - u) + t
        c = np.float32(c + np.float32(lost))
        t = u
    return float(np.float32(t + c))

trace = [neumaier_fp32(xn[r]) for r in range(4)]

# inspect PTX for fused multiply-add on the compensation path
ptx = kern._compensated_rows.cache[0] if False else None
ptx_text = ""
for dev_cache in kern._compensated_rows.device_caches.values():
    for k in dev_cache[0].values():
        ptx_text = k.asm.get("ptx", "")
n_fma = ptx_text.count("fma.rn.f32")
n_add = ptx_text.count("add.f32")
n_sub = ptx_text.count("sub.f32")
res = {"trace_neumaier_fp32": trace, "kernel_output": actual.tolist(),
       "trace_vs_output_max_abs": float(np.max(np.abs(np.array(trace)-actual))),
       "ptx_fma_count": n_fma, "ptx_add_f32_count": n_add, "ptx_sub_f32_count": n_sub,
       "compensation_contracted_or_lost": bool(np.max(np.abs(np.array(trace)-actual)) > 1e-4 or n_fma > 0)}
print(json.dumps(res))