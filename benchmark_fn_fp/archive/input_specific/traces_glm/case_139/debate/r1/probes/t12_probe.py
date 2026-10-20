
import json, math, importlib.util, numpy as np, torch

spec = importlib.util.spec_from_file_location("k", "/root/evidence_cases/case_e15/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

x, = k.make_inputs()
before = x.clone()
out = k.run(x)
gpu = out.detach().cpu().numpy().astype(np.float64).tolist()

xf = x.detach().cpu().numpy().astype(np.float64)
exact = [math.fsum(row) for row in xf]  # exact real-number sum, correctly rounded

rel = float(np.linalg.norm(np.array(gpu)-np.array(exact))/max(np.linalg.norm(exact),1e-12))

# CPU simulation of the kernel's fp32 Neumaier accumulation to expose mechanism
xs = x.detach().cpu().numpy()
sim = []
for row in xs:
    total = np.float32(0.0); corr = np.float32(0.0)
    trace = []
    for v in row:
        v = np.float32(v)
        updated = np.float32(total + v)
        lost = (total - updated) + v if abs(total) >= abs(v) else (v - updated) + total
        lost = np.float32(lost)
        new_corr = np.float32(corr + lost)
        trace.append(float(corr))  # correction before update
        corr = new_corr; total = updated
    sim.append(float(total + corr))

struct = (tuple(out.shape)==(4,) and out.dtype==torch.float32
          and bool(np.isfinite(out.detach().cpu().numpy()).all())
          and bool(torch.equal(x, before)))

print(json.dumps({
 "gpu_output": gpu, "exact_target_fsum": exact,
 "relative_error_vs_exact": rel, "tolerance": 1e-5,
 "within_tolerance": rel <= 1e-5,
 "cpu_fp32_neumaier_sim": sim,
 "structural_ok": struct,
 "input_row_order_interior": xs[:,1:-1].tolist()
}))
