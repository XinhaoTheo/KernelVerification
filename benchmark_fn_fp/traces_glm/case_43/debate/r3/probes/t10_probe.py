import torch, numpy as np, json, importlib.util
spec = importlib.util.spec_from_file_location("kern", "/root/numerical_cases/case_h/kernel.py")
kern = importlib.util.module_from_spec(spec); spec.loader.exec_module(kern)

values, = kern.make_inputs("cuda")
out = kern.run(values)

ref = values.double().sum(dim=1).cpu()
outc = out.float().cpu()
E = float((outc - ref).norm() / max(ref.norm(), 0.008))

# CPU sequential fp32 simulation of exact column order
vals_np = values.cpu().numpy()
sim = np.zeros(64, dtype=np.float32)
for j in range(128):
    sim = (sim + vals_np[:, j]).astype(np.float32)
E_sim = float(np.linalg.norm(sim - ref.numpy()) / max(np.linalg.norm(ref.numpy()), 0.008))

per_row_err = (outc - ref).numpy()
print(json.dumps({
    "E_gpu": E, "E_cpu_seq_fp32": E_sim, "tol": 0.1,
    "ref_norm": float(ref.norm()),
    "max_abs_err_gpu": float(np.abs(per_row_err).max()),
    "mean_abs_err_gpu": float(np.abs(per_row_err).mean()),
    "finite": bool(torch.isfinite(out).all()),
    "E_passes_gpu": E <= 0.1,
    "E_passes_sim": E_sim <= 0.1,
    "gpu_matches_sim": bool(np.array_equal(outc.numpy(), sim)),
    "per_row_err_head": per_row_err[:8].tolist(),
}))