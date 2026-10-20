import numpy as np, torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/numerical_cases/case_k/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
m_np, r_np = k.make_inputs_numpy()
A64 = m_np.astype(np.float64); b64 = r_np.astype(np.float64)
x_ref = np.linalg.solve(A64, b64)
def rich(A, b, dtype):
    x = np.zeros(16, dtype=dtype); A = A.astype(dtype); b = b.astype(dtype)
    for _ in range(64):
        x = x + (b - A @ x)
    return x
x64 = rich(A64, b64, np.float64)
x32 = rich(A64, b64, np.float32)
den = max(np.linalg.norm(x_ref), 0.004)
rel64 = np.linalg.norm(x64.astype(np.float64)-x_ref)/den
rel32 = np.linalg.norm(x32.astype(np.float64)-x_ref)/den
sym = bool(np.array_equal(m_np, m_np.T))
# device output too if available
dev = "cuda" if torch.cuda.is_available() else "cpu"
if dev == "cuda":
    m, r = k.make_inputs(dev)
    out = k.run(m, r).cpu().numpy().astype(np.float64)
    reldev = np.linalg.norm(out-x_ref)/den
else:
    reldev = None
print(json.dumps({"symmetric_f32_matrix": sym, "rel_err_f64_sim": float(rel64), "rel_err_f32_sim": float(rel32), "rel_err_device": None if reldev is None else float(reldev), "f32_rounding_shift": float(abs(rel32-rel64)), "tolerance": 0.08}))