
import sys, json
sys.path.insert(0, "/root/pilot_cases/case_96")
import torch, numpy as np
import kernel as K

q, k, v = K.make_inputs("cuda")
out = K.run(q, k, v)
torch.cuda.synchronize()

# float64 reference on the same float32 inputs
qd = q.double().cpu().numpy(); kd = k.double().cpu().numpy(); vd = v.double().cpu().numpy()
z = kd @ qd / np.sqrt(qd.shape[0])
p = np.exp(z - z.max()); p /= p.sum()
ref = p @ vd
out_np = out.float().cpu().numpy()

norm_ref = np.linalg.norm(ref)
denom = max(norm_ref, 0.001*np.sqrt(ref.size))
E = np.linalg.norm(out_np - ref) / denom
finite = bool(np.all(np.isfinite(out_np)))

# simulate: float32 softmax pipeline with FP16 rounding of p (no triton), isolate the fp16 step
z32 = k.float().cpu().numpy().astype(np.float64)  # not this; do pure fp32:
z32 = (k.float().cpu().numpy().astype(np.float64) @ q.float().cpu().numpy().astype(np.float64))
# strict fp32 simulation in numpy is hard; use torch float32 on cpu
with torch.no_grad():
    kt = k.float().cpu(); qt = q.float().cpu(); vt = v.float().cpu()
    s = (kt @ qt) / np.sqrt(qt.shape[0], dtype=np.float32)
    p32 = torch.softmax(s.float(), dim=0)
    p16 = p32.to(torch.float16).to(torch.float32)
    sim_fp16 = (p16[:, None] * vt).sum(0).numpy()
    sim_fp32 = (p32[:, None] * vt).sum(0).numpy()
E_sim_fp16 = np.linalg.norm(sim_fp16 - ref) / denom
E_sim_fp32 = np.linalg.norm(sim_fp32 - ref) / denom

print(json.dumps({
    "E_kernel": float(E), "tolerance": 0.001, "finite": finite,
    "norm_ref": float(norm_ref), "denom": float(denom),
    "E_sim_fp32_softmax_fp16_p": float(E_sim_fp16),
    "E_sim_fp32_softmax_fp32_p": float(E_sim_fp32),
    "max_abs_err_kernel": float(np.abs(out_np - ref).max()),
}))
