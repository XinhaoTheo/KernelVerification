
import sys, json
sys.path.insert(0, "/root/pilot_cases/case_96")
import torch, numpy as np
import kernel as K

q, k, v = K.make_inputs("cuda")

# float64 reference
qd = q.double().cpu().numpy(); kd = k.double().cpu().numpy(); vd = v.double().cpu().numpy()
z = kd @ qd / np.sqrt(qd.shape[0])
p = np.exp(z - z.max()); p /= p.sum()
ref = p @ vd
norm_ref = np.linalg.norm(ref)
denom = max(norm_ref, 0.001*np.sqrt(ref.size))

# FP32-only pipeline (no FP16 rounding of p): isolates FP32 score/reduction error
with torch.no_grad():
    kt = k.float().cpu(); qt = q.float().cpu(); vt = v.float().cpu()
    s = (kt @ qt) / np.sqrt(qt.shape[0], dtype=np.float32)
    p32 = torch.softmax(s.float(), dim=0)
    sim_fp32 = (p32[:, None] * vt).sum(0).numpy()
E_sim_fp32 = np.linalg.norm(sim_fp32 - ref) / denom

# also GPU fp32 pipeline via torch
with torch.no_grad():
    sG = (k.float() @ q.float()) / np.float32(np.sqrt(q.shape[0]))
    pG = torch.softmax(sG, dim=0)
    outG = (pG[:, None] * v.float()).sum(0)
    outG = outG.float().cpu().numpy()
E_gpu_fp32 = np.linalg.norm(outG - ref) / denom

print(json.dumps({
    "E_fp32_only_cpu_pipeline": float(E_sim_fp32),
    "E_fp32_only_gpu_pipeline": float(E_gpu_fp32),
    "tolerance": 0.001,
    "norm_ref": float(norm_ref),
}))
