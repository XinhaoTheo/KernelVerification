import json, importlib.util, torch, numpy as np
spec = importlib.util.spec_from_file_location("kernel", "/root/pilot_cases/case_94/kernel.py")
km = importlib.util.module_from_spec(spec); spec.loader.exec_module(km)
q, k, v = km.make_inputs("cuda")
# fp32-only pipeline emulating the kernel without the fp16 round
qf, kf, vf = q.float(), k.float(), v.float()
scores = (kf @ qf) * (32 ** -0.5)
p32 = torch.softmax(scores.float(), dim=0)
out_fp32 = (p32[:, None] * vf).sum(0)
qd = q.cpu().numpy().astype(np.float64); kd = k.cpu().numpy().astype(np.float64); vd = v.cpu().numpy().astype(np.float64)
z = kd @ qd / np.sqrt(32)
p = np.exp(z - z.max()); p /= p.sum()
ref = p @ vd
E32 = np.linalg.norm(out_fp32.cpu().numpy() - ref) / max(np.linalg.norm(ref), 0.001*np.sqrt(32))
# also fp16-rounded probability pipeline on CPU for attribution
p16 = p32.cpu().numpy().astype(np.float16).astype(np.float32)
out16 = p16 @ vf.cpu().numpy()
E16 = np.linalg.norm(out16 - ref) / max(np.linalg.norm(ref), 0.001*np.sqrt(32))
print(json.dumps({"E_fp32_only": float(E32), "E_fp16_probs": float(E16),
 "ref_norm": float(np.linalg.norm(ref)), "floor": 0.001*np.sqrt(32),
 "budget": 0.001, "fp32_only_meets_budget": bool(E32 <= 0.001)}))