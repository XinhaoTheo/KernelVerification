import sys, json
import numpy as np, torch
sys.path.insert(0, "/root/pilot_cases/case_85")
import kernel

x, w = kernel.make_inputs("cuda")
out = kernel.run(x, w)
torch.cuda.synchronize()
ref = w.double() @ x.double()
err = out.double() - ref
E = err.norm().item() / max(ref.norm().item(), 0.001 * np.sqrt(32))
# fp64 simulation of same quantizer for comparison
wf = w.double().cpu().numpy()
xf = x.double().cpu().numpy()
scale = np.max(np.abs(wf), axis=1, keepdims=True) / 7.0
wq = np.clip(np.floor(wf / scale + 0.5), -7, 7) * scale
sim = wq @ xf
E_sim = np.linalg.norm(sim - wf @ xf) / max(np.linalg.norm(wf @ xf), 0.001*np.sqrt(32))
print(json.dumps({
    "metric": "contract E on exact make_inputs workload",
    "E_kernel": E,
    "E_fp64_sim_of_quantizer": float(E_sim),
    "budget": 0.12,
    "all_finite": bool(torch.isfinite(out).all().item()),
    "ref_norm": ref.norm().item(),
    "err_norm": err.norm().item(),
    "E_exceeds_budget": bool(E > 0.12),
}))