import sys, json, numpy as np, torch
sys.path.insert(0, "/root/numerical_cases/case_u")
import kernel

coeff, off = kernel.make_inputs(device="cuda")
out = kernel.run(coeff, off)
torch.cuda.synchronize()

c64 = coeff.double().cpu().numpy(); o64 = float(off.double().cpu().numpy())
n = np.arange(64)
ref = o64 + sum(c64[0,k-1]*np.cos(2*np.pi*k*n/64) + c64[1,k-1]*np.sin(2*np.pi*k*n/64) for k in range(1,17))
got = out.double().cpu().numpy()
rel = np.linalg.norm(got-ref)/max(np.linalg.norm(ref),1e-12)
print(json.dumps({"metric":"relative L2 error vs float64 16-mode reference","shape":list(out.shape),"dtype":str(out.dtype),"finite":bool(np.isfinite(got).all()),"rel_l2":rel,"tol":0.15,"pass":rel<=0.15,"ref_norm":float(np.linalg.norm(ref))}))