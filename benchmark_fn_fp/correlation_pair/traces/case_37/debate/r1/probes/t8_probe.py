
import json, sys, importlib.util
import numpy as np, torch

spec = importlib.util.spec_from_file_location("k", "/root/cases/case_b/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

x, A, B = k.make_inputs("cuda")
out = k.run(x, A, B)
out_np = out.detach().cpu().numpy().astype(np.float64)

xd = x.cpu().numpy().astype(np.float64)
Ad = A.cpu().numpy().astype(np.float64)
Bd = B.cpu().numpy().astype(np.float64)
ref = Ad @ xd + Bd @ xd

num = np.linalg.norm(out_np - ref)
den = max(np.linalg.norm(ref), 0.001*np.sqrt(64))
E = num/den

res = dict(
  device=torch.cuda.get_device_name(0),
  shapes=[list(x.shape), list(A.shape), list(B.shape)],
  all_finite=bool(np.all(np.isfinite(out_np))),
  E=float(E), tolerance=0.1, verdict_exceeds=bool(E > 0.1),
  err_l2=float(num), ref_l2=float(den),
  ref_mean=float(ref.mean()), ref_min=float(ref.min()), ref_max=float(ref.max()),
  out_min=float(out_np.min()), out_max=float(out_np.max()),
  max_abs_err=float(np.max(np.abs(out_np-ref))),
  median_row_relerr=float(np.median(np.abs(out_np-ref)/np.abs(ref))),
  scale_A_row0=float(np.max(np.abs(Ad[0]))/7.0),
  scale_B_row0=float(np.max(np.abs(Bd[0]))/7.0),
)
print(json.dumps(res))
