import json, importlib.util, torch, numpy as np
spec = importlib.util.spec_from_file_location("k", "/root/pilot_cases/case_100/kernel.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
x, w = mod.make_inputs("cuda")
out = mod.run(x, w).double()
wd = w.double(); xd = x.double()
ref = wd @ xd
n = ref.numel(); den = max(float(ref.norm()), 0.001*np.sqrt(n))
E = float((out-ref).norm())/den
# fp64 quantized grid (input-construction grid) vs kernel fp32 grid
s64 = wd.abs().max(dim=1, keepdim=True)[0]/7.0
q64 = (np.clip(np.floor((wd/s64).cpu().numpy()+0.5), -7,7) * s64.cpu().numpy())
s32 = w.abs().max(dim=1, keepdim=True)[0]/7.0
q32 = (torch.clamp(torch.floor(w/s32+0.5), -7,7) * s32).double()
grid_match = float((q32.cpu().numpy()-q64).__abs__().max())
print(json.dumps({
  "E": E, "threshold": 0.12, "margin": 0.12-E, "E_le_0.12": E<=0.12,
  "finite": bool(torch.isfinite(out).all()),
  "grid_fp32_vs_fp64_maxdiff": grid_match,
  "err_norm": float((out-ref).norm()), "ref_norm": float(ref.norm()), "denom": den
}))