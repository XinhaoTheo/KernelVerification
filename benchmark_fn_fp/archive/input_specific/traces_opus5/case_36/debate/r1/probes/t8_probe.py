
import json, importlib.util, math
import numpy as np, torch

spec = importlib.util.spec_from_file_location("k", "/root/cases/case_a/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

x, A, B = k.make_inputs("cuda")
out = k.run(x, A, B)
torch.cuda.synchronize()

xn = x.detach().cpu().numpy().astype(np.float64)
An = A.detach().cpu().numpy().astype(np.float64)
Bn = B.detach().cpu().numpy().astype(np.float64)
ref = An @ xn + Bn @ xn
o = out.detach().cpu().numpy().astype(np.float64)

num = float(np.linalg.norm(o - ref))
den = max(float(np.linalg.norm(ref)), 0.001*math.sqrt(64))
E = num/den

# CPU emulation of the kernel's quantization for diagnostics
def q(M):
    s = np.max(np.abs(M), axis=1, keepdims=True)/7.0
    qq = np.clip(np.floor(M/s + 0.5), -7, 7)
    return qq*s, s, qq
Aq, sa, qa = q(An); Bq, sb, qb = q(Bn)
emu = Aq @ xn + Bq @ xn
E_emu = float(np.linalg.norm(emu-ref))/den
errA = float(np.linalg.norm(Aq@xn - An@xn))
errB = float(np.linalg.norm(Bq@xn - Bn@xn))

print(json.dumps({
 "E": E, "threshold": 0.1, "num": num, "den": den,
 "ref_norm": float(np.linalg.norm(ref)),
 "ref_min": float(ref.min()), "ref_max": float(ref.max()),
 "E_cpu_emulation": E_emu,
 "branchA_abs_err_norm": errA, "branchB_abs_err_norm": errB,
 "max_abs_err": float(np.abs(o-ref).max()),
 "scale_a_min": float(sa.min()), "scale_a_max": float(sa.max()),
 "clamp_hits_A": int(np.sum(np.abs(qa)>=7)), "clamp_hits_B": int(np.sum(np.abs(qb)>=7)),
 "all_finite": bool(np.isfinite(o).all()),
 "gpu": torch.cuda.get_device_name(0)
}))
