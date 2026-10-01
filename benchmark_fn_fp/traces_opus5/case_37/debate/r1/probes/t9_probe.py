
import json, importlib.util
import numpy as np, torch

spec = importlib.util.spec_from_file_location("k", "/root/cases/case_b/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

x, A, B = k.make_inputs("cuda")
gpu_out = k.run(x, A, B).detach().cpu().numpy().astype(np.float64)

xd = x.cpu().numpy().astype(np.float64)
Ad = A.cpu().numpy().astype(np.float64)
Bd = B.cpu().numpy().astype(np.float64)

def qdeq(W):
    s = np.max(np.abs(W), axis=1, keepdims=True)/7.0
    q = np.clip(np.floor(W/s + 0.5), -7.0, 7.0)
    return q*s, q, s

Aq,qa,sa = qdeq(Ad)
Bq,qb,sb = qdeq(Bd)

yA_ref = Ad@xd; yB_ref = Bd@xd
yA_q  = Aq@xd;  yB_q  = Bq@xd
eA = yA_q - yA_ref
eB = yB_q - yB_ref
comb = eA + eB
emul = yA_q + yB_q

ref = yA_ref + yB_ref
den = max(np.linalg.norm(ref), 0.001*np.sqrt(64))

nA, nB, nC = np.linalg.norm(eA), np.linalg.norm(eB), np.linalg.norm(comb)
corr = float(np.corrcoef(eA, eB)[0,1])

res = dict(
  emul_vs_gpu_max_abs_diff=float(np.max(np.abs(emul-gpu_out))),
  emul_E=float(nC/den), gpu_E=float(np.linalg.norm(gpu_out-ref)/den),
  errA_l2=float(nA), errB_l2=float(nB), err_combined_l2=float(nC),
  quadrature_pred=float(np.sqrt(nA**2+nB**2)),
  combined_gt_max_branch=bool(nC > max(nA,nB)),
  branch_err_corr=corr,
  EA_only=float(nA/den), EB_only=float(nB/den),
  ref_l2=float(den),
  codes_outside_range=int(np.sum((np.abs(qa)>7)|(np.abs(qb)>7))),
  n_codes_at_clamp=int(np.sum(np.abs(qa)==7)+np.sum(np.abs(qb)==7)),
  mean_scale_A=float(sa.mean()), mean_scale_B=float(sb.mean()),
)
print(json.dumps(res))
