
import torch, numpy as np, json, sys
sys.path.insert(0,"/root/cases/case_b")
import kernel as K
x,a,b = K.make_inputs("cuda")
out = K.run(x,a,b).double().cpu().numpy()
xn=x.double().cpu().numpy(); an=a.double().cpu().numpy(); bn=b.double().cpu().numpy()
ref = an@xn + bn@xn
err = out-ref
den = max(np.linalg.norm(ref), 0.001*np.sqrt(64))
E = np.linalg.norm(err)/den
# branch-wise
def q(m):
    s = np.abs(m).max(axis=1,keepdims=True)/7.0
    qq = np.clip(np.floor(m/s+0.5),-7,7)
    return (qq*s)@xn
qa = q(an); qb=q(bn)
print(json.dumps({
 "E": float(E), "norm_ref": float(np.linalg.norm(ref)), "norm_err": float(np.linalg.norm(err)),
 "finite": bool(np.isfinite(out).all()),
 "ref_mean": float(ref.mean()), "ref_min": float(ref.min()), "ref_max": float(ref.max()),
 "out_first5": out[:5].tolist(), "ref_first5": ref[:5].tolist(),
 "numpy_emu_E": float(np.linalg.norm(qa+qb-ref)/den),
 "maxabs_kernel_vs_emu": float(np.abs(out-(qa+qb)).max()),
}))
