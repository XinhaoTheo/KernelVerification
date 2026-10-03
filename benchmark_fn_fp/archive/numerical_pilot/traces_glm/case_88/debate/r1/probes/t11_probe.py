import json, torch, triton, triton.language as tl, numpy as np

@triton.jit
def _kernel(Q, K, V, O, N: tl.constexpr, D: tl.constexpr):
    n = tl.arange(0, N); d = tl.arange(0, D)
    q = tl.load(Q + d)
    k = tl.load(K + n[:, None] * D + d[None, :])
    v = tl.load(V + n[:, None] * D + d[None, :])
    scores = tl.sum(k * q[None, :], 1) * (D ** -0.5)
    p = tl.exp(scores - tl.max(scores, 0))
    p = p / tl.sum(p, 0)
    p = p.to(tl.float16).to(tl.float32)
    y = tl.sum(p[:, None] * v, 0)
    tl.store(O + d, y)

@triton.jit
def _kernel_nofp16(Q, K, V, O, N: tl.constexpr, D: tl.constexpr):
    n = tl.arange(0, N); d = tl.arange(0, D)
    q = tl.load(Q + d)
    k = tl.load(K + n[:, None] * D + d[None, :])
    v = tl.load(V + n[:, None] * D + d[None, :])
    scores = tl.sum(k * q[None, :], 1) * (D ** -0.5)
    p = tl.exp(scores - tl.max(scores, 0))
    p = p / tl.sum(p, 0)
    y = tl.sum(p[:, None] * v, 0)
    tl.store(O + d, y)

cfg = {'seed': 812, 'n': 64, 'd': 32, 'scale': 0.7, 'center': 0.9}
rng = np.random.Generator(np.random.PCG64(cfg['seed']))
n, d = cfg['n'], cfg['d']
q = rng.standard_normal(d)
k = rng.standard_normal((n, d)) * cfg['scale']
v = rng.standard_normal((n, d))
q32, k32 = q.astype(np.float32), k.astype(np.float32)
z = k32.astype(np.float64) @ q32.astype(np.float64) / np.sqrt(d)
p = np.exp(z - z.max()); p /= p.sum()
v -= cfg['center'] * (p @ v)[None, :]
def tensor(x): return torch.from_numpy(np.asarray(x, dtype=np.float32).copy()).cuda()
qt, kt, vt = tensor(q32), tensor(k32), tensor(v)

out = torch.empty(d, dtype=torch.float32, device=qt.device)
_kernel[(1,)](qt, kt, vt, out, n, d, enable_fp_fusion=False)
out32 = torch.empty(d, dtype=torch.float32, device=qt.device)
_kernel_nofp16[(1,)](qt, kt, vt, out32, n, d, enable_fp_fusion=False)

# float64 reference on same float32 inputs
qf, kf, vf = qt.cpu().numpy().astype(np.float64), kt.cpu().numpy().astype(np.float64), vt.cpu().numpy().astype(np.float64)
zf = kf @ qf / np.sqrt(d)
pf = np.exp(zf - zf.max()); pf /= pf.sum()
ref = pf @ vf

ko, no = out.cpu().numpy(), out32.cpu().numpy()
denom = max(np.linalg.norm(ref), 0.001*np.sqrt(ref.size))
E_kernel = np.linalg.norm(ko - ref)/denom
E_nofp16 = np.linalg.norm(no - ref)/denom
E_fp16_contrib = np.linalg.norm(ko - no)/denom
p2 = float((pf**2).sum())
print(json.dumps({
  "E_kernel": E_kernel, "E_kernel_le_0.001": bool(E_kernel <= 0.001),
  "E_nofp16_variant": E_nofp16, "E_fp16_contribution": E_fp16_contrib,
  "norm_ref": float(np.linalg.norm(ref)), "denom": denom,
  "sum_p2": p2, "all_finite": bool(np.isfinite(ko).all())
}))