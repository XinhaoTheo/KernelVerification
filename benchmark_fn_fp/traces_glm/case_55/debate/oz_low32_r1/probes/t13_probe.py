import numpy as np, torch, triton, triton.language as tl, json
rng = np.random.Generator(np.random.PCG64(711076))
a = rng.normal(0.0, 0.1, (4,8)).astype(np.float32)
f = rng.uniform(0.5, 220.0, (4,8)).astype(np.float32)
p = rng.uniform(-np.pi, np.pi, (4,8)).astype(np.float32)
pts = (np.arange(32)+0.5)/32.0
# exact float32 angles as computed in kernel: angle = f32(f*point) + f32(phase)
F32 = torch.from_numpy(f).cuda(); P32 = torch.from_numpy(p).cuda()
ang_f32 = (F32[None,:,None]*torch.from_numpy(pts.astype(np.float32)).cuda()[None,None,:] + P32[None,:,None])  # (32,4,8) order of ops may differ; emulate kernel: f*point then +phase
pt_t = torch.from_numpy(pts.astype(np.float32)).cuda()
ang = (F32[:, :, None]*pt_t[None, None, :] + P32[:, :, None]).permute(2,0,1).contiguous()  # (32,4,8) float32
@triton.jit
def _sin_probe(Ang, Out, N: tl.constexpr, BLOCK: tl.constexpr):
    off = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    m = off < N
    x = tl.load(Ang + off, mask=m)
    tl.store(Out + off, tl.sin(x), mask=m)
N = ang.numel()
ang_flat = ang.reshape(-1)
out = torch.empty_like(ang_flat)
_sin_probe[(triton.cdiv(N,1024),)](ang_flat, out, N=N, BLOCK=1024, enable_fp_fusion=False)
gpu = out.reshape(32,4,8).cpu().numpy()
ref = np.sin(ang.numpy().astype(np.float64))
# contribution to per-row integral: sum over points * a / 32
sin_err = np.abs(gpu - ref)  # (32,4,8)
per_row_err = np.sum(np.abs(a.astype(np.float64)[None]*sin_err), axis=(0,2))/32.0
print(json.dumps({"n_angles": int(N), "max_abs_angle": float(np.abs(ang.numpy()).max()),
 "max_sin_abs_err": float(sin_err.max()), "mean_sin_abs_err": float(sin_err.mean()),
 "per_row_integral_err_from_sin": per_row_err.tolist(),
 "tolerance_budget_per_row_approx": 0.035,
 "note": "per_row_integral_err is the worst-case contribution of tl.sin error to each row's integral assuming errors add coherently"}))