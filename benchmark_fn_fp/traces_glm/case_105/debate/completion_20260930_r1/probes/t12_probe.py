import numpy as np, json, torch
CONFIG = {'family': 'quantization', 'seed': 1221, 'm': 32, 'k': 128, 'mix': 0.9, 'residual': 0.5}
rng = np.random.Generator(np.random.PCG64(CONFIG["seed"]))
m, k = CONFIG["m"], CONFIG["k"]
w = rng.standard_normal((m, k)).astype(np.float32)
x = rng.standard_normal(k)
direction = w.astype(np.float64).sum(axis=0); direction /= np.linalg.norm(direction)
x /= np.linalg.norm(x)
x = CONFIG["mix"]*direction + (1-CONFIG["mix"])*x
wf = w.astype(np.float64)
scale = np.max(np.abs(wf), axis=1, keepdims=True)/7.0
residual = (np.clip(np.floor(wf/scale+0.5), -7, 7)*scale - wf).sum(axis=0)
residual /= np.linalg.norm(residual)
x += CONFIG["residual"]*residual
x32 = x.astype(np.float32); w32 = w  # rounded once to binary32
ref = w32.astype(np.float64) @ x32.astype(np.float64)
# float64 quantization model (what make_inputs' residual assumed)
wq = np.clip(np.floor(w32.astype(np.float64)/scale+0.5), -7, 7)*scale
yq = wq @ x32.astype(np.float64)
E_model = np.linalg.norm(yq-ref)/max(np.linalg.norm(ref), 0.001*np.sqrt(m))
out = None; E_kernel = None; finite = None
try:
    xt = torch.from_numpy(x32.copy()).to("cuda"); wt = torch.from_numpy(w32.copy()).to("cuda")
    import kernel as K  # not available; inline instead
except Exception:
    pass
try:
    import triton, triton.language as tl
    @triton.jit
    def _kernel(X, W, O, K_: tl.constexpr):
        row = tl.program_id(0)
        j = tl.arange(0, K_)
        x_ = tl.load(X + j)
        w_ = tl.load(W + row * K_ + j)
        sc = tl.max(tl.abs(w_), 0) / 7.0
        qi = tl.minimum(7.0, tl.maximum(-7.0, tl.floor(w_ / sc + 0.5)))
        y = tl.sum(x_ * (qi * sc), 0)
        tl.store(O + row, y)
    def run(x, w):
        mm, kk = w.shape
        o = torch.empty(mm, dtype=torch.float32, device=x.device)
        _kernel[(mm,)](x, w, o, kk, enable_fp_fusion=False)
        return o
    if torch.cuda.is_available():
        xt = torch.from_numpy(x32.copy()).to("cuda")
        wt = torch.from_numpy(w32.copy()).to("cuda")
        out = run(xt, wt).cpu().numpy().astype(np.float64)
        E_kernel = np.linalg.norm(out-ref)/max(np.linalg.norm(ref), 0.001*np.sqrt(m))
        finite = bool(np.all(np.isfinite(out)))
except Exception as e:
    out = str(e)
print(json.dumps({"E_fp64_model": float(E_model), "E_kernel_gpu": None if E_kernel is None else float(E_kernel),
  "finite": finite, "ref_norm": float(np.linalg.norm(ref)),
  "norm_thresh": float(0.001*np.sqrt(m)), "out": out if isinstance(out, str) else None,
  "gpu_available": torch.cuda.is_available()}))