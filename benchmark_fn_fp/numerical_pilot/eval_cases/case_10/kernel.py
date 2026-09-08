import torch
import triton
import triton.language as tl

@triton.jit
def _kernel(X, W, O, K: tl.constexpr):
    row = tl.program_id(0)
    j = tl.arange(0, K)
    x = tl.load(X + j)
    w = tl.load(W + row * K + j)
    scale = tl.max(tl.abs(w), 0) / 7.0
    qi = tl.minimum(7.0, tl.maximum(-7.0, tl.floor(w / scale + 0.5)))
    y = tl.sum(x * (qi * scale), 0)
    tl.store(O + row, y)

def run(x, w):
    m, k = w.shape
    out = torch.empty(m, dtype=torch.float32, device=x.device)
    _kernel[(m,)](x, w, out, k, enable_fp_fusion=False)
    return out

CONFIG = {'family': 'quantization', 'seed': 1224, 'm': 32, 'k': 256, 'mix': 0.75, 'residual': 1.0}

def make_inputs(device="cuda"):
    import numpy as np
    cfg = CONFIG
    # CPU NumPy PCG64; all inputs are then rounded once to binary32.
    rng = np.random.Generator(np.random.PCG64(cfg["seed"]))
    def tensor(x):
        return torch.from_numpy(np.asarray(x, dtype=np.float32).copy()).to(device)
    if cfg["family"] == "attention":
        n, d = cfg["n"], cfg["d"]
        q = rng.standard_normal(d)
        k = rng.standard_normal((n, d)) * cfg["scale"]
        # Structured value offset controls cancellation in the output.
        v = rng.standard_normal((n, d))
        q32, k32 = q.astype(np.float32), k.astype(np.float32)
        z = k32.astype(np.float64) @ q32.astype(np.float64) / np.sqrt(d)
        p = np.exp(z - z.max()); p /= p.sum()
        v -= cfg["center"] * (p @ v)[None, :]
        return tensor(q32), tensor(k32), tensor(v)
    if cfg["family"] == "quantization":
        m, k = cfg["m"], cfg["k"]
        w = rng.standard_normal((m, k)).astype(np.float32)
        x = rng.standard_normal(k)
        # Mix a weight direction with an independently sampled direction.
        direction = w.astype(np.float64).sum(axis=0)
        direction /= np.linalg.norm(direction)
        x /= np.linalg.norm(x)
        x = cfg["mix"] * direction + (1.0 - cfg["mix"]) * x
        wf = w.astype(np.float64)
        scale = np.max(np.abs(wf), axis=1, keepdims=True) / 7.0
        residual = (np.clip(np.floor(wf / scale + 0.5), -7, 7) * scale - wf).sum(axis=0)
        residual /= np.linalg.norm(residual)
        x += cfg["residual"] * residual
        return tensor(x), tensor(w)
    t, d = cfg["t"], cfg["d"]
    a = np.full((t, d), cfg["decay"], dtype=np.float64)
    b = rng.standard_normal((t, d)) * cfg["noise"]
    b += cfg["bias"]
    return tensor(a), tensor(b)
