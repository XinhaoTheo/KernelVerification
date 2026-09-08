import torch
import triton
import triton.language as tl

@triton.jit
def _kernel(Q, K, V, O, N: tl.constexpr, D: tl.constexpr):
    n = tl.arange(0, N)
    d = tl.arange(0, D)
    q = tl.load(Q + d)
    k = tl.load(K + n[:, None] * D + d[None, :])
    v = tl.load(V + n[:, None] * D + d[None, :])
    scores = tl.sum(k * q[None, :], 1) * (D ** -0.5)
    p = tl.exp(scores - tl.max(scores, 0))
    p = p / tl.sum(p, 0)
    p = p.to(tl.float16).to(tl.float32)
    y = tl.sum(p[:, None] * v, 0)
    tl.store(O + d, y)

def run(q, k, v):
    n, d = k.shape
    out = torch.empty(d, dtype=torch.float32, device=q.device)
    _kernel[(1,)](q, k, v, out, n, d, enable_fp_fusion=False)
    return out

CONFIG = {'family': 'attention', 'seed': 813, 'n': 128, 'd': 32, 'scale': 1.1, 'center': 0.5}

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
