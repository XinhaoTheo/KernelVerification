
import numpy as np, json

rng = np.random.Generator(np.random.PCG64(119130))
query = (16.0 + rng.normal(0.0, 0.5, 32)).astype(np.float32)
anchors = (query.astype(np.float64)[None, :] +
           rng.normal(0.0, 0.015625, (16, 32))).astype(np.float32)
values = rng.normal(0.0, 1.0, 16).astype(np.float32)

# Emulate kernel fp32 accumulation exactly: per-k fp32 adds, fusion disabled
def f32(x): return np.float32(x)
N, D = 16, 32
a32 = anchors.astype(np.float32); q32 = query.astype(np.float32); v32 = values.astype(np.float32)
anchor_norm = np.zeros(N, np.float32); query_norm = f32(0.0); dot = np.zeros(N, np.float32)
for k in range(D):
    a = a32[:, k].astype(np.float32); q = q32[k].astype(np.float32)
    anchor_norm = (anchor_norm + (a*a).astype(np.float32)).astype(np.float32)
    query_norm = (query_norm + (q*q).astype(np.float32)).astype(np.float32)
    dot = (dot + (a*q).astype(np.float32)).astype(np.float32)
norm_sum = (anchor_norm + query_norm).astype(np.float32)
twice_dot = (f32(2.0)*dot).astype(np.float32)
pre_clamp = (norm_sum - twice_dot).astype(np.float32)
clamped = np.maximum(pre_clamp, f32(0.0))

# true distances
a64 = anchors.astype(np.float64); q64 = query.astype(np.float64)
d_true = ((a64 - q64[None, :])**2).sum(axis=1)

n_neg = int((pre_clamp < 0).sum())
print(json.dumps({
    "n_negative_preclamp": n_neg,
    "preclamp_min": float(pre_clamp.min()),
    "preclamp_values": [float(x) for x in pre_clamp],
    "true_dist_min": float(d_true.min()),
    "true_dist_values": [float(x) for x in d_true],
    "clamped_mask": [bool(x) for x in (pre_clamp < 0)],
}))
