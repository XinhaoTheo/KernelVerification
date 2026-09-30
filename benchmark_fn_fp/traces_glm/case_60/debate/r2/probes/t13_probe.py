import numpy as np, json, sys
sys.path.insert(0, "/root/numerical_cases/case_y")
from kernel import make_inputs_numpy

query, anchors, values = make_inputs_numpy()
q = query.astype(np.float32)
a = anchors.astype(np.float32)
N, D = a.shape

# replicate kernel float32 accumulation per anchor (expanded identity, fp32 rounding)
f32 = np.float32
computed = []
for i in range(N):
    an = f32(0.0); qn = f32(0.0); dot = f32(0.0)
    for k in range(D):
        av = f32(a[i,k]); qv = f32(q[k])
        an = f32(an + f32(av*av))
        qn = f32(qn + f32(qv*qv))
        dot = f32(dot + f32(av*qv))
    norm_sum = f32(an + qn)
    twice_dot = f32(f32(2.0) * dot)
    computed.append(f32(norm_sum - twice_dot))
computed = np.array(computed, dtype=np.float32)

d64 = ((a.astype(np.float64) - q.astype(np.float64))**2).sum(axis=1)
clamped = int((computed < 0).sum())
print(json.dumps({
    "computed_fp32_distances": [float(x) for x in computed],
    "true_fp64_distances": [float(x) for x in d64],
    "negative_computed_count": clamped,
    "clamped_to_zero_count": clamped,
    "max_abs_distance_error": float(np.abs(computed.astype(np.float64) - d64).max()),
    "true_distance_range": [float(d64.min()), float(d64.max())],
}))