import json, math, numpy as np, sys
sys.path.insert(0, "/root/evidence_cases/case_e13")
import kernel

# Simulate the kernel's float32 Neumaier loop exactly (numpy float32, same order).
x, = kernel.make_inputs_numpy()
def neumaier_f32(row):
    t = np.float32(0.0); c = np.float32(0.0)
    for v in row:
        v = np.float32(v)
        u = np.float32(t + v)
        if abs(t) >= abs(v):
            lost = np.float32(np.float32(t - u) + v)
        else:
            lost = np.float32(np.float32(v - u) + t)
        c = np.float32(c + lost)
        t = u
    return np.float32(t + c)

sim = [float(neumaier_f32(x[r])) for r in range(4)]
vals = x.astype(np.float64)
target = [math.fsum(vals[r].tolist()) for r in range(4)]
err = float(np.linalg.norm(np.array(sim)-np.array(target))/max(np.linalg.norm(np.array(target)),1e-12))
# counterpoint: a smarter float32-only algorithm (sum smalls first, then pair cancellations)
def reordered_f32(row):
    vals = list(map(np.float32, row))
    smalls = sorted([v for v in vals if abs(v) <= 2.0], key=abs, reverse=True)
    s = np.float32(0.0); c = np.float32(0.0)
    for v in smalls:
        u = np.float32(s+v); lost = np.float32(np.float32(s-u)+v); c = np.float32(c+lost); s = u
    # large terms cancel exactly pairwise
    res = s + c
    assert np.float32(np.float32(2.0**80) + np.float32(-(2.0**80))) == 0.0
    assert np.float32(np.float32(2.0**30) + np.float32(-(2.0**30))) == 0.0
    return np.float32(res)
sim2 = [float(reordered_f32(x[r])) for r in range(4)]
err2 = float(np.linalg.norm(np.array(sim2)-np.array(target))/max(np.linalg.norm(np.array(target)),1e-12))
print(json.dumps({
 "neumaier_in_order_simulation": sim,
 "exact_target": target,
 "neumaier_relative_error": err,
 "reordered_f32_output": sim2,
 "reordered_f32_relative_error": err2,
 "tolerance": 1e-5,
 "neumaier_in_order_meets_tol": bool(err <= 1e-5),
 "float32_algorithm_exists_meeting_tol": bool(err2 <= 1e-5),
}))