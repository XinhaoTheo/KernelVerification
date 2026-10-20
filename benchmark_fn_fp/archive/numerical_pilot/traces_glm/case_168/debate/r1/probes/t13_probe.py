import numpy as np, json
rng = np.random.Generator(np.random.PCG64(1221))
m, k = 32, 128
w = rng.standard_normal((m, k)).astype(np.float32)
# consume x draws to keep stream identical (not needed for W but harmless)
_ = rng.standard_normal(k)
wf = w.astype(np.float64)
scale64 = np.max(np.abs(wf), axis=1, keepdims=True)/7.0
qi64 = np.clip(np.floor(wf/scale64+0.5), -7, 7)
# kernel arithmetic: float32 scale, float32 division/add, then floor
wf32 = w.astype(np.float32)
scale32 = (np.max(np.abs(wf32), axis=1, keepdims=True)/np.float32(7.0)).astype(np.float32)
t32 = (wf32/scale32 + np.float32(0.5)).astype(np.float32)
qi32 = np.clip(np.floor(t32), -7, 7).astype(np.float32)
diff = (qi32.astype(np.float64) - qi64)
n_diff = int(np.count_nonzero(diff))
# distance of each weight from half-level boundary in fp32 ulp-ish terms
u = (wf/scale64 + 0.5)
dist_int = np.abs(u - np.round(u))
n_close_1e_3 = int(np.count_nonzero(dist_int < 1e-3))
n_close_1e_5 = int(np.count_nonzero(dist_int < 1e-5))
# E impact: use the actual x from make_inputs; if level shifts exist, quantify
rng2 = np.random.Generator(np.random.PCG64(1221))
w2 = rng2.standard_normal((m, k)).astype(np.float32)
x = rng2.standard_normal(k)
direction = w2.astype(np.float64).sum(axis=0); direction/=np.linalg.norm(direction)
x/=np.linalg.norm(x); x = 0.9*direction + 0.1*x
res = (np.clip(np.floor(wf/scale64+0.5), -7, 7)*scale64 - wf).sum(axis=0); res/=np.linalg.norm(res)
x += 0.5*res
x32 = x.astype(np.float32)
ref = wf @ x32.astype(np.float64)
y64 = (qi64*scale64) @ x32.astype(np.float64)
y32 = (qi32.astype(np.float64)*scale64) @ x32.astype(np.float64)
E64 = np.linalg.norm(y64-ref)/max(np.linalg.norm(ref), 0.001*np.sqrt(m))
E32 = np.linalg.norm(y32-ref)/max(np.linalg.norm(ref), 0.001*np.sqrt(m))
print(json.dumps({"n_qi_level_shifts": n_diff, "n_boundary_dist_lt_1e-3": n_close_1e_3,
 "n_boundary_dist_lt_1e-5": n_close_1e_5, "E_fp64_qi": float(E64), "E_fp32_qi": float(E32),
 "E_delta": float(E32-E64), "weights_total": m*k}))