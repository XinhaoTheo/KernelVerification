
import numpy as np, torch, itertools, json

rng = np.random.Generator(np.random.PCG64(194001))
center = rng.choice(np.asarray([-1,1]), size=12)
flip_probability = rng.uniform(0.05,0.4)
flips = np.where(rng.uniform(size=(6,12)) < flip_probability, -1, 1)
magnitudes = rng.choice(np.asarray([0.5,1.0]), size=(6,12))
W = (center[None,:]*flips*magnitudes).astype(np.float32)
b = (0.75*np.abs(W).sum(axis=1)).astype(np.float32)
c = np.full(6, 0.25, dtype=np.float32)

# all 2^12 vertices of [-1,1]^12
V = np.array(list(itertools.product([-1.0,1.0], repeat=12)), dtype=np.float64)
Wd = W.astype(np.float64); bd = b.astype(np.float64); cd = c.astype(np.float64)
res = np.maximum(V @ Wd.T - bd, 0.0) @ cd
print("vertices:", V.shape, "max residual:", res.max(), "argmax vertex idx:", int(res.argmax()))
print("vertex:", V[res.argmax()].tolist())
print("num vertices with residual > 1.0:", int((res > 1.0).sum()))
print("residual > 0.99 count:", int((res > 0.99).sum()))

# also random sampling in box
Xr = np.random.default_rng(0).uniform(-1,1,size=(200000,12))
rr = np.maximum(Xr @ Wd.T - bd, 0.0) @ cd
print("random max residual:", rr.max())

worst = V[res.argmax()]
# run actual kernel on worst vertex (as float32, n=1) and compare to float64 reference
import importlib.util, os
spec = importlib.util.spec_from_file_location("kernel", "/root/evidence_cases/case_e11/kernel.py")
K = importlib.util.module_from_spec(spec); spec.loader.exec_module(K)
x = torch.from_numpy(worst.astype(np.float32)).reshape(1,12).cuda()
w = torch.from_numpy(W).cuda(); bb = torch.from_numpy(b).cuda(); cc = torch.from_numpy(c).cuda()
actual = K.run(x, w, bb, cc)
expected = 0.25*worst[0] + 0.5*worst[1] + res.max()
print("kernel output at worst vertex:", actual.item(), "reference:", expected, "abs error:", abs(actual.item()-expected))
result = {"max_residual_box_vertex_enumeration": float(res.max()),
          "count_vertices_residual_gt_1": int((res>1.0).sum()),
          "kernel_abs_error_at_worst_vertex": abs(actual.item()-expected)}
print(json.dumps(result))
