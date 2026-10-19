import numpy as np, torch, itertools
import importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/evidence_cases/case_e11/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
smoke, W, b, c = k.make_inputs_numpy()
Wd, bd, cd = W.astype(np.float64), b.astype(np.float64), c.astype(np.float64)
print("W=", W); print("b=", b); print("l1=", np.abs(Wd).sum(1))

def residual(x):  # x (n,12) float32
    x64 = x.astype(np.float64)
    return (np.maximum(x64 @ Wd.T - bd, 0) * cd).sum(1)

# candidate adversarial inputs
cands = []
for j in range(6):  # aligned to each neuron
    cands.append(np.sign(Wd[j])[None, :])
rng = np.random.default_rng(0)
cands.append(rng.uniform(-1,1,(4096,12)))
cands.append(rng.choice([-1.,1.],(4096,12)))
# greedy per-coordinate: coordinate maximize sum_j c_j relu(W[j]x-b[j]) is convex; try sign of each neuron and convex combos
cands.append(np.sign(Wd.sum(0))[None,:])
X = np.concatenate([np.clip(a,-1,1).astype(np.float32) for a in cands])
r = residual(X)
i = int(np.argmax(r))
print("max residual", r.max(), "at x=", X[i])
# also confirm on GPU kernel output error for best x
best = np.tile(X[i],(1,1))
xb, wb, bb, cb = [torch.from_numpy(v).to("cuda") for v in (best, W, b, c)]
actual = k.run(xb, wb, bb, cb)
x64 = torch.from_numpy(best).double().cuda()
expected = 0.25*x64[:,0]+0.5*x64[:,1] + torch.relu(x64@torch.from_numpy(Wd).cuda().T - torch.from_numpy(bd).cuda()) @ torch.from_numpy(cd).cuda()
print("kernel output", actual.item(), "expected", expected.item(), "abs err", abs(actual.item()-expected.item()))
import json
print(json.dumps({"max_residual": float(r.max()), "kernel_abs_err_at_best": abs(actual.item()-expected.item())}))
