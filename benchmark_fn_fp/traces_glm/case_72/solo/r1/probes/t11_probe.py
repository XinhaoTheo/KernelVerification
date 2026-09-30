import numpy as np, torch, itertools, json
import importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/evidence_cases/case_e11/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
smoke, W, b, c = k.make_inputs_numpy()
Wd, bd, cd = W.astype(np.float64), b.astype(np.float64), c.astype(np.float64)
V = np.array(list(itertools.product([-1.0,1.0], repeat=12)), dtype=np.float32)
X = torch.from_numpy(V).cuda()
Wt, bt, ct = [torch.from_numpy(v).cuda() for v in (W, b, c)]
actual = k.run(X, Wt, bt, ct)
x64 = X.double()
expected = 0.25*x64[:,0] + 0.5*x64[:,1] + torch.relu(x64 @ torch.from_numpy(Wd).cuda().T - torch.from_numpy(bd).cuda()) @ torch.from_numpy(cd).cuda()
err = (actual.double() - expected).abs()
print("vertex max err", err.max().item(), "argmax x", V[int(err.argmax())])
print("min err", err.min().item())
before = [v.clone() for v in (X, Wt, bt, ct)]
print("shape", actual.shape, actual.dtype, "finite", bool(torch.isfinite(actual).all()),
      "inputs_unmodified", all(torch.equal(a,z) for a,z in zip((X,Wt,bt,ct), before)))
print(json.dumps({"vertex_max_abs_err": float(err.max()), "vertex_min_abs_err": float(err.min()),
                  "tolerance": 1.0, "within_tolerance": bool(err.max() <= 1.0)}))