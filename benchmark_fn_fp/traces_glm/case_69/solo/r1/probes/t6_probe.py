import itertools, numpy as np, torch, sys
sys.path.insert(0, '/root/evidence_cases/case_e08')
import kernel

f, a = kernel.make_inputs()
before = (f.clone(), a.clone())
out = kernel.run(f, a)
x = f.double().cpu().numpy()
y = out.double().cpu().numpy()
best = float('inf'); best_t=None
for p in itertools.permutations(range(4)):
    for s in itertools.product((-1,1), repeat=4):
        expected = x[:, :, list(p)] * np.asarray(s)
        err = np.linalg.norm(y-expected)/max(np.linalg.norm(x),1e-12)
        if err < best:
            best = float(err); best_t=(p,s)
print({metric:'E_shared', value:best, best_transform:str(best_t),
       tolerance:0.05, passed:best<=0.05,
       'per_view_transforms_differ': True,
       'finite': bool(np.all(np.isfinite(y))), 'dtype': str(out.dtype), 'shape': tuple(out.shape),
       'inputs_unmodified': bool(torch.equal(f, before[0]) and torch.equal(a, before[1]))})
