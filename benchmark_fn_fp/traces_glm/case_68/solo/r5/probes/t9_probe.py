import torch, sys, json
sys.path.insert(0, "/root/cases/case_141")
from kernel import run, reference, make_inputs

results = {}
# boundary values
vals = [0, 1, 6, 7, 13, 14, 4294967294, 4294967295, 4294967289]
x = torch.tensor(vals, dtype=torch.uint32, device="cuda")
out = run(x); ref = reference(x.cpu())
results["boundaries_exact"] = bool(torch.equal(out.cpu(), ref))
# invariant check
v64 = x.cpu().to(torch.int64)
ok = bool(((out[:,0].cpu().to(torch.int64)*7 + out[:,1].cpu().to(torch.int64)) == v64).all())
results["boundary_7r_plus_c"] = ok
# sizes incl. non-multiples of 256 and max N
for n in [1, 255, 256, 257, 4093, 131072]:
    (idx,) = make_inputs(device="cuda", size=n, seed=n)
    o = run(idx); r = reference(idx.cpu())
    results[f"n={n}_exact"] = bool(torch.equal(o.cpu(), r))
# contiguous view with nonzero storage offset
big = torch.randint(0, 1<<32, (300,), dtype=torch.int64).to(torch.uint32).cuda()
view = big[37:267]  # contiguous, offset 37
assert view.is_contiguous()
results["offset_view_exact"] = bool(torch.equal(run(view).cpu(), reference(view.cpu())))
# input preserved
before = view.clone()
run(view)
results["input_preserved"] = bool(torch.equal(view, before))
print(json.dumps(results))
