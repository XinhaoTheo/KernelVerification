import json, torch, sys
sys.path.insert(0, '/root/cases/case_151')
import kernel

out = {}
width = 64
# structured probes (impulse, constant) plus random rows
torch.manual_seed(7)
rows = []
imp = torch.zeros(width, dtype=torch.int32); imp[1] = 1
rows.append(imp)  # impulse at j=1 -> out[k] = omega^k, natural order check
rows.append(torch.full((width,), 5, dtype=torch.int32))  # constant -> only k=0 nonzero
x = torch.stack(rows).cuda()
ref = kernel.reference(x)
got = kernel.run(x)
out['structured_exact'] = bool((ref == got).all())
if not out['structured_exact']:
    out['impulse_ref'] = ref[0].tolist()
    out['impulse_got'] = got[0].tolist()

xr = torch.randint(0, 3329, (8, width), dtype=torch.int32, device='cuda')
refr = kernel.reference(xr)
gotr = kernel.run(xr)
exact = bool((refr == gotr).all())
out['random_exact'] = exact
if not exact:
    # is the mismatch a pure row-wise permutation of outputs?
    sorted_ref, _ = torch.sort(refr, dim=1)
    sorted_got, _ = torch.sort(gotr, dim=1)
    out['sorted_rows_equal'] = bool((sorted_ref == sorted_got).all())
print(json.dumps(out))