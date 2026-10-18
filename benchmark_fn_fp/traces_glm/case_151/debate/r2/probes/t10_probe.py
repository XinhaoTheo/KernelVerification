import sys, json
sys.path.insert(0, "/root/cases/case_151")
import torch
from kernel import run, reference, make_inputs

Q = 3329
res = {}
for width in (64, 128):
    x, = make_inputs(device="cuda", batch=4, width=width, seed=3)
    out = run(x)
    ref = reference(x)
    # direct mismatch
    direct_mism = int((out != ref).sum())
    # inverse transform signature: out == forward transform with omega^{-j*k}?
    omega = pow(17, 256 // width, Q)
    inv = pow(omega, -1, Q)
    m = torch.tensor([[pow(inv, k * j, Q) for j in range(width)] for k in range(width)],
                     dtype=torch.int64, device=x.device)
    ref_inv = (x.to(torch.int64)[:, None, :] * m[None]).sum(2) % Q
    inv_mism = int((out.to(torch.int64) != ref_inv).sum())
    # permutation signature: does sorted row multiset match?
    perm_ok = all(torch.equal(torch.sort(out[b].to(torch.int64))[0], torch.sort(ref[b].to(torch.int64))[0]) for b in range(x.shape[0]))
    res[f"N{width}"] = {"direct_mismatch": direct_mism, "inverse_mismatch": inv_mism,
                        "row_multiset_matches_reference": bool(perm_ok)}

print(json.dumps({"metric": "exact mismatch vs forward and vs inverse DFT, plus row-multiset permutation check",
 "reason": "a conjugated/inverse twiddle mapping would match the inverse transform and preserve row multisets",
 **res}))