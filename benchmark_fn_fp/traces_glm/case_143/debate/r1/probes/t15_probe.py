import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("kernel", "/root/cases/case_143/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
res = {}
for N in (32, 64, 128):
    # effective basis: row j of eff = run(e_j); compare to reference W_N rows (cols of ref basis)
    eff = torch.zeros((N, N), dtype=torch.int32, device="cuda")
    for j in range(N):
        e = torch.zeros((1, N), dtype=torch.int32, device="cuda"); e[0, j] = 1
        e16 = e.to(torch.int16)
        eff[j] = k.run(e16)[0]
    x = torch.eye(N, dtype=torch.int32, device="cuda").to(torch.int16)
    ref = k.reference(x)  # ref basis W_N: row k = coefficient vector for e_k? actually out[k,j] for e_k
    W = ref  # out[0,j] = W_N[0,j] since x = identity rows? careful: ref[i,j] = sum_l W[i,l]*x[j,l] = W[i,j]
    # kernel: eff[j] = out for e_j -> out[k] = eff_basis[k,j] -> basis columns
    basis = eff.T.contiguous()  # basis[k,j] = coefficient k when input=e_j
    exact = torch.equal(basis, W)
    # sign-change counts of kernel basis rows vs reference rows (sequency check)
    def changes(M):
        s = (M[:, 1:] != M[:, :-1]).sum(dim=1)
        return s.cpu().tolist()
    cb, cw = changes(basis), changes(W)
    # is kernel basis a row permutation of W (possibly with sign flips)?
    key = {tuple(r.tolist()): i for i, r in enumerate(W)}
    perm = []
    for r in basis:
        t = tuple(r.tolist()); nt = tuple((-r).tolist())
        if t in key: perm.append(key[t])
        elif nt in key: perm.append(-key[nt])
        else: perm.append(None)
    n_unmatched = sum(1 for p in perm if p is None)
    # is there a fixed index mapping idx such that basis[k] == W[idx[k]]?
    keyonly = {tuple(r.tolist()): i for i, r in enumerate(W)}
    n_unmatched_signed = sum(1 for r in basis if tuple(r.tolist()) not in keyonly)
    res[f"N{N}"] = {"basis_equals_W": bool(exact),
                    "kernel_sequency_counts": cb,
                    "ref_sequency_counts": cw[:8],
                    "rows_unmatched_up_to_sign": n_unmatched,
                    "rows_unmatched_exact": n_unmatched_signed,
                    "perm": perm}
print(json.dumps(res))