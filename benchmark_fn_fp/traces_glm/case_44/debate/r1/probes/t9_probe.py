import sys, torch
sys.path.insert(0, "/root/cases/case_117")
import kernel

dev = "cuda"
gen = torch.Generator(device="cpu").manual_seed(7)
worst = None
fail = None
n_fail = 0
combos = []
for K in list(range(1, 65)) + [96, 127, 128, 129, 255, 256, 257, 383, 384, 511, 512]:
    for sk in (1, 2, 4, 8):
        M, N = 17, 33
        a = ((torch.rand((M, K), generator=gen) - .5)).half()
        b = ((torch.rand((K, N), generator=gen) - .5)).half()
        bias = (2 * torch.rand((N,), generator=gen) - 1).float()
        a, b, bias = a.to(dev), b.to(dev), bias.to(dev)
        try:
            out = kernel.run(a, b, bias, sk)
        except Exception as e:
            fail = {"K": K, "split_k": sk, "error": str(e)[:200]}
            n_fail += 1
            combos.append(f"K{K}/sk{sk}:EXC")
            continue
        ref = torch.matmul(a.double(), b.double()) + bias.double()[None, :]
        err = (out.double() - ref).abs()
        tol = 0.01 + 0.001 * ref.abs()
        viol = (err > tol)
        max_excess = float((err - tol).max())
        combos.append(f"K{K}/sk{sk}:{'OK' if not viol.any() else 'FAIL'}")
        if viol.any() and (worst is None or max_excess > worst["max_excess"]):
            worst = {"K": K, "split_k": sk, "n_violations": int(viol.sum().item()), "max_excess": max_excess, "max_abs_err": float(err.max())}
# also split-invariance check on one nontrivial K
a = ((torch.rand((17, 257), generator=gen) - .5)).half().to(dev)
b = ((torch.rand((257, 33), generator=gen) - .5)).half().to(dev)
bias = (2 * torch.rand((33,), generator=gen) - 1).float().to(dev)
outs = {sk: kernel.run(a, b, bias, sk) for sk in (1,2,4,8)}
inv = {f"sk{sk}_vs_sk1": float((outs[sk] - outs[1]).abs().max()) for sk in (2,4,8)}
result = {"n_combos": len(combos), "n_fail": n_fail, "first_fail": fail, "worst_violation": worst, "split_invariance_max_diff": inv, "all_ok": worst is None and fail is None}
print("SUMMARY_LINE")
import json
print(json.dumps(result))
