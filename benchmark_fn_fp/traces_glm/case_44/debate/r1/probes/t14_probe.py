import sys, torch, json
sys.path.insert(0, "/root/cases/case_117")
import kernel

dev = "cuda"
gen = torch.Generator(device="cpu").manual_seed(11)
shapes = [(1,1,7),(1,1,512),(1,64,33),(63,63,257),(17,33,257),(64,63,511),(3,5,2),(63,1,511),(1,33,1),(31,47,100)]
worst = None; fails = []
for M, N, K in shapes:
    a = ((torch.rand((M, K), generator=gen) - .5)).half().to(dev)
    b = ((torch.rand((K, N), generator=gen) - .5)).half().to(dev)
    bias = (2 * torch.rand((N,), generator=gen) - 1).float().to(dev)
    try:
        out = kernel.run(a, b, bias, 4)
        out8 = kernel.run(a, b, bias, 8)
    except Exception as e:
        fails.append({"M": M, "N": N, "K": K, "error": str(e)[:150]})
        continue
    ref = torch.matmul(a.double(), b.double()) + bias.double()[None, :]
    for tag, o in [("sk4", out), ("sk8", out8)]:
        err = (o.double() - ref).abs()
        tol = 0.01 + 0.001 * ref.abs()
        viol = err > tol
        exc = float((err - tol).max())
        if viol.any():
            fails.append({"M": M, "N": N, "K": K, "variant": tag, "n_viol": int(viol.sum()), "max_excess": exc})
        if worst is None or exc > worst["max_excess"]:
            worst = {"M": M, "N": N, "K": K, "variant": tag, "max_excess": exc, "max_abs_err": float(err.max())}
# input preservation check
a = ((torch.rand((63, 33), generator=gen) - .5)).half().to(dev)
b = ((torch.rand((33, 47), generator=gen) - .5)).half().to(dev)
bias = (2 * torch.rand((47,), generator=gen) - 1).float().to(dev)
a0, b0, bi0 = a.clone(), b.clone(), bias.clone()
_ = kernel.run(a, b, bias, 4)
preserved = bool((a == a0).all() and (b == b0).all() and (bias == bi0).all())
result = {"n_shapes": len(shapes), "violations": fails, "worst": worst, "inputs_preserved": preserved, "claim_confirmed": len(fails) > 0}
print("SUMMARY_LINE")
print(json.dumps(result))
