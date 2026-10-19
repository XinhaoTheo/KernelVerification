import torch, sys, json
sys.path.insert(0, "/root/cases/case_116")
import kernel

device = "cuda"
tol = lambda t: 0.01 + 0.001 * t.abs()

results = []
worst = None
for trial in range(120):
    gen = torch.Generator(device="cpu").manual_seed(trial)
    m, n, k = 64, 64, 512
    a = ((torch.rand((m, k), generator=gen) - .5)).half() * 2 * .5  # in [-.5,.5]
    b = ((torch.rand((k, n), generator=gen) - .5)).half() * 2 * .5
    bias = (2 * torch.rand((n,), generator=gen) - 1).float()
    a, b, bias = a.to(device), b.to(device), bias.to(device)
    # worst magnitude: push to max 0.5 by sign trick
    a = a.sign() * 0.5
    b = b.sign() * 0.5
    for sk in (8, 4, 2):
        out = kernel.run(a, b, bias, sk)
        ref = kernel.reference(a, b, bias, sk)
        diff = (out - ref).abs()
        viol = (diff > tol(ref))
        nv = int(viol.sum())
        if nv > 0 and (worst is None or diff.max() > worst["max_diff"]):
            worst = {"trial": trial, "split_k": sk, "violations": nv,
                     "max_diff": float(diff.max()), "max_tol": float(tol(ref).max()),
                     "max_ref": float(ref.abs().max())}
        if nv > 0:
            results.append({"trial": trial, "sk": sk, "violations": nv, "max_diff": float(diff.max())})

print(json.dumps({"violating_configs": len(results), "worst": worst,
                  "note": "A,B set to +/-0.5 exactly (max magnitude), K=512"}))