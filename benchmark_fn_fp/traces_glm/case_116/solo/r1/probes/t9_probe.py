import torch, sys, json
sys.path.insert(0, "/root/cases/case_116")
import kernel

device = "cuda"
tol = lambda t: 0.01 + 0.001 * t.abs()
worst = None
nviol = 0
ncases = 0

# Construct: within each 64-lane segment, B aligns with A's signs so each split
# partial is large (~8..16, NOT a multiple of 0.25) while alignment flips sign
# across split halves so partials cancel and target is near 0.
for trial in range(60):
    gen = torch.Generator(device="cpu").manual_seed(1000 + trial)
    m, n, k = 4, 8, 512
    A = ((torch.rand((m, k), generator=gen) * 0.5)).half()  # magnitudes in (0,0.5]
    signs = torch.sign(A[0])
    # replace zeros with positive
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    Bsign = signs[:, None].repeat(1, n).clone()
    seg = k // 8
    Bsign[4*seg:, :] = -Bsign[4*seg:, :]  # flip last 4 splits' alignment
    B = (Bsign * (torch.rand((k, n), generator=gen) * 0.5).half()).half()
    B = torch.where(B == 0, torch.tensor(0.25, dtype=torch.float16), B)
    bias = torch.zeros(n, dtype=torch.float32)
    A, B, bias = A.to(device), B.to(device), bias.to(device)
    for sk in (8, 4, 2):
        out = kernel.run(A, B, bias, sk)
        ref = kernel.reference(A, B, bias, sk)
        diff = (out - ref).abs()
        v = diff > tol(ref)
        nv = int(v.sum())
        ncases += 1
        if nv > 0:
            nviol += nv
            if worst is None or float(diff.max()) > worst["max_diff"]:
                worst = {"trial": trial, "sk": sk, "violations": nv,
                         "max_diff": float(diff.max()),
                         "max_tol_at_viol": float(tol(ref)[v].max()),
                         "example_target": float(ref.flatten()[v.flatten()][0]),
                         "example_actual": float(out.flatten()[v.flatten()][0])}

print(json.dumps({"cases": ncases, "total_violations": nviol, "worst": worst,
                  "note": "large non-representable cancelling partials, target near 0"}))