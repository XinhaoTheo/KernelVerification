import sys, torch, json
sys.path.insert(0, "/root/cases/case_117")
import kernel

dev = "cuda"
gen = torch.Generator(device="cpu").manual_seed(3)
K = 512
M, N = 64, 64
# adversarial: |a|=|b|=0.5 with sign patterns chosen for cancellation
# a columns alternate signs so per-column partial sums cancel; b all +0.5
a = (torch.full((M, K), 0.5, dtype=torch.float16))
a[:, 1::2] = -0.5
b = torch.full((K, N), 0.5, dtype=torch.float16)
bias = torch.zeros(N, dtype=torch.float32)
a, b, bias = a.to(dev), b.to(dev), bias.to(dev)
# per-row sign flip variant to force near-zero targets with max magnitudes
a2 = a.clone(); a2[1::2] *= -1
b2 = b.clone(); b2[:, 1::2] = -0.5
results = {}
worst = None
for name, aa, bb in [("cancel", a, b), ("cancel2", a2, b2), ("random_max", None, None)]:
    if name == "random_max":
        aa = (torch.where(torch.rand((M, K), generator=gen) < .5, -0.5, 0.5)).half().to(dev)
        bb = (torch.where(torch.rand((K, N), generator=gen) < .5, -0.5, 0.5)).half().to(dev)
    for sk in (1, 8):
        out = kernel.run(aa, bb, bias, sk)
        ref = torch.matmul(aa.double(), bb.double()) + bias.double()[None, :]
        err = (out.double() - ref).abs()
        tol = 0.01 + 0.001 * ref.abs()
        excess = (err - tol).max().item()
        results[f"{name}_sk{sk}"] = {"max_abs_err": float(err.max()), "max_excess": float(excess), "n_viol": int((err > tol).sum().item()), "min_ref_abs": float(ref.abs().min())}
        if worst is None or excess > worst["max_excess"]:
            worst = {"case": name, "split_k": sk, "max_excess": float(excess), "max_abs_err": float(err.max()), "n_viol": int((err > tol).sum().item())}
result = {"cases": results, "worst": worst, "claim_confirmed": worst is not None and worst["max_excess"] > 0}
print("SUMMARY_LINE")
print(json.dumps(result))
