import torch, json, sys
sys.path.insert(0, "/root/cases/case_163")
import kernel

results = {}
for N in (64, 128, 256):
    g = torch.Generator(device="cpu").manual_seed(0)
    vals = torch.randn((7, N), generator=g).to(torch.float16).cuda()
    out = kernel.run(vals)
    ref = kernel.reference(vals)  # FP64 [B, M+1, 2]
    err = (out.double() - ref).abs()
    tol = 1e-4 + 1e-6 * ref.abs()
    viol = err > tol
    # count violations per frequency bin index
    per_k = viol.any(dim=(0, 2))
    results[f"N{N}"] = {
        "max_abs_err": float(err.max()),
        "num_violations": int(viol.sum()),
        "bins_violated_any_row": per_k.tolist(),
        "n_bins_violated": int(per_k.sum()),
    }
print(json.dumps(results))