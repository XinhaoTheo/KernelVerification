import torch, json, sys
sys.path.insert(0, "/root/cases/case_151")
import kernel

results = {}
mismatch = None
for width in (64, 128):
    for batch in (1, 3, 128):
        g = torch.Generator(device="cpu").manual_seed(1234 + batch)
        x = torch.randint(0, 3329, (batch, width), generator=g, dtype=torch.int32).cuda()
        out = kernel.run(x)
        ref = kernel.reference(x)
        eq = torch.equal(out, ref)
        results[f"N{width}_B{batch}"] = bool(eq)
        if not eq and mismatch is None:
            diff = (out != ref).nonzero()
            mismatch = {"width": width, "batch": batch, "n_diff": int(diff.shape[0]),
                        "first": [diff[0].tolist(), out[tuple(diff[0])].item(), ref[tuple(diff[0])].item()]}
print(json.dumps({"all_equal": all(results.values()), "results": results, "mismatch": mismatch}))