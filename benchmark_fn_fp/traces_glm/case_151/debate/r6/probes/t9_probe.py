import sys, os, torch
sys.path.insert(0, "/root/cases/case_151")
import kernel
dev = "cuda"
results = {}
mismatch = None
for width in (64, 128):
    rows = []
    for pos in (0, 1, width//2, width-1, 7):
        r = torch.zeros(1, width, dtype=torch.int32)
        r[0, pos] = 1
        rows.append(r)
    g = torch.Generator().manual_seed(123)
    rows.append(torch.randint(0, 3329, (3, width), generator=g, dtype=torch.int32))
    x = torch.cat(rows, dim=0).to(dev)
    out = kernel.run(x)
    ref = kernel.reference(x.cpu()).to(dev)
    eq = torch.equal(out, ref)
    results[f"N{width}"] = {"all_rows_exact": bool(eq), "impulse_rows_exact": bool(torch.equal(out[:5], ref[:5]))}
    if not eq and mismatch is None:
        bad = (out != ref).nonzero()
        mismatch = {"width": width, "first_bad_idx": bad[0].tolist(),
                    "kernel_val": out[bad[0][0], bad[0][1]].item(), "ref_val": ref[bad[0][0], bad[0][1]].item()}
print({"results": results, "first_mismatch": mismatch})