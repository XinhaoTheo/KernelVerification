import sys, json
sys.path.insert(0, "/root/cases/case_142")
import torch
import kernel

dev = "cuda"
results = []
for width in (32, 64, 128):
    for batch in (1, 5):
        gen = torch.Generator(device="cpu").manual_seed(width * 100 + batch)
        x = torch.randint(-32768, 32768, (batch, width), generator=gen, dtype=torch.int32).to(device=dev, dtype=torch.int16)
        # also extreme rows: all min, alternating extremes
        ext = torch.tensor([[32767]*width, [-32768]*width, [(-1)**i * 32767 for i in range(width)]], dtype=torch.int32).to(device=dev, dtype=torch.int16)
        if batch > 1:
            x = torch.cat([x, ext], dim=0)
        else:
            x = ext
        out = kernel.run(x)
        ref = kernel.reference(x)
        exact = torch.equal(out, ref)
        mismatches = (out != ref).sum().item() if not exact else 0
        # first mismatch detail
        detail = None
        if not exact:
            idx = (out != ref).nonzero()[0].tolist()
            detail = {"first_mismatch": idx, "out": out[tuple(idx)].item(), "ref": ref[tuple(idx)].item()}
        results.append({"width": width, "batch": x.shape[0], "exact": exact, "mismatches": mismatches, "detail": detail})
# also check input preserved and output properties on one case
x = kernel.make_inputs()[0]
xc = x.clone()
out = kernel.run(x)
pres = torch.equal(x, xc)
ok_props = out.dtype == torch.int32 and out.shape == x.shape and out.is_cuda
print(json.dumps({"results": results, "input_preserved": pres, "output_props_ok": ok_props}))
