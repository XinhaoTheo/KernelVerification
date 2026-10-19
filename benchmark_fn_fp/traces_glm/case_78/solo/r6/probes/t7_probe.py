import torch, sys
sys.path.insert(0, "/root/cases/case_151")
import kernel

results = {}
for width in (64, 128):
    for batch in (1, 3, 128):
        gen = torch.Generator().manual_seed(batch*100+width)
        x = torch.randint(0, 3329, (batch, width), generator=gen, dtype=torch.int32).cuda()
        out = kernel.run(x)
        ref = kernel.reference(x)
        mismatch = (out != ref).sum().item()
        results[f"w{width}_b{batch}"] = {"mismatch_count": int(mismatch),
                                        "total": batch*width,
                                        "out_dtype": str(out.dtype), "out_shape": tuple(out.shape),
                                        "out_device": str(out.device),
                                        "range_ok": bool(((out>=0)&(out<3329)).all().item())}
        # adversarial: extreme values
        x2 = torch.full((2, width), 3328, dtype=torch.int32).cuda()
        x2[1] = torch.arange(width, dtype=torch.int32).cuda() % 3329
        out2 = kernel.run(x2); ref2 = kernel.reference(x2)
        results[f"w{width}_extreme_mismatch"] = int((out2 != ref2).sum().item())

print(results)
ok = all(v == 0 for k, v in results.items() if k.endswith("mismatch") or k.endswith("mismatch_count"))
import json
print(json.dumps({"metric": "exact integer match count vs reference DFT", "all_exact": ok, "details": results}))