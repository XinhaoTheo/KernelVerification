
import sys, torch, json
sys.path.insert(0, "/root/cases/case_151")
import kernel

dev = "cuda"
results = {}
fail = False
for width in (64, 128):
    mismatches = 0
    checked = 0
    for seed in range(8):
        g = torch.Generator("cpu").manual_seed(seed)
        x = torch.randint(0, 3329, (16, width), generator=g, dtype=torch.int32).to(dev)
        out = kernel.run(x)
        ref = kernel.reference(x)
        eq = torch.equal(out, ref)
        checked += x.numel()
        if not eq:
            mismatches += (out != ref).sum().item()
            diff = (out != ref).nonzero()[:5]
            print(width, seed, diff.tolist())
        # verify output properties
        assert out.dtype == torch.int32 and out.shape == x.shape and out.device == x.device
    # edge cases: all zeros, all 3328, batch extremes
    for x in [torch.zeros(1, width, dtype=torch.int32, device=dev),
              torch.full((128, width), 3328, dtype=torch.int32, device=dev),
              torch.randint(0, 3329, (128, width), dtype=torch.int32, device=dev)]:
        out = kernel.run(x)
        ref = kernel.reference(x)
        checked += x.numel()
        if not torch.equal(out, ref):
            mismatches += (out != ref).sum().item()
            fail = True
    results[f"N{width}"] = {"mismatched_elements": mismatches, "elements_checked": checked,
                            "exact_match": mismatches == 0}

print(json.dumps({"metric": "exact integer equality vs direct modular DFT (tol 0)",
                  "reason": "contract requires exact outputs for every canonical input",
                  **results}))
