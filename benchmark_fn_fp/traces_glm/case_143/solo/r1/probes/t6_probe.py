
import json, torch, sys
sys.path.insert(0, "/root/cases/case_143")
import kernel

results = []
mismatches = 0
for width in (32, 64, 128):
    for seed in range(3):
        x = kernel.make_inputs(batch=7, width=width, seed=seed)[0]
        out = kernel.run(x)
        ref = kernel.reference(x)
        eq = torch.equal(out, ref)
        if not eq:
            mismatches += 1
            results.append({"case": "random", "width": width, "seed": seed, "equal": eq,
                            "first_diff": (out[0][out[0] != ref[0]]).tolist()[:5]})
        results.append({"case": "random", "width": width, "seed": seed, "equal": eq})

# extremes
for width in (32, 64, 128):
    for fill in (-32768, 32767):
        x = torch.full((3, width), fill, dtype=torch.int16, device="cuda")
        out = kernel.run(x); ref = kernel.reference(x)
        eq = torch.equal(out, ref)
        mismatches += (not eq)
        results.append({"case": "extreme", "width": width, "fill": fill, "equal": eq})

# max magnitude stress (alternating signs)
for width in (32, 64, 128):
    vals = torch.tensor([(-32768 if (i % 2 == 0) else 32767) for i in range(width)], dtype=torch.int16, device="cuda").unsqueeze(0).repeat(2, 1)
    out = kernel.run(vals); ref = kernel.reference(vals)
    eq = torch.equal(out, ref); mismatches += (not eq)
    results.append({"case": "alternating", "width": width, "equal": eq, "max_abs": int(out.abs().max())})

# nonzero storage offset contiguous view
base = torch.randint(-32768, 32768, (70,), dtype=torch.int16, device="cuda")
x = base[5:5+64].contiguous().view(1, 64)
out = kernel.run(x); ref = kernel.reference(x)
eq = torch.equal(out, ref); mismatches += (not eq)
results.append({"case": "storage_offset", "equal": eq, "offset": x.storage_offset()})

# batch boundary B=128
x = torch.randint(-32768, 32768, (128, 64), dtype=torch.int16, device="cuda")
out = kernel.run(x); ref = kernel.reference(x)
eq = torch.equal(out, ref); mismatches += (not eq)
results.append({"case": "batch128", "equal": eq})

# input preserved?
x0 = kernel.make_inputs()[0]; xc = x0.clone()
kernel.run(x0)
results.append({"case": "input_preserved", "equal": torch.equal(x0, xc)})
print(json.dumps({"mismatches": mismatches, "results": results}))
