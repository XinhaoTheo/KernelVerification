import torch, sys, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_160/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

dev = "cuda"
mismatches = 0
total = 0
worst = None
# random sweeps, various batch sizes incl. non-multiples of 16, multiple seeds
for seed in range(12):
    for B in [1, 2, 7, 15, 16, 17, 37, 100, 257, 512]:
        t, f = k.make_inputs(device=dev, batch=B, seed=seed)
        out = k.run(t, f)
        ref = k.reference(t, f)
        bad = (out != ref)
        n = int(bad.sum()); total += B*16
        if n:
            mismatches += n
            idx = bad.nonzero()[0]
            if worst is None:
                worst = (seed, B, idx.tolist()[:3], out[bad][:5].tolist(), ref[bad][:5].tolist())
# adversarial extreme values
for seed in range(20):
    g = torch.Generator().manual_seed(seed)
    t = torch.randint(0, 2, (64, 6, 6), generator=g, dtype=torch.int8) * 127 - 1  # mixture
    # full extreme sweep: all -128 / all 127 / alternating
    for fill_t, fill_f in [(-128,-128), (127,127), (-128,127), (127,-128)]:
        t = torch.full((9, 6, 6), fill_t, dtype=torch.int8, device=dev)
        f = torch.full((9, 3, 3), fill_f, dtype=torch.int8, device=dev)
        out = k.run(t, f); ref = k.reference(t, f)
        bad = (out != ref); n = int(bad.sum()); total += 9*16
        if n:
            mismatches += n
            if worst is None:
                worst = ("extreme", fill_t, fill_f, out[bad][:5].tolist(), ref[bad][:5].tolist())
    # random extremes
    for _ in range(3):
        t = torch.from_numpy(torch.randint(-128, 128, (33,6,6)).numpy().astype("int8")).to(dev)
        f = torch.from_numpy(torch.randint(-128, 128, (33,3,3)).numpy().astype("int8")).to(dev)
        out = k.run(t, f); ref = k.reference(t, f)
        bad = (out != ref); n = int(bad.sum()); total += 33*16
        if n:
            mismatches += n
            if worst is None:
                worst = ("randext", out[bad][:5].tolist(), ref[bad][:5].tolist())
# also verify input preservation
t, f = k.make_inputs(device=dev, batch=37, seed=3)
t0, f0 = t.clone(), f.clone()
k.run(t, f)
preserved = bool(torch.equal(t, t0) and torch.equal(f, f0))
import json
print(json.dumps({"metric": "exact integer equality (atol=0,rtol=0) vs direct 9-term reference",
  "reason": "contract requires exact equality over full INT8 domain",
  "mismatch_count": mismatches, "positions_checked": total, "worst_case": worst,
  "inputs_preserved": preserved, "verdict_pass": mismatches == 0 and preserved}))
