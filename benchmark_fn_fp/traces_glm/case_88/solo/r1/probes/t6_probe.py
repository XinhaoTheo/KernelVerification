import torch, sys
sys.path.insert(0, "/root/cases/case_161")
import kernel

mismatch_total = 0
worst = None
for seed in range(30):
    b = [1, 2, 3, 15, 16, 17, 32, 37, 100, 2048][seed % 10]
    t, f = kernel.make_inputs(device="cuda", batch=b, seed=seed)
    out = kernel.run(t, f)
    ref = kernel.reference(t, f).to(torch.int32)
    if not torch.equal(out, ref):
        bad = (out != ref)
        n = int(bad.sum())
        mismatch_total += n
        idx = bad.nonzero()[0]
        if worst is None:
            worst = (seed, b, n, idx.tolist(), out[bad][0].item(), ref[bad][0].item())

print({"mismatch_positions": mismatch_total, "worst": worst})

# Also a targeted zero/unit test: one-hot tiles and filters exercise every coefficient.
torch.manual_seed(1)
bad = 0
for ti in range(36):
    for fi in range(9):
        t = torch.zeros(1,6,6, dtype=torch.int8, device="cuda"); t.view(-1)[ti] = 1
        f = torch.zeros(1,3,3, dtype=torch.int8, device="cuda"); f.view(-1)[fi] = 1
        out = kernel.run(t, f); ref = kernel.reference(t, f)
        bad += int((out != ref).sum())
# extreme values
t, f = kernel.make_inputs(batch=37, seed=99)
t = torch.where(torch.rand_like(t.float())<0.5, torch.tensor(-128.,device="cuda"), torch.tensor(127.,device="cuda")).to(torch.int8)
f = torch.where(torch.rand_like(f.float())<0.5, torch.tensor(-128.,device="cuda"), torch.tensor(127.,device="cuda")).to(torch.int8)
out = kernel.run(t, f); ref = kernel.reference(t, f)
bad += int((out != ref).sum())
print({"unit_and_extreme_mismatches": bad})
