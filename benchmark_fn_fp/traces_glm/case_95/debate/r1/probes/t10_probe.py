import json, sys, torch
sys.path.insert(0, '/root/cases/case_167')
import kernel

results = {}
for N in [1, 255, 256, 257, 1027, 131072]:
    x = torch.randint(0, 256, (N,), dtype=torch.uint8, device='cuda')
    out = kernel.run(x)
    ref = kernel.reference(x)
    results[N] = int((out != ref).sum())
# OOB check: run on small tensor and verify neighbors of a larger buffer untouched
buf = torch.randint(0, 256, (300,), dtype=torch.uint8, device='cuda')
before = buf.clone()
kernel.run(buf[:7])
oob = int((buf != before).sum())
print(json.dumps({"mismatch_by_N": results, "oob_writes_outside_slice": oob}))