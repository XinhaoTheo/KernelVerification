import torch, sys, os
sys.path.insert(0, "/root/cases/case_158")
import importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_158/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev = "cuda"
results = {}
for N in range(1, 9):
    B = 1
    # all-ones matrix, nonzero bias and initial
    matrix = torch.ones(B, N, 2, 2, dtype=torch.int8, device=dev)
    bias = torch.randint(-1000, 1000, (B, N, 2), dtype=torch.int16, device=dev)
    initial = torch.tensor([[3, -7]], dtype=torch.int16, device=dev)
    out = k.run(matrix, bias, initial)
    ref = k.reference(matrix, bias, initial)
    results[str(N)] = bool(torch.equal(out, ref))
print({"per_N_exact": results, "claim": "c1", "metric": "bitwise exact int64 equality vs sequential reference, N=1..8, all-ones matrix"})