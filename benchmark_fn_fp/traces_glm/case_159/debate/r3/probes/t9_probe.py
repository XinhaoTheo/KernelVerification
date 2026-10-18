
import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_159/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev = "cuda"
# Noncommuting off-diagonal matrices alternating, nonzero bias, nonzero initial
B, N = 8, 32
M1 = torch.tensor([[0,1],[1,0]], dtype=torch.int8)
M2 = torch.tensor([[1,1],[0,1]], dtype=torch.int8)
M3 = torch.tensor([[0,1],[-1,0]], dtype=torch.int8)
mats = [M1, M2, M3]
matrix = torch.zeros(B, N, 2, 2, dtype=torch.int8, device=dev)
for b in range(B):
    for t in range(N):
        matrix[b, t] = mats[(b + t) % 3].to(dev)
g = torch.Generator("cpu").manual_seed(1)
bias = torch.randint(-32768, 32768, (B, N, 2), generator=g, dtype=torch.int16).to(dev)
initial = torch.randint(-32768, 32768, (B, 2), generator=g, dtype=torch.int16).to(dev)
try:
    out = k.run(matrix, bias, initial)
    ref = k.reference(matrix, bias, initial)
    exact = bool(torch.equal(out, ref))
    mism = (out != ref)
    print(json.dumps({"claim": "c1", "exact_match": exact,
                      "num_mismatch": int(mism.sum()),
                      "max_abs_diff": int((out - ref).abs().max()) if mism.any() else 0}))
except Exception as ex:
    print(json.dumps({"claim": "c1", "error": f"{type(ex).__name__}: {ex}"}))
