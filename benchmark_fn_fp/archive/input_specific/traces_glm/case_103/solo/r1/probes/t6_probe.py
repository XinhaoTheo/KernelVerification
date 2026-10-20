import torch, json, sys
sys.path.insert(0, "/root/cases/case_34")
from kernel import blocked_sum

torch.manual_seed(0)
x = torch.randn(65536, device="cuda", dtype=torch.float32)
ref = float(x.sum().double())

results = []
for _ in range(200):
    results.append(blocked_sum(x).item())
bits = set()
for r in results:
    bits.add(torch.tensor([r], dtype=torch.float32).view(torch.int32).item())

print(json.dumps({
    "metric": "bitwise distinct float32 outputs across 200 repeated calls on identical input",
    "n": 65536,
    "num_calls": 200,
    "distinct_bit_patterns": len(bits),
    "sample_values": results[:5],
    "ref_fp64": ref,
    "max_abs_err_vs_fp64": max(abs(r - ref) for r in results),
}))
