import torch, json, sys
sys.path.insert(0, "/root/cases/case_35")
from kernel import blocked_sum

torch.manual_seed(0)
N = 65536
dev = "cuda"

def grid_input(mode, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    k = torch.randint(-127, 128, (N,), device=dev, generator=g, dtype=torch.int32)
    if mode == "allmax": k = torch.full((N,), 127, device=dev, dtype=torch.int32)
    if mode == "mixmax": k = torch.where(torch.rand(N, device=dev, generator=g) < 0.5, 127, -127).int()
    return (k.float()) * (2.0 ** -7)

results = {}
for mode in ["randint", "allmax", "mixmax"]:
    x = grid_input(mode, 42)
    ref = x.sum(dtype=torch.float64)
    bits = []
    for i in range(200):
        r = blocked_sum(x)
        torch.cuda.synchronize()
        bits.append(r.item())
    results[mode] = {
        "num_distinct_bit_patterns": len(set(bits)),
        "first": bits[0],
        "distinct_values": sorted(set(bits))[:5],
        "ref_fp64": ref.item(),
        "matches_ref": bits[0] == ref.item(),
        "max_abs_diff_from_ref": max(abs(b - ref.item()) for b in bits),
    }

print(json.dumps(results, indent=2))
