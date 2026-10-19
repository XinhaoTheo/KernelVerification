import torch, json, sys, random
sys.path.insert(0, "/root/cases/case_05")
from kernel import count_tied_at_boundary

dev = "cuda"
EPS = 1e-3

def ref(scores, pivot):
    above = scores[scores > pivot]
    if len(above) == 0:
        return 0
    m = above.min().item()
    return int((above - m).abs().lt(EPS).sum().item())

cases = {}
def add(name, vals, pivot):
    s = torch.tensor(vals, dtype=torch.float32, device=dev)
    n = 1 << (len(vals)-1).bit_length()
    if len(vals) != n:
        s = torch.cat([s, torch.full((n - len(vals),), -1e9, dtype=torch.float32, device=dev)])
    cases[name] = (s, float(pivot))

add("exact_ties", [0.5, 0.5, 0.5, 0.1, 0.2, 0.9, 0.4, 0.3], 0.25)
add("tolerance", [0.3000, 0.3005, 0.3015, 0.3020, 0.0001, 0.1, 0.2, 0.4], 0.2999)
add("chain", [0.500, 0.5004, 0.5008, 0.5016, 0.5020, 0.1, 0.2, 0.3], 0.4)
add("pivot_boundary", [0.5, 0.5, 0.5009, 0.5010, 0.2, 0.2, 0.3, 0.4], 0.5009)
add("pivot_equal", [0.5, 0.5, 0.6, 0.6, 0.1, 0.2, 0.3, 0.4], 0.5)
add("empty", [0.1, 0.2, 0.3, 0.4, 0.05, 0.15, 0.25, 0.35], 0.9)
add("all_above_single_min", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 0.0)
add("size1_pad", [0.7], 0.1)

results, mismatches = {}, []
for name, (s, p) in cases.items():
    got = count_tied_at_boundary(s, p)
    exp = ref(s, p)
    results[name] = {"got": got, "expected": exp}
    if got != exp:
        mismatches.append(name)

rng = torch.Generator().manual_seed(0)
rnd_mis = 0
for i in range(50):
    n = 2 ** int(torch.randint(0, 8, (1,), generator=rng).item())
    s = (torch.rand(n, generator=rng) * 2 - 1).to(dev)
    p = float((torch.rand(1, generator=rng) * 2 - 1).item())
    got = count_tied_at_boundary(s, p)
    exp = ref(s, p)
    if got != exp:
        rnd_mis += 1
        mismatches.append(f"random_{i}_n{n}")

print(json.dumps({"metric": "exact integer tie-count vs EPS-based reference", "cases": results, "mismatches": mismatches, "random_trials": 50, "random_mismatches": rnd_mis}))