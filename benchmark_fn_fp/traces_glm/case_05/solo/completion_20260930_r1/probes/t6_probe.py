
import torch, json, sys
sys.path.insert(0, "/root/cases/case_05")
from kernel import count_tied_at_boundary

EPS = 1e-3  # kernel's own tolerance; contract doesn't fix it, use same for reference

def ref(scores, pivot):
    above = scores[scores > pivot]
    if len(above) == 0:
        return 0
    m = above.min().item()
    return int((above - m).abs().lt(EPS).sum().item())

cases = {}
def add(name, vals, pivot):
    s = torch.tensor(vals, dtype=torch.float32)
    if (len(vals) & (len(vals)-1)) != 0:
        s = torch.cat([s, torch.full(( (1<<(len(vals)-1).bit_length()) - len(vals), ), -1e9, dtype=torch.float32)])
    cases[name] = (s, float(pivot))

# exact ties
add("exact_ties", [0.5, 0.5, 0.5, 0.1, 0.2, 0.9, 0.4, 0.3], 0.25)
# within/outside EPS of min-above
add("tolerance", [0.3000, 0.3005, 0.3015, 0.3020, 0.0001, 0.1, 0.2, 0.4], 0.2999)
# chain ties: consecutive within EPS, endpoints beyond
add("chain", [0.500, 0.5004, 0.5008, 0.5016, 0.5020, 0.1, 0.2, 0.3], 0.4)
# pivot boundary: value exactly at pivot (excluded), just above
add("pivot_boundary", [0.5, 0.5, 0.5009, 0.5010, 0.2, 0.2, 0.3, 0.4], 0.5009)
add("pivot_equal", [0.5, 0.5, 0.6, 0.6, 0.1, 0.2, 0.3, 0.4], 0.5)
# no values above pivot
add("empty", [0.1, 0.2, 0.3, 0.4, 0.05, 0.15, 0.25, 0.35], 0.9)
# all above, one min
add("all_above_single_min", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 0.0)
# small tile
add("size1_pad", [0.7], 0.1)

results = {}
mismatches = []
for name, (s, p) in cases.items():
    got = count_tied_at_boundary(s, p)
    exp = ref(s, p)
    results[name] = {"got": got, "expected": exp}
    if got != exp:
        mismatches.append(name)

# random sweep
rng = torch.Generator().manual_seed(0)
rnd_mis = 0
for i in range(50):
    n = 2 ** rng.integers(0, 8).item()  # 1..128
    s = (rng.random(n).float() * 2 - 1)
    p = float(rng.random().item()) * 2 - 1
    got = count_tied_at_boundary(s, p)
    exp = ref(s, p)
    if got != exp:
        rnd_mis += 1
        mismatches.append(f"random_{i}")

print(json.dumps({"cases": results, "mismatches": mismatches, "random_trials": 50, "random_mismatches": rnd_mis}))
