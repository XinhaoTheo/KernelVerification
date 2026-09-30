import torch, json, sys
sys.path.insert(0, '/root/cases/case_32')
from kernel import triton_argsort

def check(keys):
    N, W = keys.shape
    s, p = triton_argsort(keys.cuda())
    s = s.cpu(); p = p.cpu()
    sorted_ok = bool(torch.all(s[:, :-1] <= s[:, 1:]))
    gather_ok = bool(torch.all(s == torch.gather(keys.cpu(), 1, p)))
    # stability: reference stable argsort
    ref = torch.argsort(keys.cpu(), dim=1, stable=True)
    stable_ok = bool(torch.all(p == ref))
    return sorted_ok, gather_ok, stable_ok

results = {}
torch.manual_seed(0)

# 1: power-of-2 W, duplicates
k = torch.randint(0, 5, (64, 16), dtype=torch.int64)
results['W16_dup'] = check(k)
# 2: non-power-of-2 W
k = torch.randint(0, 7, (64, 13), dtype=torch.int64)
results['W13_dup'] = check(k)
k = torch.randint(-100, 100, (32, 9), dtype=torch.int64)
results['W9_rand'] = check(k)
k = torch.randint(-100, 100, (16, 3), dtype=torch.int64)
results['W3_rand'] = check(k)
k = torch.randint(-100, 100, (16, 1), dtype=torch.int64)
results['W1_rand'] = check(k)
# 3: extreme values including INT64_MAX/MIN as actual keys
k = torch.randint(-2**62, 2**62, (32, 11), dtype=torch.int64)
k[0, :] = torch.tensor([2**63-1]*5 + [0]*3 + [-2**63]*3)
k[1, :] = torch.tensor([2**63-1, -2**63, 2**63-1, -2**63, 5, 5])
results['W11_extremes'] = check(k)
# 4: all-equal row
k = torch.full((8, 10), 42, dtype=torch.int64)
results['W10_all_equal'] = check(k)
# 5: many random sizes
bad = []
for W in [2,3,4,5,6,7,8,12,17,32,33]:
    for t in range(10):
        k = torch.randint(-3, 3, (16, W), dtype=torch.int64)
        s_ok, g_ok, st_ok = check(k)
        if not (s_ok and g_ok and st_ok):
            bad.append((W, t, s_ok, g_ok, st_ok))
results['sweep_bad'] = bad

print(json.dumps({k: (list(v) if isinstance(v, tuple) else v) for k, v in results.items()}))