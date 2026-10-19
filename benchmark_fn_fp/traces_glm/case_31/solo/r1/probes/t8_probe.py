import torch, json, sys
sys.path.insert(0, '/root/cases/case_32')
from kernel import triton_argsort

def check(keys):
    N, W = keys.shape
    s, p = triton_argsort(keys.cuda())
    s = s.cpu(); p = p.cpu()
    kc = keys.cpu()
    sorted_ok = bool(torch.all(s[:, :-1] <= s[:, 1:]))
    gather_ok = bool(torch.all(s == torch.gather(kc, 1, p)))
    ref = torch.argsort(kc, dim=1, stable=True)
    stable_ok = bool(torch.all(p == ref))
    return sorted_ok, gather_ok, stable_ok

results = {}
torch.manual_seed(0)

results['W16_dup'] = check(torch.randint(0, 5, (64, 16), dtype=torch.int64))
results['W13_dup'] = check(torch.randint(0, 7, (64, 13), dtype=torch.int64))
results['W9_rand'] = check(torch.randint(-100, 100, (32, 9), dtype=torch.int64))
results['W3_rand'] = check(torch.randint(-100, 100, (16, 3), dtype=torch.int64))
results['W1_rand'] = check(torch.randint(-100, 100, (16, 1), dtype=torch.int64))
k = torch.randint(-2**62, 2**62, (32, 11), dtype=torch.int64)
k[0, :] = torch.tensor([2**63-1]*5 + [0]*3 + [-2**63]*3, dtype=torch.int64)
k[1, :] = torch.tensor([2**63-1, -2**63, 2**63-1, -2**63, 5, 5, 0, 0, 5, -2**63, 2**63-1], dtype=torch.int64)
results['W11_extremes'] = check(k)
results['W10_all_equal'] = check(torch.full((8, 10), 42, dtype=torch.int64))
bad = []
for W in [2,3,4,5,6,7,8,12,17,32,33]:
    for t in range(10):
        kk = torch.randint(-3, 3, (16, W), dtype=torch.int64)
        s_ok, g_ok, st_ok = check(kk)
        if not (s_ok and g_ok and st_ok):
            bad.append((W, t, s_ok, g_ok, st_ok))
results['sweep_bad'] = bad

print(json.dumps({k2: (list(v) if isinstance(v, tuple) else v) for k2, v in results.items()}))