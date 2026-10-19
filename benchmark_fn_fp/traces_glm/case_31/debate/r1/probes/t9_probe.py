
import torch, json
import importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_32/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

dev = 'cuda'
results = {}
cases = []
# duplicate-heavy rows, various W including non-power-of-two
for W in [2,3,4,5,7,8]:
    torch.manual_seed(0)
    keys = torch.randint(0, 3, (256, W), device=dev, dtype=torch.int64)
    cases.append((f"dup_W{W}", keys))
# all-equal rows
cases.append(("all_eq", torch.full((64, 4), 7, device=dev, dtype=torch.int64)))
# INT64_MAX rows
cases.append(("maxkey", torch.full((64, 3), 0x7FFFFFFFFFFFFFFF, device=dev, dtype=torch.int64)))
# random int64 full range
torch.manual_seed(1)
cases.append(("rand", torch.randint(-(2**63), 2**63-1, (512, 6), device=dev, dtype=torch.int64)))

for name, keys in cases:
    ref_perm = torch.argsort(keys.cpu(), dim=1, stable=True).to(dev)
    ref_sorted = torch.gather(keys, 1, ref_perm)
    outs = []
    for r in range(10):
        s, p = m.triton_argsort(keys.clone())
        torch.cuda.synchronize()
        outs.append((s.clone(), p.clone()))
    sorted_ok = all(torch.equal(s, ref_sorted) for s,_ in outs)
    perm_ok = all(torch.equal(p, ref_perm) for _,p in outs)
    determinism = all(torch.equal(outs[0][0], s) and torch.equal(outs[0][1], p) for s,p in outs[1:])
    perm_valid = all(bool(torch.equal(torch.sort(p[i]).values, torch.arange(p.shape[1], device=dev, dtype=torch.int64))) for n in range(1) for p in [outs[0][1]] for i in [0]) if True else False
    # perm validity all rows
    pv = True
    for p in [outs[0][1]]:
        for i in range(p.shape[0]):
            if not torch.equal(torch.sort(p[i]).values, torch.arange(p.shape[1], device=dev, dtype=torch.int64)):
                pv = False; break
    # gather consistency
    gather_ok = all(torch.equal(torch.gather(keys,1,p), s) for s,p in outs)
    results[name] = {"matches_stable_reference": bool(sorted_ok and perm_ok),
                     "sorted_keys_match_ref": bool(sorted_ok),
                     "perm_match_stable_ref": bool(perm_ok),
                     "deterministic_10_runs": bool(determinism),
                     "perm_is_permutation": bool(pv),
                     "gather_consistent": bool(gather_ok),
                     "sample_kernel_perm_row0": outs[0][1][0].cpu().tolist(),
                     "ref_perm_row0": ref_perm[0].cpu().tolist()}

print(json.dumps(results))
