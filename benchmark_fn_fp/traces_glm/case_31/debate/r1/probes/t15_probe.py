import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_32/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
dev = 'cuda'
results = {}
# Minimal decisive case: all-equal rows (every pair is a tie)
for W in [2,4,8]:
    keys = torch.full((8, W), 7, device=dev, dtype=torch.int64)
    s, p = m.triton_argsort(keys.clone()); torch.cuda.synchronize()
    ref = torch.argsort(keys.cpu(), dim=1, stable=True).to(dev)
    results[f"all_eq_W{W}"] = {"perm": p[0].cpu().tolist(), "stable_ref": ref[0].cpu().tolist(),
        "perm_matches_stable": bool(torch.equal(p, ref)),
        "perm_is_reversed_arange": p[0].cpu().tolist() == list(range(W-1,-1,-1))}
# Mixed duplicates row
keys = torch.tensor([[5,3,5,5,3,0,3,5]], device=dev, dtype=torch.int64)
s, p = m.triton_argsort(keys.clone()); torch.cuda.synchronize()
ref = torch.argsort(keys.cpu(), dim=1, stable=True).to(dev)
results["mixed_dup_W8"] = {"perm": p[0].cpu().tolist(), "stable_ref": ref[0].cpu().tolist(),
    "perm_matches_stable": bool(torch.equal(p, ref)),
    "sorted_keys_nondecreasing": bool((s[:,:-1] <= s[:,1:]).all().item()),
    "gather_consistent": bool(torch.equal(torch.gather(keys,1,p), s))}
results["metric"] = "exact perm equality vs torch.argsort(stable=True); ties must order by lower original index first"
print(json.dumps(results))