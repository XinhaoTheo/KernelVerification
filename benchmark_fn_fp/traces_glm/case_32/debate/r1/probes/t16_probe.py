import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_32/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
dev = 'cuda'
results = {}
for W in [2,3,4,5,6,7,8,9,12]:
    viol = {}
    for trial in range(5):
        torch.manual_seed(trial)
        keys = torch.randint(-1000, 1000, (512, W), device=dev, dtype=torch.int64)
        s, p = m.triton_argsort(keys.clone()); torch.cuda.synchronize()
        ref_sorted = torch.sort(keys, dim=1).values
        nondec = (s[:, :-1] <= s[:, 1:]).all().item() if W > 1 else True
        viol[f"trial{trial}"] = {"nondecreasing": bool(nondec),
            "matches_sorted_ref": bool(torch.equal(s, ref_sorted)),
            "perm_is_perm": bool(torch.equal(torch.sort(p, dim=1).values, torch.arange(W, device=dev).unsqueeze(0).expand(512,-1)).all().item())}
    results[f"W{W}"] = {"any_nondecreasing_violation": any(not v["nondecreasing"] for v in viol.values()),
        "any_mismatch_vs_sorted_ref": any(not v["matches_sorted_ref"] for v in viol.values()),
        "perm_always_valid": all(v["perm_is_perm"] for v in viol.values())}
results["metric"] = "sorted_keys non-decreasing per row and exact equality vs torch.sort().values; invalidates asymmetric (partner<W) gating claim if rows are sorted"
print(json.dumps(results))