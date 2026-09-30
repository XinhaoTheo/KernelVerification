import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_32/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
dev = 'cuda'
results = {}
arange_cache = {W: torch.arange(W, device=dev, dtype=torch.int64) for W in [2,3,4,5,6,7,8,9,12]}
for W in [2,3,4,5,6,7,8,9,12]:
    nondec_fail = 0; mismatch = 0; perm_bad = 0
    for trial in range(5):
        torch.manual_seed(trial)
        keys = torch.randint(-1000, 1000, (512, W), device=dev, dtype=torch.int64)
        s, p = m.triton_argsort(keys.clone()); torch.cuda.synchronize()
        ref_sorted = torch.sort(keys, dim=1).values
        nondec = bool((s[:, :-1] <= s[:, 1:]).all().item()) if W > 1 else True
        if not nondec: nondec_fail += 1
        if not bool(torch.equal(s, ref_sorted)): mismatch += 1
        psorted = torch.sort(p, dim=1).values
        ar = arange_cache[W].unsqueeze(0).expand(p.shape[0], -1)
        if not bool((psorted == ar).all().item()): perm_bad += 1
    results[f"W{W}"] = {"nondecreasing_violations_of_5": nondec_fail,
        "mismatch_vs_torchsort_of_5": mismatch,
        "perm_not_permutation_of_5": perm_bad}
results["metric"] = "per-row non-decreasing check on sorted_keys and exact equality vs torch.sort().values; claim confirmed if W=3,5,6,7,9,12 show violations/mismatches while power-of-two W stay clean"
print(json.dumps(results))