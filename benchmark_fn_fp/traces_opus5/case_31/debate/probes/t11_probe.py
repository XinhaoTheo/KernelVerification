
import json, importlib.util, torch
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_32/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
torch.manual_seed(2)
dev = "cuda"
res = {"cases": []}
for W in [8, 32, 64, 128, 1024, 3, 5, 12, 100]:
    N = 64
    k = torch.randint(0, max(4, W//2), (N, W), dtype=torch.int64, device=dev)
    korig = k.clone()
    sk, p = m.triton_argsort(k)
    gather_fail = int((sk != torch.gather(korig, 1, p)).any(dim=1).sum().item())
    # perm validity
    srt = torch.sort(p, dim=1).values
    ref = torch.arange(W, device=dev, dtype=torch.int64).expand(N, W)
    notperm = int((srt != ref).any(dim=1).sum().item())
    input_mutated = bool((k != korig).any().item())
    # determinism across 3 runs
    runs = []
    for _ in range(3):
        s2, p2 = m.triton_argsort(korig.clone())
        runs.append((s2.clone(), p2.clone()))
    det = all(bool((runs[0][0]==r[0]).all().item() and (runs[0][1]==r[1]).all().item()) for r in runs)
    res["cases"].append({"W": W, "rows": N, "gather_identity_fail_rows": gather_fail,
                         "perm_not_permutation_rows": notperm,
                         "input_tensor_mutated": input_mutated,
                         "deterministic_3runs": det})
print(json.dumps(res))
