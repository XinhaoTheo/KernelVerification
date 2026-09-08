
import json, importlib.util, torch
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_32/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
torch.manual_seed(1)
dev = "cuda"
res = {"per_W": []}

# targeted hand-trace case
a = torch.tensor([[3,1,2]], dtype=torch.int64, device=dev)
sk, p = m.triton_argsort(a)
res["W3_312_sorted_keys"] = sk[0].tolist()
res["W3_312_perm"] = p[0].tolist()

for W in [3,5,6,7,9,12,100]:
    N = 32
    # distinct keys per row via random permutation of 0..W-1
    k = torch.stack([torch.randperm(W, device=dev) for _ in range(N)]).to(torch.int64)
    sk, p = m.triton_argsort(k)
    nonmono = int((sk.diff(dim=1) < 0).any(dim=1).sum().item())
    valmismatch = int((sk != torch.sort(k, dim=1).values).any(dim=1).sum().item())
    res["per_W"].append({"W": W, "rows": N, "nonmonotonic_rows": nonmono, "value_mismatch_rows": valmismatch})

# control: power-of-two widths with distinct keys should be fine
ctrl = []
for W in [4,8,16,64]:
    N = 32
    k = torch.stack([torch.randperm(W, device=dev) for _ in range(N)]).to(torch.int64)
    sk, p = m.triton_argsort(k)
    ctrl.append({"W": W, "nonmonotonic_rows": int((sk.diff(dim=1) < 0).any(dim=1).sum().item()),
                 "value_mismatch_rows": int((sk != torch.sort(k, dim=1).values).any(dim=1).sum().item())})
res["pow2_control"] = ctrl
print(json.dumps(res))
