
import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_09/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev="cuda"
res={}
for N in [2,4,8,16,32,64,128]:
    torch.manual_seed(1)
    scores = torch.randn(512, N, device=dev)
    out = k.sorted_topk_indices(scores, 1).squeeze(-1)
    ref = scores.argmax(dim=1)
    res[f"N{N}"] = {"mismatch": int((out!=ref).sum().item()), "total": 512}
# also check k>1 ordering matches descending sort on tie-free input
torch.manual_seed(2)
s = torch.randn(64, 16, device=dev)
out = k.sorted_topk_indices(s, 5)
ref = torch.argsort(s, dim=1, descending=True, stable=True)[:, :5]
res["k5_exact_match"] = bool(torch.equal(out, ref))
print(json.dumps(res))
