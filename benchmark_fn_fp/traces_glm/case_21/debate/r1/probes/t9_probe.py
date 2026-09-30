import torch, sys, json
sys.path.insert(0, "/root/cases/case_21")
from kernel import paged_gather

result = {}
for hd in [48, 96, 8]:
    try:
        kv = torch.randn(8, 4, hd, device="cuda")
        bt = torch.randint(0, 8, (2, 2), dtype=torch.int32, device="cuda")
        out = paged_gather(kv, bt, 4)
        t = torch.arange(4, device="cuda")
        ref = kv[bt[:, None].expand(2, 4), t//4, t%4]
        result[f"head_dim_{hd}"] = {"status": "ok", "mismatches": (out != ref).sum().item()}
    except Exception as e:
        result[f"head_dim_{hd}"] = {"status": "error", "type": type(e).__name__, "msg": str(e)[:300]}
print(json.dumps(result))