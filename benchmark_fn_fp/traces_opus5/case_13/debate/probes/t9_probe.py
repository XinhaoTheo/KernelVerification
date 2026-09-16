
import json, sys, torch, traceback
sys.path.insert(0, "/root/cases/case_13")
from kernel import gqa_gather

res = {}
for dim in [64, 80, 96, 128, 192, 256]:
    kv = torch.randn(4, dim, device="cuda", dtype=torch.float32)
    try:
        out = gqa_gather(kv, 8)
        torch.cuda.synchronize()
        res[str(dim)] = {"raised": False, "shape": list(out.shape),
                         "power_of_two": dim & (dim-1) == 0}
    except Exception as e:
        res[str(dim)] = {"raised": True, "err_type": type(e).__name__,
                         "err_msg": str(e)[:300],
                         "power_of_two": dim & (dim-1) == 0}
print(json.dumps(res))
