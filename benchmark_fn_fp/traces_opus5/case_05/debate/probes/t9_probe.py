import json, importlib.util, torch
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_05/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
import triton
res = {"triton_version": triton.__version__}
def ref(s, pivot, eps=0.0):
    a = s[s > pivot]
    if a.numel() == 0: return 0
    mn = a.min()
    return int((a == mn).sum().item()) if eps == 0.0 else int(((a - mn).abs() < eps).sum().item())
cases = {}
# vary number of exactly-tied boundary values k = 1..5 in N=8 tile
for k in range(1, 6):
    vals = [1.0]*k + [5.0]*(5-k+1)
    s = torch.tensor((vals + [0.0]*8)[:8], dtype=torch.float32, device="cuda")
    got = m.count_tied_at_boundary(s, 0.6)
    cases[f"k={k}"] = {"got": got, "expected": ref(s, 0.6), "tile": s.tolist()}
res["cases"] = cases
res["parity_pattern"] = all(v["got"] == v["expected"] % 2 for v in cases.values())
res["all_match"] = all(v["got"] == v["expected"] for v in cases.values())
print(json.dumps(res))
