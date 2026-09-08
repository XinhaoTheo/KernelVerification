import json, importlib.util, torch
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_05/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
out = {}
def exact_ref(s, pivot):
    a = s[s > pivot]
    if a.numel() == 0: return 0
    return int((a == a.min()).sum().item())
# distinct near-boundary values within 1e-3
s = torch.tensor([1.0, 1.0005, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=torch.float32, device="cuda")
out["near_tie_1e-3"] = {"got": m.count_tied_at_boundary(s, 0.5), "exact_ref": exact_ref(s, 0.5)}
# large-magnitude logits: many distinct values inside 1e-3 window
base = 1000.0
vals = [base + i*1e-4 for i in range(6)] + [2000.0, 3000.0]
s2 = torch.tensor(vals, dtype=torch.float32, device="cuda")
out["large_mag"] = {"got": m.count_tied_at_boundary(s2, 500.0), "exact_ref": exact_ref(s2, 500.0),
                    "distinct_repr": len(set(s2.tolist()))}
# fp32 spacing check near 1000: is 1e-4 representable
out["ulp_at_1000"] = float(torch.tensor(1000.0).nextafter(torch.tensor(1001.0)).item() - 1000.0)
# separated-by-more-than-eps control: should not merge
s3 = torch.tensor([1.0, 1.01, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=torch.float32, device="cuda")
out["separated_control"] = {"got": m.count_tied_at_boundary(s3, 0.5), "exact_ref": exact_ref(s3, 0.5)}
# exact-tie sanity at large magnitude
s4 = torch.tensor([1000.0]*3 + [2000.0]*5, dtype=torch.float32, device="cuda")
out["exact_tie_large"] = {"got": m.count_tied_at_boundary(s4, 500.0), "exact_ref": exact_ref(s4, 500.0)}
print(json.dumps(out))
