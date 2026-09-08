import json, importlib.util, torch
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_05/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
res = {}
# N=16 with five exact copies of 2.0 above pivot=1.0
s = torch.tensor([2.0]*5 + [7.0]*3 + [0.5]*8, dtype=torch.float32, device="cuda")
got5 = m.count_tied_at_boundary(s, 1.0)
# N=32 with 12 copies (even, >1) to distinguish bool-saturate (1) from parity (0)
s2 = torch.cat([torch.full((12,), 3.0), torch.full((4,), 9.0), torch.zeros(16)]).cuda()
got12 = m.count_tied_at_boundary(s2, 1.0)
# empty above-pivot case
s3 = torch.zeros(8, dtype=torch.float32, device="cuda")
got0 = m.count_tied_at_boundary(s3, 1.0)
res = {"got_k5": got5, "expected_k5": 5, "got_k12": got12, "expected_k12": 12,
       "got_empty": got0, "expected_empty": 0,
       "bool_saturate_signature": (got5 == 1 and got12 == 1),
       "parity_signature": (got5 == 1 and got12 == 0),
       "all_match": (got5 == 5 and got12 == 12 and got0 == 0)}
print(json.dumps(res))
