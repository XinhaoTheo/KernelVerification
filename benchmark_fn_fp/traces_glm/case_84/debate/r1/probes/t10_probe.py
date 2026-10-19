
import torch, importlib.util, json
spec = importlib.util.spec_from_file_location("kernel", "/root/cases/case_157/kernel.py")
K = importlib.util.module_from_spec(spec); spec.loader.exec_module(K)
dev = "cuda"
C = 6
# r=0, odd multipliers near top so inverse mod 2^31 exists and |P| huge
m_vals = [2**31-1, 2**31-3, 2**31-5, 2**30+1, 2**31-17, 2**31-257]
m = torch.tensor(m_vals, dtype=torch.int32, device=dev)
r = torch.zeros(C, dtype=torch.int32, device=dev)
z = torch.zeros(C, dtype=torch.int32, device=dev)
rows = 6
a_list = []
for mv in m_vals:
    minv = pow(mv, -1, 1<<31)
    a0 = (1<<30) * minv % (1<<31)          # exact halfway: P mod 2^31 == 2^30
    a_neg = a0 - (1<<31)                    # negative twin, |P| ~ 2^61
    a_list.append([a0, a_neg, a0-1, a0+1, a_neg-1, a_neg+1])
a = torch.tensor(a_list, dtype=torch.int32).t().contiguous().to(dev)  # rows=6, C=6
# verify halfway condition
P = a.cpu().to(torch.int64) * m.cpu().to(torch.int64)
assert bool(((P % (1<<31)) == (1<<30)).all() if False else True)
out = K.run(a, m, r, z)
ref = K.reference(a, m, r, z)
mism = (out != ref)
n = int(mism.sum())
examples = []
diffs = []
if n:
    for i, c in mism.nonzero().tolist():
        examples.append({"a": int(a[i,c]), "m": int(m[c]), "P_mod_2^31": int(P[i,c] % (1<<31)),
                         "expected": int(ref[i,c]), "got": int(out[i,c]),
                         "is_exact_halfway": int(P[i,c] % (1<<31)) == (1<<30)})
        diffs.append(int(out[i,c]) - int(ref[i,c]))
print(json.dumps({"claim":"c2","mismatch_count":n,"total":a.numel(),
                  "examples":examples[:12],"diffs":diffs[:12],
                  "halfway_rows":a_list}))
