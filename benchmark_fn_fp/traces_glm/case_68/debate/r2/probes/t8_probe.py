import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("kernel", "/root/cases/case_141/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

xs = [0, 6, 3435973834, 3435973840, 3435973841, 4000000005, 4294967288, 4294967289, 4294967292, 4294967293, 4294967295]
x = torch.tensor(xs, dtype=torch.uint32, device="cuda")
out = k.run(x); ref = k.reference(x.cpu()).cuda()
mism = (out != ref).any(dim=1).cpu()
targeted = [{"x": v, "x_mod7": v % 7, "row": int(out[i,0]), "col": int(out[i,1]),
             "ref_row": int(ref[i,0]), "ref_col": int(ref[i,1]), "bad": bool(mism[i])}
            for i, v in enumerate(xs)]

# full tail sweep in legal chunks: all x==6 mod 7 in [3435973841, 2^32-1]
base = 3435973841
assert base % 7 == 6
total = ((1<<32) - 1 - base) // 7 + 1
CH = 131072
bad_total = 0; inv_total = 0; first_bad = None; last_bad = None; n_ok = 0
i = 0
bad_rows = []
while i < total:
    m = min(CH, total - i)
    vals = (torch.arange(i, i+m, dtype=torch.int64) * 7 + base).to(torch.uint32).cuda()
    o = k.run(vals); r = k.reference(vals.cpu()).cuda()
    bad = (o != r).any(dim=1)
    inv = (o[:,0].to(torch.int64) * 7 + o[:,1].to(torch.int64)) != vals.to(torch.int64)
    b = int(bad.sum()); bad_total += b; inv_total += int(inv.sum())
    if b:
        idxs = bad.nonzero().flatten()
        if first_bad is None: first_bad = int(vals[idxs[0]])
        last_bad = int(vals[idxs[-1]])
        if len(bad_rows) < 3:
            for j in idxs[:3].tolist():
                bad_rows.append({"x": int(vals[j]), "row": int(o[j,0]), "col": int(o[j,1]),
                                 "ref_row": int(r[j,0]), "ref_col": int(r[j,1])})
    n_ok += m
    i += m
print(json.dumps({"targeted": targeted, "sweep_total_tested": n_ok,
                  "sweep_mismatch_count": bad_total,
                  "sweep_invariant_violation_count": inv_total,
                  "first_bad_x": first_bad, "last_bad_x": last_bad,
                  "sample_bad_rows": bad_rows,
                  "multiplier": k._MULTIPLIER, "shift": k._SHIFT}))