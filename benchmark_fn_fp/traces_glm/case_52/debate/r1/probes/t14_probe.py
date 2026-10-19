import importlib.util, torch, json
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_125/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev = "cuda"
# rows must be <= 4096; cover a dense negative/positive band hitting odd negative H (r=1 ties)
# and negative H at r=0.
vals = list(range(-2047, 2048))
acc0 = torch.tensor(vals, dtype=torch.int32)
ms = [1<<30, (1<<30)+1, (1<<30)+3, (1<<30)+(1<<15), (1<<30)+(1<<29), 2**31-1]
m_list, r_list = [], []
for r in (0, 1):
    for m in ms:
        m_list.append(m); r_list.append(r)
C = len(m_list)
acc = acc0.reshape(len(vals),1).repeat(1,C).contiguous().to(dev)
mult = torch.tensor(m_list, dtype=torch.int32).to(dev)
shift = torch.tensor(r_list, dtype=torch.int32).to(dev)
z = torch.zeros(C, dtype=torch.int32).to(dev)
s = torch.ones(C, dtype=torch.float32).to(dev)
out = k.run(acc, mult, shift, z, s)
ref = k.reference(acc, mult, shift, z, s)
mism = ((out - ref).abs() > 1e-6).cpu()
A = acc.cpu(); M = mult.cpu(); R = shift.cpu()
P = A.to(torch.int64) * M.to(torch.int64)
H = torch.div(P + (1<<30), 1<<31, rounding_mode="floor")
neg_tie_r1 = (R == 1) & (H < 0) & (torch.div(H, 2, rounding_mode="floor") * 2 + 1 == H)
pos_tie_r1 = (R == 1) & (H > 0) & (H % 2 == 1)
r1_neg_odd_mism = int((mism & neg_tie_r1).sum())
r1_neg_odd_total = int(neg_tie_r1.sum())
r1_other_mism = int((mism & (R == 1) & ~neg_tie_r1).sum())
r0_mism = int((mism & (R == 0)).sum())
r0_negH_mism = int((mism & (R == 0) & (H < 0)).sum())
examples = []
for i, j in torch.nonzero(mism).tolist()[:25]:
    a=int(A[i,j]); m=int(M[j]); r=int(R[j]); h=int(H[i,j])
    examples.append({"a":a,"m":m,"r":r,"H":h,"H_neg_odd_tie_r1": bool(r==1 and h<0 and h%2!=0),
                     "out":float(out[i,j]),"ref":float(ref[i,j])})
print(json.dumps({
  "n_elements": out.numel(), "n_rows": len(vals), "n_channels": C,
  "mismatch_count": int(mism.sum()),
  "r1_neg_odd_tie_total": r1_neg_odd_total,
  "r1_neg_odd_tie_mismatches": r1_neg_odd_mism,
  "r1_other_mismatches": r1_other_mism,
  "r0_mismatches": r0_mism,
  "r0_negH_mismatches": r0_negH_mism,
  "examples": examples
}))
