import importlib.util, torch, json
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_125/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev = "cuda"
# Construct exact stage-2 half ties for several m: with m=2^30, H = (a + 2^30*m... ) = floor((a*m+2^30)/2^31)
# For m=2^30: H = (a + 1) // 2 (floor). a = 2*t-1 -> H = t (positive odd ties at any r),
# a = -(2*t+1) -> H = -t (negative ties). Cover r=1..31 across tie and non-tie H residues.
vals = []
for t in range(1, 130):
    vals += [2*t-1, 2*t, -(2*t-1), -(2*t), 2*t+1]   # ties, adjacent controls both signs
vals = sorted(set(vals))
ms = [1<<30, (1<<30)+1, (1<<30)+(1<<15), 2**31-1]
m_list, r_list = [], []
for r in range(1, 32):
    for m in ms:
        m_list.append(m); r_list.append(r)
C = len(m_list)
acc0 = torch.tensor(vals, dtype=torch.int32)
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
stage2_tie = torch.zeros_like(mism)
for j in range(C):
    r = int(R[j])
    stage2_tie[:, j] = (H[:, j] % (1 << r)) == (1 << (r-1))
examples = []
for i, j in torch.nonzero(mism).tolist()[:15]:
    examples.append({"a":int(A[i,j]),"m":int(M[j]),"r":int(R[j]),"H":int(H[i,j]),
                     "H_mod_2r":int(H[i,j] % (1<<int(R[j]))),
                     "stage2_tie":bool(stage2_tie[i,j]),
                     "out":float(out[i,j]),"ref":float(ref[i,j])})
mism_by_r, tie_mism_by_r = {}, {}
for j in range(C):
    r = int(R[j])
    c = int(mism[:, j].sum()); ct = int((mism & stage2_tie)[:, j].sum())
    if c: mism_by_r[r] = c
    if ct: tie_mism_by_r[r] = ct
print(json.dumps({
  "n_elements": out.numel(), "n_rows": len(vals), "n_channels": C,
  "mismatch_count": int(mism.sum()),
  "mismatch_on_stage2_tie": int((mism & stage2_tie).sum()),
  "mismatch_off_tie": int((mism & ~stage2_tie).sum()),
  "mism_by_r": mism_by_r, "tie_mism_by_r": tie_mism_by_r,
  "examples": examples
}))