
import importlib.util, torch, json
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_125/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev = "cuda"
vals = list(range(-1024, 1025))
acc0 = torch.tensor(vals, dtype=torch.int32)
ms = [1<<30, (1<<30)+1, (1<<30)+7, (1<<30)+(1<<20), (1<<30)+(1<<29), 2**31-1]
m_list, r_list = [], []
for r in range(32):
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
diff = (out - ref).abs()
mism = (diff > 1e-6).cpu()
A = acc.cpu(); M = mult.cpu(); R = shift.cpu()
P = A.to(torch.int64) * M.to(torch.int64)
H = torch.div(P + (1<<30), 1<<31, rounding_mode="floor")
sign_diff = (((A < 0) & (H >= 0)) | ((A >= 0) & (H < 0)))
examples = []
for i, j in torch.nonzero(mism).tolist()[:20]:
    a=int(A[i,j]); m=int(M[j]); r=int(R[j]); h=int(H[i,j])
    examples.append({"a":a,"m":m,"r":r,"H":h,"out":float(out[i,j]),"ref":float(ref[i,j]),"sign_a_neg":a<0,"sign_H_neg":h<0})
# specifically the cited case a=-1, m=2^30
j_case = m_list.index(1<<30)  # r==0 slot for m=2^30
row_case = vals.index(-1)
case_rows = {}
for r in range(32):
    j = r_list.index(r) and None
for r in range(32):
    jx = next(idx for idx,(mm,rr) in enumerate(zip(m_list,r_list)) if mm==(1<<30) and rr==r)
    case_rows[r] = {"kernel": float(out[row_case, jx]), "ref": float(ref[row_case, jx])}
print(json.dumps({
  "n_elements": out.numel(),
  "mismatch_count": int(mism.sum()),
  "sign_diff_count": int(sign_diff.sum()),
  "mismatch_and_sign_diff": int((sign_diff & mism).sum()),
  "mismatch_when_signs_match": int((~sign_diff & mism).sum()),
  "examples": examples,
  "a_minus1_m2p30_by_r": {str(r): case_rows[r] for r in [0,1,2,3,4,5,10,20,31]}
}))
