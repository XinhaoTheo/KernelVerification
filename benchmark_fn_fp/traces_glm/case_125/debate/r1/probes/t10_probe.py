
import importlib.util, torch, json
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_125/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev = "cuda"
g = torch.Generator("cpu").manual_seed(1234)
rows, C = 2048, 64
acc = torch.randint(-(1<<31), 1<<31, (rows, C), dtype=torch.int64, generator=g).to(torch.int32)
acc[0,0] = -(1<<31); acc[0,1] = (1<<31)-1
# also boundary-adjacent values: small a and values making a*m near tie points
for j in range(2, 12):
    acc[1, j] = j - 6
mult = torch.randint(1<<30, 1<<31, (C,), dtype=torch.int64, generator=g).to(torch.int32)
mult[0] = 1<<30; mult[1] = 2**31 - 1
shift = torch.randint(0, 32, (C,), dtype=torch.int32, generator=g)
for i in range(32):  # guarantee every r is covered
    shift[i] = i
z = torch.randint(-128, 128, (C,), dtype=torch.int32, generator=g)
exp = torch.randint(-8, 1, (C,), dtype=torch.int32, generator=g)
s = torch.pow(torch.tensor(2.0), exp).to(torch.float32)
acc, mult, shift, z, s = (t.to(dev) for t in (acc, mult, shift, z, s))
out = k.run(acc, mult, shift, z, s)
ref = k.reference(acc, mult, shift, z, s)
diff = (out - ref).abs()
mism = (diff > 1e-6).cpu()
A = acc.cpu(); M = mult.cpu(); R = shift.cpu()
P = A.to(torch.int64) * M.to(torch.int64)
H = torch.div(P + (1<<30), 1<<31, rounding_mode="floor")
examples = []
for i, j in torch.nonzero(mism).tolist()[:25]:
    a=int(A[i,j]); m=int(M[j]); r=int(R[j]); h=int(H[i,j])
    Pval = a*m
    tie2 = (h % (1 << r) == (1 << (r-1))) if r >= 1 else False
    tie1 = ((Pval + (1<<30)) % (1<<31) == 0)
    examples.append({"a":a,"m":m,"r":r,"H":h,"H_mod_2r":h % (1<<r) if r>=1 else 0,"stage2_tie":bool(tie2),"stage1_tie":bool(tie1),"out":float(out[i,j]),"ref":float(ref[i,j])})
mism_r = {}
for j in range(C):
    cnt = int(mism[:, j].sum())
    if cnt: mism_r[int(R[j])] = cnt
print(json.dumps({
  "n_elements": out.numel(),
  "mismatch_count": int(mism.sum()),
  "mismatch_by_r": mism_r,
  "examples": examples
}))
