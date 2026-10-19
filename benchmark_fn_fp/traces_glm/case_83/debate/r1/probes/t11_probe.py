import sys, json, torch
sys.path.insert(0, "/root/cases/case_156")
import kernel as K
dev = "cuda"
g = torch.Generator().manual_seed(7)
# r=31 with large-magnitude negative and positive a, plus r=30,29
rows, C = 512, 4
a = torch.randint(-(1<<31), 1<<31, (rows, C), generator=g, dtype=torch.int64).to(torch.int32)
# force extremes in first rows
a[0,:] = torch.tensor([-(1<<31), (1<<31)-1, -(1<<31)+1, (1<<31)-2], dtype=torch.int32)
mul = torch.tensor([1<<30, (1<<31)-1, 1<<30, (1<<31)-1], dtype=torch.int32)
rsh = torch.tensor([31, 31, 30, 30], dtype=torch.int32)
zp = torch.tensor([0, 127, -128, 127], dtype=torch.int32)
out = K.run(a.to(dev), mul.to(dev), rsh.to(dev), zp.to(dev))
ref = K.reference(a.to(dev), mul.to(dev), rsh.to(dev), zp.to(dev))
mism = (out != ref).sum().item()
md = []
if mism:
    idx = (out != ref).nonzero()
    for i in idx[:10].tolist():
        md.append({"i":i,"a":a[i[0],i[1]].item(),"c":i[1],"m":mul[i[1]].item(),"r":rsh[i[1]].item(),"z":zp[i[1]].item(),"kernel":out[i[0],i[1]].item(),"ref":ref[i[0],i[1]].item()})
# expected R for extremes at r=31
def R(a,m,r):
    return (a*m + (1<<(30+r))) >> (31+r)
exp = {f"r31_a{a[0,j].item()}":R(a[0,j].item(),mul[j].item(),31) for j in range(4)}
print(json.dumps({"metric":"exact_int8_equality_r31_shift62_large_products","rows":rows,"mismatches":mism,"mismatch_detail":md,"expected_R_extremes_r31":exp}))
