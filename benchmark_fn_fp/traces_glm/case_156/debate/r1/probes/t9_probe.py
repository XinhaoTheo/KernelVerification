import sys, json, torch
sys.path.insert(0, "/root/cases/case_156")
import kernel as K

dev = "cuda"
# Negative exact-halfway products: a*m = -(2k+1) * 2^(30+r) for small r
cases = []
for r in [0,1,2,3,5,10,15,20,31]:
    # want a*m = -3 * 2^(30+r); m in [2^30, 2^31-1] so pick m=2^30*3? too big. m must be <=2^31-1.
    # choose m = 3*2^k where k <= 30 such that m <= 2^31-1, and a = -2^(30+r-k)
    for k in range(0, 31):
        m = 3 * (1 << k)
        if m > (1<<31)-1: continue
        if 30 + r - k < 0 or 30 + r - k > 31: continue
        a = -(1 << (30 + r - k))
        if not (-(1<<31) <= a < (1<<31)): continue
        cases.append((a, m, r))
cases = list(dict.fromkeys(cases))
# dedupe and also positive halfway
rows = len(cases) + 4
C = 1
acc = torch.zeros((rows, C), dtype=torch.int32, device=dev)
mul = torch.zeros(C, dtype=torch.int32, device=dev)
rsh = torch.zeros(C, dtype=torch.int32, device=dev)
zp = torch.zeros(C, dtype=torch.int32, device=dev)
# put cases across rows, channel 0
a_list = [c[0] for c in cases] + [3, 3, 1, -1]
acc[:, 0] = torch.tensor(a_list, dtype=torch.int32, device=dev)
mul[0] = (1 << 30)  # placeholder, vary per row not possible; instead run kernel per case via C=rows trick
# Better: use per-row channel trick: make C = rows, each channel own m/r
C = len(a_list)
acc2 = torch.tensor(a_list, dtype=torch.int32, device=dev).reshape(1, C)
mul2 = torch.tensor([c[1] for c in cases] + [1<<31-1, (1<<31)-1, (1<<31)-1, (1<<31)-1], dtype=torch.int32, device=dev).reshape(C)
# last four: use m=(2^31-1) doesn't give exact halves; skip those, keep m as 1<<30
mul2 = torch.tensor([c[1] for c in cases] + [1<<30]*4, dtype=torch.int32, device=dev)
rsh2 = torch.tensor([c[2] for c in cases] + [0,0,0,0], dtype=torch.int32, device=dev)
zp2 = torch.zeros(C, dtype=torch.int32, device=dev)
out = K.run(acc2, mul2, rsh2, zp2)
ref = K.reference(acc2, mul2, rsh2, zp2)
mism = (out != ref).sum().item()
# expected R per case: quotient = a*m/2^(31+r) = -(2k+1)/2 * ... check -1.5 -> -1
exp = []
for (a,m,r) in cases:
    R = (a*m + (1 << (30+r))) >> (31+r)
    exp.append(R)
detail = []
for i,(a,m,r) in enumerate(cases):
    q = a*m / (1 << (31+r))
    if out[0,i].item() != ref[0,i].item():
        detail.append({"a":a,"m":m,"r":r,"quotient":q,"kernel":out[0,i].item(),"ref":ref[0,i].item(),"math_floor":exp[i]})
print(json.dumps({"metric":"exact_int8_equality_on_negative_halfway_cases","n_cases":len(cases),"mismatches":mism,"mismatch_detail":detail,"sample_quotients":[a*m/(1<<(31+r)) for (a,m,r) in cases[:5]]}))
