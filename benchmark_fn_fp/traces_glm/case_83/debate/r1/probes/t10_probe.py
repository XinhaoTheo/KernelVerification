import sys, json, torch
sys.path.insert(0, "/root/cases/case_156")
import kernel as K
dev = "cuda"
# Saturating extremes and R just outside [-255,255]: large |a| with m near 2^31 and small r
extremes = [-(1<<31), (1<<31)-1, -(1<<31)+1, (1<<31)-2, 1<<31-1,
            # targets for R in 256..383 and -383..-256: R = a*m/2^(31+r) roughly
            3*(1<<30), 2*(1<<31)-1, -3*(1<<30), -(2*(1<<31)-1)]
a_list = extremes
# for each a, use r=0 (max |R|) and m=2^31-1
C = len(a_list)*2
acc = torch.tensor(a_list + a_list, dtype=torch.int32, device=dev).reshape(2, len(a_list))
mul = torch.full((C,), (1<<31)-1, dtype=torch.int32, device=dev)
rsh = torch.tensor([0]*len(a_list) + [1]*len(a_list), dtype=torch.int32, device=dev)
zp = torch.full((C,), -128, dtype=torch.int32, device=dev)  # z=-128 pulls R>255 region
zp2 = torch.full((C,), 127, dtype=torch.int32, device=dev)
out1 = K.run(acc, mul, rsh, zp)
ref1 = K.reference(acc, mul, rsh, zp)
out2 = K.run(acc, mul, rsh, zp2)
ref2 = K.reference(acc, mul, rsh, zp2)
m1 = (out1 != ref1).sum().item(); m2 = (out2 != ref2).sum().item()
# also scan R boundary band: choose a,m,r so R lands 250..260 and -250..-260
detail = []
for a in a_list:
    for r in [0,1,2]:
        pass
print(json.dumps({"metric":"exact_int8_equality_saturation_extremes","a_list":a_list,"z=-128_mismatches":m1,"z=+127_mismatches":m2,
 "R_samples_r0":[((a*((1<<31)-1))+(1<<30))>>31 for a in a_list],
 "out_z_neg128":out1.flatten().tolist(),"ref_z_neg128":ref1.flatten().tolist(),
 "out_z_pos127":out2.flatten().tolist(),"ref_z_pos127":ref2.flatten().tolist()}))
