import torch, sys
sys.path.insert(0, "/root/cases/case_122")
import kernel as K

results = {}
# case 1: delta at corner of a 3x3 field, steps=4
x = torch.zeros(3,3, dtype=torch.float16, device="cuda"); x[0,0]=1.0
r = K.run(x, 4); ref = K.reference(x, 4)
results["delta_3x3_steps4"] = (r-ref).abs().max().item()

# case 2: delta at edge of a larger field, steps=8
x2 = torch.zeros(31,33, dtype=torch.float16, device="cuda"); x2[0,5]=1.0
r2 = K.run(x2, 8); ref2 = K.reference(x2, 8)
results["delta_edge_steps8"] = (r2-ref2).abs().max().item()

# case 3: random in-domain input
xr, steps = K.make_inputs()
r3 = K.run(xr, steps); ref3 = K.reference(xr, steps)
results["random_31x33_steps4"] = (r3-ref3).abs().max().item()

print({k: v for k,v in results.items()})
import json
print("JSON:" + json.dumps({"metric":"max abs error vs FP64 per-step reference","tolerance":2e-5,"errors":results}))
