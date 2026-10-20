import torch, json, sys
sys.path.insert(0, "/root/cases/case_166")
import kernel as K

# all legal FP16 values > -1
vals = []
for bits in range(0x0000, 0x10000):
    v = torch.tensor(bits, dtype=torch.int32).to(torch.uint16).view(torch.float16).item()
    import struct
    v = struct.unpack('<e', struct.pack('<H', bits))[0]
    if v == v and abs(v) != float('inf') and v > -1:
        vals.append(bits)
x = torch.tensor([struct.unpack('<e', struct.pack('<H', b))[0] for b in vals], dtype=torch.float16)
x = x.to("cuda")
out = K.run(x)
ref = torch.log1p(x.to(torch.float64)).to("cuda")
err = (out.to(torch.float64) - ref).abs()
maxerr = err.max().item()
i = err.argmax().item()
# also focus region x in (0.6,0.66) and (2.2,2.35)
mask1 = (x > 0.6) & (x < 0.66)
mask2 = (x > 2.2) & (x < 2.35)
res = {
 "metric": "max abs error vs FP64 log1p over all 15359 legal FP16 values",
 "n": x.numel(),
 "max_abs_err": maxerr,
 "worst_x": x[i].item(),
 "tolerance": 1e-6,
 "n_violations": int((err > 1e-6).sum().item()),
 "max_err_region_0.6_0.66": err[mask1].max().item() if mask1.any() else None,
 "max_err_region_2.2_2.35": err[mask2].max().item() if mask2.any() else None,
 "top5_err": sorted(err.tolist())[-5:],
}
print(json.dumps(res))
