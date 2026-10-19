import sys, json, torch, struct
sys.path.insert(0, "/root/cases/case_147")
import kernel

# FP16 neighbors around boundaries 0.66 and sqrt(2)+1
def fp16_neighbors(v, k=8):
    f = torch.tensor([v], dtype=torch.float16)
    # gather a grid of fp16 values around v by bit-stepping
    code = struct.unpack("<H", struct.pack("<e", float(f[0])))[0]
    codes = [max(0,min(0x7BFF, code + d)) for d in range(-k, k+1)]
    return torch.tensor([struct.unpack("<e", struct.pack("<H", c))[0] for c in codes], dtype=torch.float16)

cand = torch.cat([fp16_neighbors(0.66), fp16_neighbors(2.414213562373095)])
cand = torch.cat([cand, -cand])
cand = cand.unique()
x = cand.to("cuda")
out = kernel.run(x)
ref = torch.atan(x.to(torch.float64))
err = (out.to(torch.float64) - ref).abs()
mx, idx = err.max(dim=0)
# also check discontinuity: compare errors just below/above boundaries
print(json.dumps({
  "values": x.cpu().tolist(),
  "errors": err.cpu().tolist(),
  "max_abs_err": float(mx),
  "worst_input": float(x[idx]),
  "tolerance": 1e-6,
  "n_violations": int((err > 1e-6).sum()),
}))