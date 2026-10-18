import importlib.util, sys, numpy as np, torch, struct

spec = importlib.util.spec_from_file_location("k", "/root/cases/case_149/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

allbits = np.arange(0, 65536, dtype=np.uint16)
xf16 = torch.from_numpy(allbits.view(np.float16)).to("cuda")

# exclude NaN/Inf: NaN exponent all-ones and nonzero mantissa; Inf exponent all-ones mantissa 0
f32 = xf16.cpu().numpy().astype(np.float32)
f64 = xf16.cpu().numpy().astype(np.float64)
finite = np.isfinite(f64)
print("finite count:", finite.sum())

xf = xf16.cpu()[torch.from_numpy(finite).to(torch.bool)].to("cuda")
print("N:", xf.numel())

out = k.run(xf)
# reference: FP64 sigmoid then round-to-nearest-even to FP16 via struct pack "<e"
ref = bytearray()
vals = torch.sigmoid(xf.cpu().to(torch.float64)).tolist()
for v in vals:
    ref += struct.pack("<e", v)
ref = torch.frombuffer(bytes(ref), dtype=torch.float16).clone().to("cuda")

same = torch.equal(out, ref)
mism = (out.view(torch.int16) != ref.view(torch.int16)).nonzero().flatten()
print("bit_exact_all:", same, "num_mismatch:", mism.numel())
if mism.numel() > 0:
    idx = mism[:20].cpu().tolist()
    for i in idx:
        x_in = xf[i].item(); print("case", i, "x=", x_in, "hex=", hex(xf[i].view(torch.int16).item() & 0xFFFF),
              "kernel=", out[i].item(), "ref=", ref[i].item())
import json
print(json.dumps({"metric":"bit-exact vs FP64-rounded FP16 over all finite FP16 inputs",
 "num_finite_inputs": int(finite.sum()), "num_mismatch": int(mism.numel()),
 "bit_exact": bool(same)}))
