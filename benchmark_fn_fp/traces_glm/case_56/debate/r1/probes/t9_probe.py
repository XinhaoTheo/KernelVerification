
import json, math, torch
import kernel as K

dev = "cuda" if torch.cuda.is_available() else "cpu"
# Build all positive FP16 values in the mid-range band [3, 13] plus a broad full-domain sample
vals = []
v = torch.tensor([0], dtype=torch.float16)
# enumerate all positive finite FP16 values via bit patterns
bits = torch.arange(0, 65536, dtype=torch.int32)
allf16 = bits.to(torch.uint16).view(torch.float16)
pos = allf16[torch.isfinite(allf16) & (allf16 > 0)]
band = pos[(pos >= 2.0) & (pos <= 14.0)]
negband = -band
x = torch.cat([band, negband]).to(dev)
out = K.run(x)
ref = K.reference(x).to(torch.float64).to(dev)
err = (out.to(torch.float64) - ref).abs()
tol = 1e-6
bad = err > tol
i = int(torch.argmax(err))
print(json.dumps({
    "n": int(x.numel()),
    "max_abs_err": float(err.max()),
    "tol": tol,
    "n_violations": int(bad.sum()),
    "worst_x": float(x.flatten()[i]),
    "worst_err": float(err.flatten()[i]),
    "dev": dev,
}))
