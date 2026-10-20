import torch, sys, json
sys.path.insert(0, "/root/cases/case_100")
from kernel import run

dev = "cuda" if torch.cuda.is_available() else "cpu"
bits = torch.arange(0, 65536, dtype=torch.int32)
f16 = bits.to(torch.uint16).view(torch.float16)
fin = f16[torch.isfinite(f16)]
mags = fin.abs().unique()
vals = mags[mags > 8.0].contiguous().to(dev)

ref = torch.asinh(vals.to(torch.float64))
actual = run(vals).to(torch.float64)
err = (actual - ref).abs()
tol = 1e-6
worst_i = int(err.argmax())
r = {"claim":"c2","device":dev,"n":int(vals.numel()),"max_err":float(err.max()),
     "violations":int((err>tol).sum()),"worst_x":float(vals[worst_i]),
     "worst_asinh":float(ref[worst_i]),"tol":tol,
     "err_65504":float(err[vals == 65504.0].item()) if (vals==65504.0).any() else None}
print(json.dumps(r))
