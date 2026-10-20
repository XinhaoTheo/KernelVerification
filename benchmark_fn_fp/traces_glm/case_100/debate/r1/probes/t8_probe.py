import torch, sys
sys.path.insert(0, "/root/cases/case_100")
from kernel import run

bits = torch.arange(0, 65536, dtype=torch.int32)
f16 = bits.to(torch.uint16).view(torch.float16)
fin = f16[torch.isfinite(f16)]
mags = fin.abs()
sel = mags[mags <= 1.05]  # polynomial branch domain
vals = sel.unique().contiguous()

ref = torch.asinh(vals.to(torch.float64))
actual = run(vals).to(torch.float64)
err = (actual - ref).abs()
tol = 1e-6
worst_i = int(err.argmax())
print({
 "n_values": int(vals.numel()),
 "max_err": float(err.max()),
 "tolerance": tol,
 "violations": int((err > tol).sum()),
 "worst_x": float(vals[worst_i]),
 "worst_asinh": float(ref[worst_i]),
})
import json
print(json.dumps({"claim":"c1","max_err":float(err.max()),"violations":int((err>tol).sum()),"worst_x":float(vals[worst_i]),"tol":tol,"n":int(vals.numel())}))
