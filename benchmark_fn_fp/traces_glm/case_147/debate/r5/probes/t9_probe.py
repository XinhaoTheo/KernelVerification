import torch, json, sys
sys.path.insert(0, "/root/cases/case_147")
import kernel as K

# All representable FP16 values in middle interval (0.66, 1+sqrt(2)]
all_u16 = torch.arange(65536, dtype=torch.int32).to(torch.uint16)
vals = all_u16.view(torch.float16).to(torch.float64)
finite = torch.isfinite(vals)
mag = vals.abs()
sel = finite & (mag > 0.66) & (mag <= 2.414213562373095)
xs = all_u16[sel].numpy()
import numpy as np
x = torch.tensor(xs, dtype=torch.float16, device="cuda")
out = K.run(x).to(torch.float64)
ref = torch.atan(x.to(torch.float64).cpu()).to(x.device) if False else torch.atan(x.to(torch.float64))
err = (out.to(torch.float64) - ref).abs()
mx = err.max().item()
i = int(err.argmax())
print(json.dumps({"claim": "c1", "count": int(sel.sum()), "max_abs_err": mx,
                 "worst_x": float(x[i].item()), "worst_err": float(err[i].item()),
                 "tolerance": 1e-6, "violations": int((err > 1e-6).sum())}))