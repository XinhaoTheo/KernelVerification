import numpy as np, torch, json, sys
sys.path.insert(0, "/root/cases/case_147")
import kernel as K

bits = np.arange(65536, dtype=np.uint16)
vals = bits.view(np.float16).astype(np.float64)
finite = np.isfinite(vals)
mag = np.abs(vals)
sel = finite & (mag <= 0.66)
xs = bits[sel].view(np.float16).copy()
x = torch.from_numpy(xs).to("cuda")
out = K.run(x)
ref = torch.atan(x.to(torch.float64))
err = (out.to(torch.float64) - ref).abs()
mx = err.max().item(); i = int(err.argmax())
negzero = (x == -0.0)
nz_out = out[negzero.to(out.device)]
signed_zero_ok = bool((nz_out == 0.0).all().item())
print(json.dumps({"claim": "c3", "count": int(sel.sum()), "max_abs_err": mx,
                  "worst_x": float(x[i].item()), "worst_err": float(err[i].item()),
                  "tolerance": 1e-6, "violations": int((err > 1e-6).sum()),
                  "neg_zero_count": int(negzero.sum()), "neg_zero_outputs_all_zero": signed_zero_ok}))