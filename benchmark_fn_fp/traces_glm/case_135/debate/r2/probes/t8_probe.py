import sys, json
sys.path.insert(0, "/root/cases/case_135")
import torch
import kernel as k

# All finite FP16 values: every bit pattern except inf/nan
bits = torch.arange(65536, dtype=torch.int32)
v = bits.to(torch.uint16).view(torch.float16)
v = v[torch.isfinite(v)]  # 65504 finite values incl. +/-0, subnormals, +/-65504
v = v.to("cuda")
N = v.numel()
out = k.run(v)  # kernel path, validates inputs
tgt = k.reference(v.cpu()).to("cuda")  # FP64 reference
err = (out.to(torch.float64) - tgt).abs()
maxerr = err.max().item()
idx = int(err.argmax().item())
vals_sorted, order = err.sort(descending=True)
top = [(float(v[i].item()), float(err[i].item()), float(out[i].item()), float(tgt[i].item())) for i in order[:10].tolist()]
nviol = int((err > 1e-6).sum().item())
# where is the max: large |x| or intermediate?
print(json.dumps({
  "n_values": N,
  "max_abs_err": maxerr,
  "tolerance": 1e-6,
  "n_violations": nviol,
  "worst_x": top[0][0],
  "worst_out": top[0][2],
  "worst_target": top[0][3],
  "top10_x_err_out_tgt": top,
  "extrema_err": [float(err[v.abs().argmax().item() if False else 0].item())] if False else None
}))
# extrema specifically
i65504 = int((v == 65504).nonzero()[0].item())
i_neg = int((v == -65504).nonzero()[0].item())
print(json.dumps({"err_at_65504": float(err[i65504].item()), "err_at_neg65504": float(err[i_neg].item())}))