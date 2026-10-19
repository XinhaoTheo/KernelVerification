import torch, json, sys
sys.path.insert(0, "/root/cases/case_146")
import kernel

# all finite FP16 values (nonzero magnitudes plus both zeros)
vals = torch.arange(0, 0x8000, dtype=torch.int32)
bits_hi = 0x8000 | vals
x_neg = bits_hi.to(torch.uint16).view(torch.float16)
x_pos = vals.to(torch.uint16).view(torch.float16)
# remove inf/nan
def finite_mask(t):
    return torch.isfinite(t)
x_pos = x_pos[finite_mask(x_pos)]
x_neg = x_neg[finite_mask(x_neg)]
x_all = torch.cat([x_neg.flip(0), x_pos]).to(torch.float16).to("cuda")

out = kernel.run(x_all)
ref = torch.atan(x_all.to(torch.float64)).to(torch.float32)
err = (out.double() - ref.double()).abs()
i = int(err.argmax())
res = {
  "n_tested": int(x_all.numel()),
  "max_abs_err": float(err[i]),
  "argmax_x": float(x_all[i]),
  "n_violations": int((err > 1e-6).sum()),
}
# a few worst offenders
top = torch.topk(err, 10)
res["top10"] = [(float(x_all[int(j)]), float(e)) for j, e in zip(top.indices, top.values)]
print(json.dumps(res))