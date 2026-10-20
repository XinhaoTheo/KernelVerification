
import torch, json, sys
sys.path.insert(0, "/root/cases/case_97")
import kernel as K

# enumerate every finite FP16 value in [-10,10], including signed zeros and subnormals
bits = torch.arange(0, 65536, dtype=torch.int32)
fp16 = bits.to(torch.uint16).view(torch.float16)
vals = fp16[torch.isfinite(fp16) & (fp16.abs() <= 10.0)]
# include -0.0 explicitly? -0.0 is bit 0x8000 which is finite; included
print("num legal values:", vals.numel(), "min:", vals.min().item(), "max:", vals.max().item())
x = vals.to("cuda")
out = K.run(x)
ref = torch.expm1(x.to(torch.float64)).to("cuda")  # target in real arithmetic
err = (out.to(torch.float64) - ref).abs()
tol = 1e-7 + 1e-6 * ref.abs()
bad = err > tol
print("violations:", int(bad.sum()))
if bad.any():
    idxs = bad.nonzero().flatten()
    print("worst cases:")
    order = (err - tol).argsort(descending=True)[:10]
    for i in order.tolist():
        print(f"x={x[i].item():.6g} out={out[i].item():.9g} ref={ref[i].item():.9g} err={err[i].item():.3e} tol={tol[i].item():.3e}")
data = {
  "n_legal_values": int(vals.numel()),
  "n_violations": int(bad.sum()),
  "max_abs_err": float(err.max()),
  "max_violation_margin": float((err-tol).max()),
  "worst_x": float(x[int((err-tol).argmax())]),
}
print(json.dumps(data))
