import sys, json, torch
sys.path.insert(0, "/root/cases/case_147")
import kernel as k

bits = torch.arange(65536, dtype=torch.int32)
x = bits.to(torch.uint16).view(torch.float16).to(torch.float32)  # all bit patterns as fp16
finite = torch.isfinite(x)
x = x[finite]
# validate_inputs needs fp16 contiguous cuda
x16 = x.to(torch.float16).contiguous().cuda()
ref = torch.atan(x16.cpu().to(torch.float64)).cuda()
out = k.run(x16)
err = (out.to(torch.float64) - ref).abs()
tol = 1e-6
viol = (err > tol)
print(json.dumps({
    "metric": "max abs error vs FP64 atan over all finite FP16 values",
    "n_values": int(x16.numel()),
    "max_abs_err": float(err.max()),
    "max_err_value": float(x16.cpu()[int(err.argmax())]),
    "n_violations_1e-6": int(viol.sum()),
    "max_exceedance": float((err - tol).max()),
}))
if viol.any():
    idx = torch.nonzero(viol).flatten()[:10]
    print(json.dumps({"examples": [[float(x16.cpu()[i]), float(out.cpu()[i]), float(ref.cpu()[i]), float(err[i])] for i in idx]}))
