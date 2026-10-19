import torch, json, sys
sys.path.insert(0, "/root/cases/case_136")
import kernel

bits = torch.arange(65536, dtype=torch.int32)
x = torch.tensor(bits, dtype=torch.float16).view(torch.int16).view(torch.int16).to(torch.int16)
x16 = bits.to(torch.int16).view(torch.float16)
x16 = x16.to("cuda")
finite = torch.isfinite(x16)
assert finite.all(), "all bit patterns should be finite after masking"
# skip NaN/Inf encodings
mask = torch.isfinite(x16)
idx = torch.nonzero(mask).flatten()
xv = x16[idx].contiguous()
out = kernel.run(xv)
ref = kernel.reference(xv.cpu())
err = (out.cpu().to(torch.float64) - ref).abs()
max_err = err.max().item()
worst_i = int(err.argmax())
data = {
    "n_tested": int(xv.numel()),
    "max_abs_err": max_err,
    "tolerance": 1e-6,
    "violations": int((err > 1e-6).sum()),
    "worst_x": float(xv.cpu()[worst_i]),
    "worst_out": float(out.cpu()[worst_i]),
    "worst_ref": float(ref[worst_i]),
}
print(json.dumps(data))
