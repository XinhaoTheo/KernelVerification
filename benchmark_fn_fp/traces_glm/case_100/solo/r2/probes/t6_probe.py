
import torch, numpy as np, json, sys
sys.path.insert(0, "/root/cases/case_100")
import kernel

# all finite FP16 bit patterns
bits = np.arange(65536, dtype=np.uint16)
vals = torch.from_numpy(bits.view(np.float16)).float()
x16 = torch.from_numpy(bits.view(np.float16))
finite = torch.isfinite(x16)
x_finite = x16[finite].cuda()
xf = x_finite.to(torch.float64)
ref = torch.asinh(xf)

out = kernel.run(x_finite)
err = (out.to(torch.float64) - ref).abs()
max_err = err.max().item()
imax = int(err.argmax().item())
data = {
    "num_inputs": int(x_finite.numel()),
    "max_abs_err": max_err,
    "worst_input": float(x_finite[imax].item()),
    "worst_ref": float(ref[imax].item()),
    "worst_actual": float(out[imax].item()),
    "num_violations_gt_1e-6": int((err > 1e-6).sum().item()),
}
# top offenders
k = torch.topk(err, 5).indices
data["top5"] = [[float(x_finite[i].item()), float(err[i].item())] for i in k.tolist()]
print(json.dumps(data))
