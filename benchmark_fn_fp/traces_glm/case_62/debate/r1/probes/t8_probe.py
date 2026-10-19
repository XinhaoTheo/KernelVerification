import torch, numpy as np, json, sys
sys.path.insert(0, "/root/cases/case_135")
import kernel

# all finite FP16 values
bits = np.arange(65536, dtype=np.uint16)
vals = bits.view(np.float16)
finite = np.isfinite(vals)
x = torch.from_numpy(vals[finite].copy()).to("cuda")
print("num_finite", x.numel(), file=sys.stderr)

out = kernel.run(x)
ref = x.to(torch.float64) / torch.sqrt(1.0 + x.to(torch.float64) * x.to(torch.float64))
err = (out.to(torch.float64) - ref).abs()
maxi = int(err.argmax())
print(json.dumps({
    "num_inputs": int(x.numel()),
    "max_abs_err": float(err.max()),
    "tolerance": 1e-6,
    "argmax_x": float(x[maxi].item()),
    "worst_x_abs": float(abs(x[maxi].item())),
    "num_violations": int((err > 1e-6).sum()),
    "out_at_worst": float(out[maxi].item()),
    "ref_at_worst": float(ref[maxi].item())
}))
