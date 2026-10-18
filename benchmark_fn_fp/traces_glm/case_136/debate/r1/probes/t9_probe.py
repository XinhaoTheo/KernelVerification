import sys, torch, numpy as np, json
sys.path.insert(0, "/root/cases/case_136")
import kernel

# all finite FP16 values (65536 minus NaN/Inf), N <= 131072 fits
bits = np.arange(65536, dtype=np.uint16)
finite = bits[np.isfinite(np.frombuffer(bits.tobytes(), dtype=np.float16)).astype(bool)]
allv = torch.from_numpy(np.frombuffer(finite.tobytes(), dtype=np.float16).copy()).to(torch.float16).cuda()
n = allv.numel()
out = kernel.run(allv)
ref = kernel.reference(allv.cpu()).cuda()
err = (out.double() - ref.double()).abs()
print(json.dumps({
    "metric": "max abs error over exhaustive FP16 domain vs FP64 reference",
    "n_values": int(n),
    "max_abs_err": float(err.max()),
    "tolerance": 1e-6,
    "n_violations": int((err > 1e-6).sum()),
    "worst_x": float(allv[err.argmax()].item()),
    "worst_out": float(out[err.argmax()].item()),
    "worst_ref": float(ref[err.argmax()].item()),
    "max_err_near_zero_1e-3": float(err[(allv.abs() < 1e-3)].max()) if (allv.abs() < 1e-3).any() else 0.0,
    "max_err_x8_32": float(err[(allv.abs() >= 8) & (allv.abs() < 32)].max()),
}))