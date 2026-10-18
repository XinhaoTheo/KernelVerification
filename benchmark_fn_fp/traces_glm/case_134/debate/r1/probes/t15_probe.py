import torch, json, importlib.util, numpy as np
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_134/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
# signed zeros, smallest/largest subnormals both signs, and all 2046 subnormal patterns
m = np.arange(1024, dtype=np.int32)
sub = np.frombuffer((m & 0x03FF).astype(np.uint16).tobytes(), dtype=np.float16).copy()
neg_sub = np.frombuffer(((m & 0x03FF) | 0x8000).astype(np.uint16).tobytes(), dtype=np.float16).copy()
parts = [
    np.array([0.0, -0.0], dtype=np.float16),
    np.array([6.104e-5, -6.104e-5, 5.96e-8, -5.96e-8], dtype=np.float16),
    sub, neg_sub,
]
xa = np.concatenate(parts)
x = torch.from_numpy(xa.copy()).cuda()
out = k.run(x)
ref = k.reference(x)
err = (out.double() - ref).abs()
i = int(err.argmax())
# sign of output for -0.0 (position 1): numerically equivalent per contract, but record it
negzero_out = out[1].item()
negzero_signbit = bool(np.float32(negzero_out).view(np.uint32) & 0x80000000)
res = {
  "n": int(x.numel()), "max_abs_err": float(err[i]), "tolerance": 1e-6,
  "num_exceed": int((err > 1e-6).sum()),
  "worst_x": float(x[i]), "worst_out": float(out[i]), "worst_ref": float(ref[i]),
  "negzero_out_value": negzero_out, "negzero_out_signbit": negzero_signbit,
  "min_subnormal_x": 5.96e-8,
}
print(json.dumps(res))