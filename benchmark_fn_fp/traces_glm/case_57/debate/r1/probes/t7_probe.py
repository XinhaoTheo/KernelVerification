import torch, numpy as np, json
import importlib.util
spec = importlib.util.spec_from_file_location("kernelmod", "/root/cases/case_130/kernel.py")
km = importlib.util.module_from_spec(spec); spec.loader.exec_module(km)

# All finite FP16 values, exhaustive
bits = np.arange(65536, dtype=np.uint32)
h = torch.from_numpy(bits.astype(np.uint16).view(np.float16))
finite = torch.isfinite(h)
vals = h[finite]  # all finite FP16 values, both signs
print("count finite fp16:", vals.numel(), "min", vals.min().item(), "max", vals.max().item())

x = vals.to("cuda")
out = km.run(x)

# Real-arithmetic SiLU on exact stored values, FP64
v = x.to(torch.float64)
ref = v * torch.sigmoid(v)
err = (out.to(torch.float64) - ref).abs()
max_err = err.max().item()
max_idx = err.argmax().item()

# region breakdown
mag = v.abs()
regions = {
    "all": err.max().item(),
    "mid_1_20": err[(mag>=1)&(mag<=20)].max().item(),
    "small_lt1": err[mag<1].max().item(),
    "large_gt20_le103": err[(mag>20)&(mag<=103)].max().item(),
    "huge_gt103": err[mag>103].max().item(),
}
print(json.dumps({
    "metric": "max abs error vs FP64 real-arithmetic SiLU over exhaustive finite FP16 domain",
    "tolerance": 1e-6,
    "max_abs_err": max_err,
    "breaches": int((err > 1e-6).sum().item()),
    "worst_input": x[max_idx].item(),
    "region_max_abs_err": regions,
    "verdict_signal": "breach" if max_err > 1e-6 else "pass_all_domain",
}))