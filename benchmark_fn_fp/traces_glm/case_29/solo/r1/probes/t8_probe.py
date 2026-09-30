import torch, json
from kernel import fp8_roundtrip

torch.manual_seed(0)

# Reference: PyTorch native e4m3 cast (round-to-nearest, saturating)
def ref(x):
    return x.to(torch.float8_e4m3fn).to(torch.float32)

# Targeted range probes
x = torch.tensor([0.001, 0.0009, 2.0**-9, 2.0**-7, 0.01, 500.0, 449.5, 1024.0, 3.7, -3.7, 7.6, 0.0], device='cuda')
out = fp8_roundtrip(x)
r = ref(x)

# e4m3fn representable values (subnormals step 2^-9, max 448)
def is_e4m3(v):
    v = torch.tensor(v, device='cuda')
    return bool((v.to(torch.float8_e4m3fn).to(torch.float32) == v).all()) if v.numel() else True

res = {
    "inputs": x.tolist(),
    "kernel_out": out.tolist(),
    "native_ref": r.tolist(),
    "kernel_outputs_representable": [is_e4m3([v]) for v in out.tolist()],
    "mismatch": (out != r).tolist(),
    "max_abs_err": float((out - r).abs().max()),
}

# Broad sweep across magnitudes
mags = torch.tensor([10.0**k for k in range(-5, 5)], device='cuda')
xs = (mags[:, None] * torch.linspace(-1.05, 1.05, 41, device='cuda')[None, :]).reshape(-1)
xs = xs[torch.isfinite(xs)]
outs = fp8_roundtrip(xs)
rs = ref(xs)
back = outs.to(torch.float8_e4m3fn).to(torch.float32)
res["sweep_size"] = int(xs.numel())
res["sweep_not_representable_count"] = int((back != outs).sum())
res["sweep_mismatch_vs_native_count"] = int((outs != rs).sum())
# worst cases where kernel deviates from native beyond half-step
d = (outs - rs).abs()
step = torch.exp2(torch.floor(torch.log2(rs.abs().clamp_min(1e-30))) - 3.0)
res["sweep_max_abs_err"] = float(d.max())
res["sweep_rel_err_violations_gt_half_step"] = int((d > 0.5*step + 1e-12).sum())
print(json.dumps(res))