import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_11/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

torch.manual_seed(0)
S, H, D = 8, 3, 64
dev = "cuda"
x = torch.randn(S, H, D, device=dev, dtype=torch.float32)
cos = torch.randn(S, D // 2, device=dev)
sin = torch.randn(S, D // 2, device=dev)

out = k.apply_rotary(x, cos, sin, True)
# independent reference: pair dim 2i with 2i+1
x0 = x[..., 0::2]; x1 = x[..., 1::2]
ref = torch.empty_like(x)
ref[..., 0::2] = x0 * cos[:, None, :] - x1 * sin[:, None, :]
ref[..., 1::2] = x0 * sin[:, None, :] + x1 * cos[:, None, :]

diff = (out - ref).abs()
max_err = diff.max().item()
# per-lane worst error
lane0 = diff[..., 0::2].max().item(); lane1 = diff[..., 1::2].max().item()
# also check cross-mode: does interleaved output match non-interleaved-style pairing?
xn0 = x[..., :D//2]; xn1 = x[..., D//2:]
refn = torch.empty_like(x)
refn[..., :D//2] = xn0*cos[:,None,:] - xn1*sin[:,None,:]
refn[..., D//2:] = xn0*sin[:,None,:] + xn1*cos[:,None,:]
cross_err = (out - refn).abs().max().item()
print(json.dumps({"claim":"c1","D":D,"max_abs_err_vs_interleaved_ref":max_err,
  "max_err_even_lanes":lane0,"max_err_odd_lanes":lane1,
  "max_abs_err_vs_noninterleaved_ref":cross_err,
  "matches_interleaved_ref": max_err < 1e-5}))
