import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_11/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

torch.manual_seed(1)
S, H, D = 4, 2, 65  # odd headdim
dev = "cuda"
x = torch.randn(S, H, D, device=dev, dtype=torch.float32)
cos = torch.randn(S, D // 2, device=dev)
sin = torch.randn(S, D // 2, device=dev)
out = k.apply_rotary(x, cos, sin, True)
ref = torch.empty_like(x)
x0 = x[..., 0::2][:, :, :D//2]; x1 = x[..., 1::2][:, :, :D//2]
c = cos[:, None, :]; s = sin[:, None, :]
ref[..., 0::2][:, :, :D//2] = x0*c - x1*s
ref[..., 1::2][:, :, :D//2] = x0*s + x1*c
# reference for last dim: unspecified by contract; check what kernel produced
last_dim_vals = out[..., -1]
x_last = x[..., -1]
print(json.dumps({"claim":"c2","D":D,"last_dim_allclose_to_input":bool(torch.allclose(last_dim_vals, x_last)),
  "last_dim_equals_zero":bool((last_dim_vals==0).all()),
  "last_dim_max_abs_vs_input":(last_dim_vals-x_last).abs().max().item(),
  "note":"last dim never written; out=torch.empty_like so value depends on allocator; mismatch vs input is not per se a contract violation since pairing contract presumes even headdim"}))
