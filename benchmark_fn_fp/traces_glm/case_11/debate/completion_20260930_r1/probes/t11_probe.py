import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_11/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

torch.manual_seed(2)
S, H, D = 6, 2, 96  # not power of two -> BLOCK_K=128
dev = "cuda"
x = torch.randn(S, H, D, device=dev, dtype=torch.float32)
cos = torch.randn(S, D // 2, device=dev)
sin = torch.randn(S, D // 2, device=dev)
out = k.apply_rotary(x, cos, sin, True)
x0 = x[..., 0::2]; x1 = x[..., 1::2]
ref = torch.empty_like(x)
ref[..., 0::2] = x0*cos[:,None,:] - x1*sin[:,None,:]
ref[..., 1::2] = x0*sin[:,None,:] + x1*cos[:,None,:]
diff = (out - ref).abs()
print(json.dumps({"claim":"c3","D":D,"BLOCK_K":128,
  "max_abs_err_interleaved":diff.max().item(),
  "max_err_even_lanes":diff[...,0::2].max().item(),
  "max_err_odd_lanes":diff[...,1::2].max().item(),
  "allclose_1e-5":bool(torch.allclose(out, ref, atol=1e-5))}))
