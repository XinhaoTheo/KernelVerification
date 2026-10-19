import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_23/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)
N, C, H, W = 2, 4, 3, 3
x = torch.randn(N, C, H, W, device=dev)  # default NCHW-contiguous
assert x.is_contiguous() and not x.is_contiguous(memory_format=torch.channels_last)
s = torch.arange(1, C+1, dtype=x.dtype, device=dev)  # generic non-uniform
out = k.scale_channels(x, s, C)
ref = x * s.view(1, C, 1, 1)
mism = (out != ref)
lin = torch.arange(N*C*H*W, device=dev) % C
wrong_ref = x.reshape(-1) * s[lin]
match_wrong = torch.equal(out.reshape(-1), wrong_ref)
print(json.dumps({
  "layout": "NCHW contiguous", "shape": [N,C,H,W],
  "max_abs_err": (out-ref).abs().max().item(),
  "num_mismatch": mism.sum().item(), "numel": out.numel(),
  "matches_scale_of_linear_mod_C": match_wrong
}))