
import torch, json
import sys
sys.path.insert(0, "/root/cases/case_23")
from kernel import scale_channels

torch.manual_seed(0)
dev = "cuda"
N,C,H,W = 2, 64, 8, 8
x_nhwc = torch.randn(N,H,W,C, device=dev, dtype=torch.float32).contiguous()
scale = torch.randn(C, device=dev) + 3.0
out = scale_channels(x_nhwc, scale, C)
ref = x_nhwc * scale  # broadcasting over last dim (channel)
err = (out - ref).abs().max().item()
res = {"metric": "max_abs_error NHWC channels-contiguous", "max_abs_error": err,
       "shape": list(out.shape), "dtype": str(out.dtype), "n_mismatch_gt_1e-5": int((out-ref).abs().gt(1e-5).sum())}
print(json.dumps(res))
