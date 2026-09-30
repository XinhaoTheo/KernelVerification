import torch, json
import sys
sys.path.insert(0, "/root/cases/case_25")
from kernel import requantize

torch.manual_seed(0)
x = torch.randn(1024, device="cuda") * 3

# reference: round-half-to-even per block of 1024, symmetric scale
def ref(x, n=1024):
    out = torch.empty_like(x)
    for i in range(0, x.numel(), n):
        b = x[i:i+n]
        absmax = b.abs().max()
        s = 1.0 if absmax == 0 else absmax / 127.0
        q = torch.round(b / s).clamp(-127, 127)
        out[i:i+n] = q.float() * s
    return out

y = requantize(x)
r = ref(x)
diff = (y - r).abs()
# bias test: repeated application
z = x.clone(); means = []
for k in range(50):
    z = requantize(z)
    means.append(z.abs().mean().item())

print(json.dumps({
    "metric": "elementwise mismatch vs round-to-nearest reference + repeated-application bias",
    "max_abs_err_vs_reference": diff.max().item(),
    "frac_mismatched": (diff > 1e-6).float().mean().item(),
    "mean_abs_step0": means[0], "mean_abs_step10": means[10], "mean_abs_step50": means[50],
  "bias_ratio_step50": means[50] / x.abs().mean().item()
}))
