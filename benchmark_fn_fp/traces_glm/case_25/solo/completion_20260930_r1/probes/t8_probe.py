import torch, json, sys
sys.path.insert(0, "/root/cases/case_25")
from kernel import requantize

torch.manual_seed(0)
x = torch.randn(1024, device="cuda") * 3

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

z = x.clone(); means = []
for k in range(51):
    z = requantize(z)
    means.append(z.abs().mean().item())

res = {
    "metric": "elementwise mismatch vs round-to-nearest reference + repeated-application bias",
    "n": int(x.numel()),
    "max_abs_err_vs_reference": float(diff.max()),
    "frac_mismatched": float((diff > 1e-6).float().mean()),
    "mean_abs_initial": float(x.abs().mean()),
    "mean_abs_after_1": means[0],
    "mean_abs_after_10": means[9],
    "mean_abs_after_50": means[49],
    "bias_ratio_after_50": means[49] / x.abs().mean().item(),
}
print("RESULT_JSON_BEGIN")
print(json.dumps(res))
print("RESULT_JSON_END")
