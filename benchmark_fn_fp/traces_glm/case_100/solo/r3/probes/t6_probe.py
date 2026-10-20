import sys, torch, numpy as np
sys.path.insert(0, "/root/cases/case_100")
import kernel

bits = np.arange(65536, dtype=np.uint16)
f16 = torch.from_numpy(bits.view(np.float16).copy())
# exclude NaN/Inf
finite = torch.isfinite(f16)
x = f16[finite].to(torch.float32)
print("num finite fp16 values:", x.numel(), "max:", x.max().item(), "min subnormal/normal present")

dev = "cuda"
# run in chunks of 131072 -> whole set fits (61440 finite values? actually ~65504)
xd = x.to(torch.float16).to(dev)
out = kernel.run(xd)
ref = torch.asinh(xd.to(torch.float64))
err = (out.to(torch.float64) - ref).abs()
mx = err.max().item()
idx = int(err.argmax().item())
print("max abs err:", mx)
print("worst input:", xd[idx].item(), "out:", out[idx].item(), "ref:", ref[idx].item())
print("count > 1e-6:", int((err > 1e-6).sum().item()))
print("count > 0 (nonzero err):", int((err > 0).sum().item()))
top = torch.topk(err, 10).values
topi = torch.topk(err, 10).indices
for v,i in zip(top.tolist(), topi.tolist()):
    print(f"x={xd[i].item():.6g} err={v:.3e} out={out[i].item():.9f} ref={ref[i].item():.9f}")
import json
print(json.dumps({"max_abs_err": mx, "n_over_tol": int((err>1e-6).sum().item()), "n_tested": x.numel()}))
