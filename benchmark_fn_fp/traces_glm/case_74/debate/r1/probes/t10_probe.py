import torch, json, sys
sys.path.insert(0, "/root/cases/case_147")
import kernel

# Large-magnitude FP16 values: dense down from 65504 plus a wide sparse sweep
vals = [65504.0]
t = torch.tensor(65504.0, dtype=torch.float16)
for _ in range(500):
    t = torch.nextafter(t.to(torch.float32), torch.tensor(0.0, dtype=torch.float32)).to(torch.float16)
    vals.append(t.item())
# sparse sweep of log-spaced large magnitudes
import math
vals += [65504.0/(2**k) for k in range(0, 20)]
vals += [x*65504.0/64 for x in [1.0,0.99,0.5,0.1,0.01]]
x = torch.tensor(sorted(set(vals)), dtype=torch.float16, device="cuda")
out = kernel.run(x)
ref = torch.atan(x.to(torch.float64)).cpu()
err = (out.cpu().to(torch.float64) - ref).abs()
res = {
    "n_large_values": x.numel(),
    "max_abs_err": err.max().item(),
    "argmax_x": x.cpu().float()[err.argmax()].item(),
    "err_at_65504": err[x.cpu().float() == 65504.0].item() if (x.cpu().float()==65504.0).any() else None,
    "n_violations_1e-6": int((err > 1e-6).sum()),
    "tolerance": 1e-6,
}
print(json.dumps(res))