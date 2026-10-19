import torch, json, sys
sys.path.insert(0, "/root/cases/case_147")
import kernel

# All FP16 representable values in (0.66, 1]: start from 0.666 (first step above 0.66)
xs = []
v = 0.666015625
while v <= 1.0:
    xs.append(v)
    t = torch.tensor(v, dtype=torch.float16)
    nxt = t + torch.tensor(float(2**-10), dtype=torch.float16) if v < 1 else None
    # iterate via representable successor
    if v == 1.0: break
    # use tensor nextafter through float16 casting
    base = torch.tensor(v, dtype=torch.float32)
    # find next representable fp16 above
    nxt = torch.tensor(v, dtype=torch.float16)
    found = False
    for d in [2**-11, 2**-10, 2**-9]:
        cand = (nxt.to(torch.float32) + d).to(torch.float16)
        if cand.to(torch.float32).item() > v:
            v = cand.item(); found = True; break
    if not found:
        v = 1.0

x = torch.tensor(xs, dtype=torch.float16, device="cuda")
out = kernel.run(x)
ref = torch.atan(x.to(torch.float64)).cpu()
err = (out.cpu().to(torch.float64) - ref).abs()
res = {
    "n_mid_values": len(xs),
    "max_abs_err": err.max().item(),
    "argmax_x": x.cpu().float()[err.argmax()].item(),
    "n_violations_1e-6": int((err > 1e-6).sum()),
    "tolerance": 1e-6,
}
print(json.dumps(res))