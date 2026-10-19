import torch, json, sys
sys.path.insert(0, "/root/cases/case_147")
import kernel

# FP16 values adjacent to thresholds 0.66 and 2.414213562373095
def neighbors(center, count=8):
    fs = torch.tensor(center, dtype=torch.float32)
    below = [fs.item()]
    for _ in range(count):
        fs = torch.nextafter(fs, torch.tensor(0.0, dtype=torch.float32))
        below.append(fs.item())
    above = []
    gs = torch.tensor(center, dtype=torch.float32)
    for _ in range(count):
        gs = torch.nextafter(gs, torch.tensor(10.0, dtype=torch.float32))
        above.append(gs.item())
    return below + above

vals = neighbors(0.66) + neighbors(2.414213562373095)
x = torch.tensor(sorted(set(vals)), dtype=torch.float16, device="cuda")
out = kernel.run(x)
ref = torch.atan(x.to(torch.float64)).cpu()
err = (out.cpu().to(torch.float64) - ref).abs()
worst = err.argmax()
res = {
    "n_boundary_values": x.numel(),
    "max_abs_err": err.max().item(),
    "argmax_x": x.cpu().float()[worst].item(),
    "per_value_err": {float(x.cpu().float()[i]): float(err[i]) for i in range(x.numel())},
    "n_violations_1e-6": int((err > 1e-6).sum()),
    "tolerance": 1e-6,
}
print(json.dumps(res))