import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("kernel", "/root/cases/case_25/kernel.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
torch.manual_seed(1)
x = torch.randn(1024, device="cuda")  # one block
x0 = x.clone()
cur = x.clone()
means = [cur.mean().item()]
for _ in range(5):
    cur = mod.requantize(cur)
    means.append(cur.mean().item())
# also positive-only tensor
p = x.abs().clone()
pm = [p.mean().item()]
for _ in range(5):
    p = mod.requantize(p)
    pm.append(p.mean().item())
print(json.dumps({
 "metric": "mean drift over repeated requantize applications",
 "reason": "floor should pull positive values down by ~half a step on first application",
 "mixed_means": means, "pos_only_means": pm,
 "mixed_first_step_delta": means[1] - means[0],
 "pos_first_step_delta": pm[1] - pm[0],
 "pos_frac_drift": (pm[-1] - pm[0]) / pm[0],
 "step_scale": x0.abs().max().item() / 127.0}))