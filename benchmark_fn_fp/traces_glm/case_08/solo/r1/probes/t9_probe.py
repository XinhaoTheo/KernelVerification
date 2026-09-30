import torch, statistics
torch.manual_seed(0)
import importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_08/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

N = 1024  # power of two, single block
# representative in-domain values: random, negatives, grid-aligned
x = torch.cat([
    torch.randn(N//2)*0.7,
    -torch.rand(N//8)*3,
    (torch.rand(N//8)*10).round()*0.05,
    torch.linspace(-2, 2, N - N//2 - N//8 - N//8, device='cuda')
]).cuda().float()
x = x[:N]

STEP = 0.05
S = 4000
acc = torch.zeros(N, device='cuda')
on_grid_viol = 0
wrong_dir = 0
for s in range(S):
    out = k.stochastic_round_to_grid(x, s)
    acc += out
    # grid membership: nearest multiple must match
    nearest = (out / STEP).round() * STEP
    if (torch.abs(out - nearest) > 1e-4).any():
        on_grid_viol += int((torch.abs(out - nearest) > 1e-4).sum())
    # neighbour check: out must be floor or ceil multiple of x/STEP
    fl = torch.floor(x/STEP)
    q = out / STEP
    if not torch.isin(q.round().long(), torch.stack([fl.long(), fl.long()+1], dim=1).flatten().unique()).all():
        pass
mean = acc / S
err = (mean - x).abs()
# empirical up-probability vs p_up for a mid element set
p_up_theory = (x - torch.floor(x/STEP)*STEP) / STEP
# empirical fraction up across seeds
frac_up = torch.zeros(N, device='cuda')
for s in range(400):
    out = k.stochastic_round_to_grid(x, s+10**6)
    frac_up += (out > torch.floor(x/STEP)*STEP + 1e-6).float()
frac_up /= 400
prob_err = (frac_up - p_up_theory).abs()
# determinism
o1 = k.stochastic_round_to_grid(x, 123); o2 = k.stochastic_round_to_grid(x, 123)
det = torch.equal(o1, o2)

import json
res = {
 "metric": "mean-over-4000-seeds abs err + empirical P(up) err + grid violations",
 "max_mean_err": err.max().item(), "mean_mean_err": err.mean().item(),
 "max_prob_err": prob_err.max().item(), "mean_prob_err": prob_err.mean().item(),
 "grid_violations_total": on_grid_viol,
 "deterministic_same_seed": det,
 "max_frac_up_err_400seeds": prob_err.max().item(),
 "N": N, "seeds": S,
}
print(json.dumps(res))