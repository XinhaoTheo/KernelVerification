import torch, json
import importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_08/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

dev = 'cuda'
N = 1024  # power of two, single block
x = torch.cat([
    torch.randn(N//2, device=dev)*0.7,
    -torch.rand(N//8, device=dev)*3,
    (torch.rand(N//8, device=dev)*10).round()*0.05,
    torch.linspace(-2, 2, N - N//2 - N//8 - N//8, device=dev)
]).float()[:N]

STEP = 0.05
S = 4000
acc = torch.zeros(N, device=dev)
on_grid_viol = 0
neighbour_viol = 0
fl = torch.floor(x/STEP)
for s in range(S):
    out = k.stochastic_round_to_grid(x, s)
    acc += out
    q = out / STEP
    on_grid_viol += int((torch.abs(q - q.round()) > 1e-3).sum().item())
    # out must equal floor or floor+1 multiple of STEP for that element
    qr = q.round()
    ok = (torch.isclose(qr, fl, atol=1e-3) | torch.isclose(qr, fl+1, atol=1e-3))
    neighbour_viol += int((~ok).sum().item())
mean = acc / S
err = (mean - x).abs()

p_up_theory = (x - fl*STEP) / STEP
frac_up = torch.zeros(N, device=dev)
for s in range(400):
    out = k.stochastic_round_to_grid(x, s+10**6)
    frac_up += (out > fl*STEP + 1e-6).float()
frac_up /= 400
prob_err = (frac_up - p_up_theory).abs()

o1 = k.stochastic_round_to_grid(x, 123); o2 = k.stochastic_round_to_grid(x, 123)
det = torch.equal(o1, o2)
o3 = k.stochastic_round_to_grid(x, 124)
diff_seed_differs = not torch.equal(o1, o3)

res = {
 "metric": "mean-over-4000-seeds abs err + empirical P(up) err + grid/neighbour violations",
 "max_mean_err": err.max().item(), "mean_mean_err": err.mean().item(),
 "max_prob_err_400seeds": prob_err.max().item(), "mean_prob_err": prob_err.mean().item(),
 "grid_violations_total": on_grid_viol,
 "neighbour_violations_total": neighbour_viol,
 "deterministic_same_seed": bool(det),
 "different_seed_gives_different_output": bool(diff_seed_differs),
 "N": N, "seeds": S,
}
print(json.dumps(res))