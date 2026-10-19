import torch, json
torch.manual_seed(0)
from kernel import stochastic_round_to_grid

STEP = 0.05
N = 64  # power of two, single block
# repeated elements with varied fractional distances, incl. negatives
vals = [0.375, -0.375, 0.0125, -0.0625, 1.234, -2.567, 0.9999, -0.9999] * 8
x = torch.tensor(vals, dtype=torch.float32, device='cuda')
S = 2000
outs = torch.stack([stochastic_round_to_grid(x, s) for s in range(S)])
mean_out = outs.mean().item()
mean_in = x.mean().item()
std_mc = (outs.std() / (S**0.5)).item()
bias = mean_out - mean_in
z = bias / std_mc if std_mc > 0 else 0.0
print(json.dumps({
    "N": N, "seeds": S, "mean_in": mean_in, "mean_out": mean_out,
    "bias": bias, "mc_std": std_mc, "z_score": z,
    "metric": "sample mean vs input mean with Monte-Carlo z-score; unbiasedness E[out]==x is the contract invariant"
}))
