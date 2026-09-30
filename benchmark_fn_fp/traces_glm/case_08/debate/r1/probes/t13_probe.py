import torch, json
from kernel import stochastic_round_to_grid
STEP = 0.05
vals = [0.15, -0.25, 0.30, -0.55, 1.10, 0.05, -0.05, 0.20, 0.35, -0.10]
x = torch.tensor(vals, dtype=torch.float32, device='cuda')
res = {}
S = 100
for v in vals:
    xi = torch.full((16,), v, dtype=torch.float32, device='cuda')
    o = torch.stack([stochastic_round_to_grid(xi, s) for s in range(S)])
    # fp32 grid computation per contract formula
    lo = (torch.floor(xi / STEP) * STEP)
    p_up = ((xi - lo) / STEP)
    out_lo = o.min().item(); out_hi = o.max().item()
    frac_up = (o > lo[0].item() + STEP/2).float().mean().item()
    # distance of outputs from exact decimal grid (k*0.05)
    k = torch.round(o / STEP)
    resid = (o - k * STEP).abs().max().item()
    res[repr(v)] = {
        "p_up_fp32": p_up[0].item(),
        "output_min": out_lo, "output_max": out_hi,
        "frac_upper": frac_up,
        "max_resid_from_decimal_grid": resid,
    }
print(json.dumps({
    "per_value": res,
    "metric": "p_up (should be 0 at exact multiples) and output distance from exact decimal grid k*0.05; wrong-grid-point means output differs from input by ~0.05"
}))
