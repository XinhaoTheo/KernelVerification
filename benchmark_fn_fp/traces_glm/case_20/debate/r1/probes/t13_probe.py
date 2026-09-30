import torch, math, json, sys
sys.path.insert(0, "/root/cases/case_20")
from kernel import layer_norm

eps = 1e-5
N = 512
torch.manual_seed(0)

def ref(x, w, b, eps):
    xf = x.float()
    mean = xf.mean(-1, keepdim=True)
    var = ((xf - mean) ** 2).mean(-1, keepdim=True)
    rstd = 1.0 / torch.sqrt(var + eps)
    return (xf - mean) * rstd * w.float() + b.float()

results = {}
rows = []
deltas = [1e-4, 1e-3, 1e-2]
for d in deltas:
    # constant row + single perturbed element
    r = torch.full((N,), 0.7)
    r[3] = 0.7 + d
    rows.append(("single_perturb", d, r))
    # symmetric pattern: two elements perturbed +-d (mean stays 0.7)
    r2 = torch.full((N,), 0.7)
    r2[0] = 0.7 + d
    r2[1] = 0.7 - d
    rows.append(("symmetric", d, r2))

w = torch.randn(N)
b = torch.randn(N)

worst = {"max_abs_err": 0.0, "max_rel_err": 0.0, "label": None}
for label, d, r in rows:
    x = r.unsqueeze(0).contiguous()
    y = layer_norm(x, w, b, eps)
    yr = ref(x, w, b, eps)
    xf = x.float()
    mean = xf.mean().item()
    var = ((xf - mean) ** 2).mean().item()
    contract_rstd = 1.0 / math.sqrt(var + eps)
    kernel_rstd_implied = 1.0 / (math.sqrt(var) + eps)
    diff = (y.float() - yr).abs()
    denom = yr.abs().clamp_min(1e-12)
    rel = (diff / denom).max().item()
    abs_err = diff.max().item()
    ratio = kernel_rstd_implied / contract_rstd
    results[f"{label}_delta={d}"] = {
        "var": var, "contract_rstd": contract_rstd,
        "kernel_formula_rstd": kernel_rstd_implied,
        "rstd_ratio": ratio,
        "max_abs_err": abs_err, "max_rel_err": rel,
    }
    if abs_err > worst["max_abs_err"]:
        worst = {"max_abs_err": abs_err, "max_rel_err": rel, "label": f"{label}_delta={d}"}

# benign control: randn row (should agree)
x = torch.randn(1, N)
y = layer_norm(x, w, b, eps); yr = ref(x, w, b, eps)
results["randn_control_max_abs_err"] = (y.float() - yr).abs().max().item()

# check allclose at typical tolerance 1e-5 on the adversarial case
r = torch.full((N,), 0.7); r[0] = 0.7 + 1e-3; r[1] = 0.7 - 1e-3
x = r.unsqueeze(0).contiguous()
y = layer_norm(x, w, b, eps); yr = ref(x, w, b, eps)
results["allclose_1e-5_small_var"] = bool(torch.allclose(y.float(), yr, atol=1e-5, rtol=1e-5))

print(json.dumps({"metric": "elementwise abs/rel error of kernel y vs contract formula y with eps=1e-5, controlled-variance rows",
                  "rationale": "claim c2 states y deviates from contract 1/sqrt(var+eps) for small nonzero var; direct y comparison is authoritative",
                  "worst": worst, "cases": results}))
