
import torch, json, importlib.util, sys
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_08/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

torch.manual_seed(0)
dev='cuda'
STEP=0.05
results={}
bad_examples=[]
tot=0; bad=0
maxdev=0.0

def check(x, seed, tag):
    global tot,bad,maxdev
    out = m.stochastic_round_to_grid(x, seed)
    lower = torch.floor(x/STEP)*STEP
    upper = lower + STEP
    ok = (out==lower)|(out==upper)
    tot += x.numel(); bad += int((~ok).sum())
    # also measure distance from output to nearest of {lower,upper}
    d = torch.minimum((out-lower).abs(), (out-upper).abs()).max().item()
    maxdev = max(maxdev, d)
    if (~ok).any():
        idx=(~ok).nonzero()[:5,0]
        for i in idx.tolist():
            bad_examples.append(dict(tag=tag,x=float(x[i]),out=float(out[i]),lower=float(lower[i]),upper=float(upper[i])))
    return float((~ok).float().mean())

cases={}
x1 = (torch.rand(1024, device=dev)*4-2)          # random in [-2,2)
cases['uniform_pm2']=check(x1, 12345, 'uniform_pm2')
x2 = (torch.arange(1024, device=dev, dtype=torch.float32)-512)*STEP   # exact multiples
cases['exact_multiples']=check(x2, 777, 'exact_multiples')
x3 = x2 + 1e-7
cases['just_above_grid']=check(x3, 778, 'just_above_grid')
x4 = x2 - 1e-7
cases['just_below_grid']=check(x4, 779, 'just_below_grid')
x5 = (torch.rand(256, device=dev)*200-100)
cases['uniform_pm100']=check(x5, 4242, 'uniform_pm100')
x6 = torch.zeros(64, device=dev)
cases['zeros']=check(x6, 5, 'zeros')

print(json.dumps(dict(metric="fraction of outputs not in {lower,upper}",
                      per_case_bad_frac=cases, total=tot, total_bad=bad,
                      max_dist_to_nearest_bracket=maxdev,
                      examples=bad_examples[:10]), indent=1))
