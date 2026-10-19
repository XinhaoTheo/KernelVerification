
import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_08/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
dev='cuda'; STEP=0.05
torch.manual_seed(0)

def ulp(v):
    v=v.abs().double()
    return torch.where(v>0, torch.pow(2.0, torch.floor(torch.log2(v.clamp_min(1e-30)))-23), torch.full_like(v,1e-45))

report={}
worst=[]
maxrel=0.0
tot=0; off=0
def check(x, seed, tag):
    global maxrel, tot, off, worst
    out = m.stochastic_round_to_grid(x, seed)
    xd = x.double(); od = out.double()
    k = torch.floor(xd/0.05)
    lo = k*0.05; hi=(k+1)*0.05
    d = torch.minimum((od-lo).abs(), (od-hi).abs())
    u = ulp(od).clamp_min(ulp(xd))
    rel = d/u                     # distance to nearest exact grid pt, in ulps of the value
    mr = float(rel.max())
    maxrel = max(maxrel, mr)
    bad = rel > 2.0
    tot += x.numel(); off += int(bad.sum())
    if bad.any():
        i = int(bad.nonzero()[0,0])
        worst.append(dict(tag=tag, x=float(x[i]), out=float(out[i]), lo=float(lo[i]), hi=float(hi[i]), ulps=float(rel[i])))
    # also: is out within [lo - 2ulp, hi + 2ulp]?
    inb = (od >= lo - 2*u) & (od <= hi + 2*u)
    return dict(max_ulps_from_grid=mr, frac_gt_2ulp=float(bad.float().mean()),
                frac_outside_bracket=float((~inb).float().mean()))

report['uniform_pm2'] = check(torch.rand(1024, device=dev)*4-2, 12345, 'uniform_pm2')
report['uniform_pm100'] = check(torch.rand(1024, device=dev)*200-100, 42, 'uniform_pm100')
report['exact_multiples'] = check((torch.arange(1024, device=dev, dtype=torch.float32)-512)*STEP, 777, 'exact_mult')
report['tiny'] = check((torch.rand(1024, device=dev)*2-1)*1e-3, 9, 'tiny')
report['near_grid_eps'] = check(((torch.arange(1024, device=dev, dtype=torch.float32)-512)*STEP)*(1+1e-7), 3, 'near_grid')
report['negatives'] = check(-torch.rand(1024, device=dev)*10, 55, 'neg')

print(json.dumps(dict(metric="distance from output to nearest exact multiple of 0.05 (fp64 ref), measured in fp32 ulps of the value",
                      reason="contract requires out in {floor(x/STEP)*STEP, +STEP}; only fp32 rounding of that formula is admissible",
                      per_case=report, total=tot, total_gt_2ulp=off,
                      global_max_ulps=maxrel, examples=worst[:5])))
