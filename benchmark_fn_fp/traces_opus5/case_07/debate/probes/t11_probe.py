
import sys, json, math, torch
sys.path.insert(0,"/root/cases/case_08")
from kernel import stochastic_round_to_grid, STEP
dev="cuda"
N=4096
x=torch.full((N,), 0.125, device=dev)  # lower=0.10, p_up=0.5
lower=torch.floor(x/STEP)*STEP; upper=lower+STEP
p=float(((x-lower)/STEP)[0])

fracs=[]; runs=[]; pats=set(); bits_list=[]
for s in range(64):
    o=stochastic_round_to_grid(x, 1000+s)
    b=(o==upper).to(torch.int8)
    bits_list.append(b)
    fracs.append(float(b.double().mean()))
    bl=b.cpu().numpy()
    mx=1; cur=1
    for i in range(1,len(bl)):
        cur = cur+1 if bl[i]==bl[i-1] else 1
        mx=max(mx,cur)
    runs.append(mx)
    pats.add(bytes(bl.tobytes()))

B=torch.stack(bits_list).double()
z=[(f-p)/math.sqrt(p*(1-p)/N) for f in fracs]
# lag-1 within-seed autocorrelation, averaged
c=B-B.mean(dim=1,keepdim=True)
lag1=float(((c[:,:-1]*c[:,1:]).mean(dim=1)/c.var(dim=1,unbiased=False)).mean())
# repeat check: same seed twice
o1=stochastic_round_to_grid(x,4242); o2=stochastic_round_to_grid(x,4242)
res=dict(N=N,p_up=p,seeds=len(fracs),
 min_frac=min(fracs),max_frac=max(fracs),
 max_abs_z=max(abs(v) for v in z),
 max_run_len=max(runs), mean_run_len=sum(runs)/len(runs),
 distinct_patterns=len(pats),
 mean_lag1_autocorr=lag1,
 same_seed_reproducible=bool(torch.equal(o1,o2)),
 cross_seed_max_pattern_match=float(max(
     float((bits_list[i]==bits_list[j]).double().mean())
     for i in range(8) for j in range(8) if i<j)),
)
print(json.dumps(res))
