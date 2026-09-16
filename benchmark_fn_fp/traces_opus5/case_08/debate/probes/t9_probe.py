
import sys, json, math, torch
sys.path.insert(0,"/root/cases/case_08")
from kernel import stochastic_round_to_grid, STEP

torch.manual_seed(0)
dev="cuda"
N=1024
fr=[0.1,0.25,0.5,0.75,0.9]
ks=torch.arange(N, device=dev)//len(fr) - 100
f=torch.tensor(fr*(N//len(fr)+1), device=dev)[:N]
x=((ks.float()*STEP)+f*STEP).float()
lower=torch.floor(x/STEP)*STEP
upper=lower+STEP
p_up=((x-lower)/STEP).clamp(0,1)

S=3000
cnt=torch.zeros(N, device=dev, dtype=torch.float64)
acc=torch.zeros(N, device=dev, dtype=torch.float64)
bad_membership=0
for s in range(S):
    out=stochastic_round_to_grid(x, s+12345)
    is_up=(out==upper)
    is_lo=(out==lower)
    bad_membership+=int((~(is_up|is_lo)).sum().item())
    cnt+=is_up.double()
    acc+=out.double()

freq=cnt/S
mean_out=acc/S
sigma=torch.sqrt(p_up.double()*(1-p_up.double())/S).clamp_min(1e-12)
z=(freq-p_up.double())/sigma
maxz=float(z.abs().max()); argz=int(z.abs().argmax())
mean_err=(mean_out-x.double()).abs()
res=dict(
 N=N, seeds=S, max_abs_z=maxz,
 max_z_index=argz, max_z_p_up=float(p_up[argz]), max_z_freq=float(freq[argz]),
 frac_abs_z_gt_5=float((z.abs()>5).double().mean()),
 mean_z=float(z.mean()), overall_freq_minus_p=float((freq-p_up.double()).mean()),
 max_abs_mean_err=float(mean_err.max()),
 max_abs_mean_err_rel_STEP=float(mean_err.max()/STEP),
 grand_mean_out_minus_x=float((mean_out-x.double()).mean()),
 nonmember_count=bad_membership,
)
print(json.dumps(res))
