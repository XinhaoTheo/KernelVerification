
import sys, json, math, torch
sys.path.insert(0,"/root/cases/case_08")
from kernel import stochastic_round_to_grid, STEP

dev="cuda"
torch.manual_seed(0)
N=1024
# half structured mid-cell values (incl. negatives), half uniform random in [-5,5]
fr=[0.1,0.25,0.5,0.75,0.9]
ks=torch.arange(N//2, device=dev)//len(fr) - 50
f=torch.tensor(fr*(N//2//len(fr)+1), device=dev)[:N//2]
x1=((ks.float()*STEP)+f*STEP).float()
x2=(torch.rand(N//2, device=dev)*10.0-5.0).float()
x=torch.cat([x1,x2]).contiguous()

xd=x.double()
kf=torch.floor(xd/0.05)
lo=kf*0.05
up=lo+0.05
p=((xd-lo)/0.05).clamp(0.0,1.0)

S=8000
cnt=torch.zeros(N, device=dev, dtype=torch.float64)
acc=torch.zeros(N, device=dev, dtype=torch.float64)
max_offgrid=0.0
max_nbr_dist=0.0
outside_nbr=0
for s in range(S):
    o=stochastic_round_to_grid(x, s+999331).double()
    dlo=(o-lo).abs(); dup=(o-up).abs()
    is_up=dup<dlo
    cnt+=is_up.double()
    acc+=o
    # distance to the chosen neighbour (should be ~0, few ulp)
    nbr=torch.minimum(dlo,dup)
    max_nbr_dist=max(max_nbr_dist,float(nbr.max()))
    # off-grid: distance to nearest multiple of 0.05
    og=(o-torch.round(o/0.05)*0.05).abs()
    max_offgrid=max(max_offgrid,float(og.max()))
    # neighbour-set violation: output more than 0.6*STEP from both neighbours
    outside_nbr+=int((nbr>0.6*0.05).sum().item())

freq=cnt/S
mean_out=acc/S
sig=torch.sqrt((p*(1-p)/S)).clamp_min(1e-12)
z=(freq-p)/sig
mz=float(z.abs().max()); ai=int(z.abs().argmax())
# mean-based z (same variance scaled by STEP)
zm=(mean_out-xd)/(0.05*sig)
mzm=float(zm.abs().max()); bi=int(zm.abs().argmax())
gm=float((mean_out-xd).mean())
gsig=float((0.05*torch.sqrt(p*(1-p)/S)).pow(2).mean().sqrt()/math.sqrt(N))
res=dict(
 N=N, seeds=S, metric="double-precision nearest-neighbour classification",
 max_abs_z_freq=mz, max_z_index=ai, max_z_p_true=float(p[ai]), max_z_freq=float(freq[ai]),
 frac_abs_z_gt_5=float((z.abs()>5).double().mean()),
 mean_z=float(z.mean()), std_z=float(z.std()),
 overall_freq_minus_p=float((freq-p).mean()),
 max_abs_z_mean=mzm, max_z_mean_index=bi,
 max_abs_mean_err=float((mean_out-xd).abs().max()),
 max_abs_mean_err_rel_STEP=float((mean_out-xd).abs().max()/0.05),
 grand_mean_out_minus_x=gm, grand_mean_sigma=gsig,
 grand_mean_z=gm/gsig if gsig>0 else None,
 max_offgrid_abs=max_offgrid, max_offgrid_rel_STEP=max_offgrid/0.05,
 max_dist_to_chosen_neighbour=max_nbr_dist,
 max_dist_to_chosen_neighbour_rel_STEP=max_nbr_dist/0.05,
 outside_neighbour_count=outside_nbr,
 fp32_ulp_at_5=float(torch.tensor(5.0).nextafter(torch.tensor(6.0))-5.0),
)
print(json.dumps(res))
