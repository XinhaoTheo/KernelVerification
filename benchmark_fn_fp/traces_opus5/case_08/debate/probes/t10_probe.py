
import sys, json, torch
sys.path.insert(0,"/root/cases/case_08")
from kernel import stochastic_round_to_grid, STEP
dev="cuda"
N=128
k=torch.arange(-64,64, device=dev).float()
x_exact=(k*STEP).float()
x_dn=torch.nextafter(x_exact, torch.tensor(-1e30, device=dev))
x_up=torch.nextafter(x_exact, torch.tensor(1e30, device=dev))

def stats(x, tag, S=200):
    moved=torch.zeros(N, device=dev)
    offgrid_max=0.0
    for s in range(S):
        o=stochastic_round_to_grid(x, 777+s)
        nearest=torch.round(o/STEP)*STEP
        offgrid_max=max(offgrid_max, float((o-nearest).abs().max()))
        moved+=( (o-x).abs() > 0.5*STEP ).float()
    fr=moved/S
    return {tag+"_max_offgrid_abs":offgrid_max,
            tag+"_max_frac_moved_full_step":float(fr.max()),
            tag+"_mean_frac_moved_full_step":float(fr.mean()),
            tag+"_num_elems_moved_ge_50pct":int((fr>=0.5).sum().item())}

res={}
res.update(stats(x_exact,"exact"))
res.update(stats(x_dn,"minus1ulp"))
res.update(stats(x_up,"plus1ulp"))

# host-side inspection of the fp32 arithmetic the kernel does
lower=torch.floor(x_exact/STEP)*STEP
p=(x_exact-lower)/STEP
res["exact_p_up_min"]=float(p.min()); res["exact_p_up_max"]=float(p.max())
res["exact_frac_p_up_gt_0.5"]=float((p>0.5).float().mean())
res["exact_frac_lower_ne_x"]=float((lower!=x_exact).float().mean())
o=stochastic_round_to_grid(x_exact,1)
res["example_max_abs_out_minus_x"]=float((o-x_exact).abs().max())
res["example_mean_abs_out_minus_x"]=float((o-x_exact).abs().mean())
print(json.dumps(res))
