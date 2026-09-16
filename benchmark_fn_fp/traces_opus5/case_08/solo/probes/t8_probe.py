
import torch, json, math, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_08/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
dev='cuda'; STEP=0.05
torch.manual_seed(1)

N=1024
# x with known fractional positions: cycle p in {0.05,...,0.95} plus random
frac = torch.rand(N, device=dev)*0.98+0.01
base = (torch.randint(-20,20,(N,),device=dev).float())*STEP
x = base + frac*STEP
lower = torch.floor(x/STEP)*STEP
p_ref = ((x-lower)/STEP).clamp(0,1)

T=4000
seeds = torch.randint(0, 2**31-1, (T,)).tolist()
cnt = torch.zeros(N, device=dev)
acc = torch.zeros(N, device=dev, dtype=torch.float64)
for s in seeds:
    out = m.stochastic_round_to_grid(x, s)
    cnt += (out > lower + 0.5*STEP).float()
    acc += out.double()
phat = cnt/T
mean_out = (acc/T).float()
se = torch.sqrt((p_ref*(1-p_ref)/T).clamp_min(1e-12))
z = (phat-p_ref)/se
absz = z.abs()
# global mean over all elements&seeds
global_bias = float((mean_out - x).abs().max())
# expected max |z| for 1024 normal draws ~3.3
res = dict(
  metric="per-element empirical P(upper) vs p_up, z-score over T seeds; and |E[out]-x|",
  reason="contract fixes P(out=upper)=(x-lower)/STEP so E[out]==x",
  T=T, N=N,
  max_abs_z=float(absz.max()), mean_abs_z=float(absz.mean()),
  frac_abs_z_gt_4=float((absz>4).float().mean()),
  corr_phat_pref=float(torch.corrcoef(torch.stack([phat.double(),p_ref.double()]))[0,1]),
  max_abs_mean_err=global_bias,
  mean_signed_err=float((mean_out-x).mean()),
  phat_min=float(phat.min()), phat_max=float(phat.max()),
  sample_pairs=[[round(float(p_ref[i]),4), round(float(phat[i]),4)] for i in range(0,N,128)],
)

# degeneracy / seed-sensitivity checks
o1 = m.stochastic_round_to_grid(x, 11); o2 = m.stochastic_round_to_grid(x, 12)
res['frac_diff_seed11_vs_12'] = float((o1!=o2).float().mean())
o1b = m.stochastic_round_to_grid(x, 11)
res['same_seed_deterministic'] = bool(torch.equal(o1,o1b))
# uniformity of the implied r across offsets: use x with p=0.5 everywhere
xh = torch.full((N,), 0.5*STEP, device=dev)
ups = torch.zeros(N, device=dev)
for s in seeds[:2000]:
    ups += (m.stochastic_round_to_grid(xh, s) > 0.5*0.5*STEP+1e-9).float()
ph = ups/2000
res['p_half_mean']=float(ph.mean()); res['p_half_min']=float(ph.min()); res['p_half_max']=float(ph.max())
res['p_half_max_abs_z']=float(((ph-0.5).abs()/math.sqrt(0.25/2000)).max())
print(json.dumps(res, indent=1))
