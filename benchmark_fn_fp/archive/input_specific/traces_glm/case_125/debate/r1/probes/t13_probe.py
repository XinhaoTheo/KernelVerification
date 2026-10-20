
import importlib.util, torch, numpy as np, json
spec = importlib.util.spec_from_file_location("k", "/root/evidence_cases/case_e01/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
x, y, q = k.make_inputs()
# reproduce initial_probe's float32 reference pipeline exactly (CPU float32 torch ops mirror GPU float32 arithmetic in fp32)
stats = torch.zeros((8,), dtype=torch.float32)
for i in range(32):
    x0, x1, yi = x[i,0].cpu(), x[i,1].cpu(), y[i].cpu()
    stats = stats + torch.stack((x0,x1,yi,x0*x0,x0*x0*0+x0*x1,x1*x1,x0*yi,x1*yi)).float()
m0,m1,my,s00,s01,s11,c0,c1 = [float(v) for v in (stats/32)]
cov00=s00-m0*m0+1/1024; cov01=s01-m0*m1; cov11=s11-m1*m1+1/1024
r0=c0-m0*my; r1=c1-m1*my
det=cov00*cov11-cov01*cov01
b0=(cov11*r0-cov01*r1)/det; b1=(cov00*r1-cov01*r0)/det
qd = q.double().cpu()
fp32_ref = torch.tensor([my+(qq[0]-m0)*b0+(qq[1]-m1)*b1 for qq in qd])
# float64 contract reference
xd=x.double().cpu(); yd=y.double().cpu()
mu_x=xd.mean(0); mu_y=yd.mean(); Xc=xd-mu_x; yc=yd-mu_y
A=(Xc.T@Xc)/32+torch.eye(2)/1024; rhs=(Xc.T@yc)/32
beta=torch.linalg.solve(A,rhs); pred=mu_y+(qd-mu_x)@beta
diff=(fp32_ref.double()-pred).norm()/max(pred.norm().item(),0.1)
print(json.dumps({
 "metric":"relative L2 gap between initial_probe's float32 reference and float64 contract prediction",
 "fp32_reference": fp32_ref.tolist(), "f64_prediction": pred.tolist(),
 "relative_gap": float(diff.item()), "tolerance": 0.05,
 "materially_different": bool(diff.item()>0.05),
 "note":"fp32 reference pipeline reproduces initial_probe's float32 accumulation path; gap shows whether the probe reference stands in for the float64 target"
}))
