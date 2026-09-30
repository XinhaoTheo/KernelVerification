
import importlib.util, torch, numpy as np, json
spec = importlib.util.spec_from_file_location("k", "/root/evidence_cases/case_e01/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
x, y, q = k.make_inputs()
out = k.run(x, y, q)
# float64 reference from stored float32 values
xd = x.double(); yd = y.double(); qd = q.double()
mu_x = xd.mean(0); mu_y = yd.mean()
Xc = xd - mu_x; yc = yd - mu_y
A = (Xc.T @ Xc) / 32.0 + (1/1024.0) * torch.eye(2, dtype=torch.float64, device=xd.device)
rhs = (Xc.T @ yc) / 32.0
beta = torch.linalg.solve(A, rhs)
pred = mu_y + (qd - mu_x) @ beta
rel = (out.double() - pred).norm() / max(pred.norm().item(), 0.1)
# float32 second-moment path (kernel-style) vs float64
xs = x.float(); ys = y.float()
m0 = xs[:,0].sum()/32; m1 = xs[:,1].sum()/32; my = ys.sum()/32
a = (xs[:,0]**2).sum()/32 - m0*m0 + 1/1024
b = (xs[:,0]*xs[:,1]).sum()/32 - m0*m1
c = (xs[:,1]**2).sum()/32 - m1*m1 + 1/1024
u = (xs[:,0]*ys).sum()/32 - m0*my
v = (xs[:,1]*ys).sum()/32 - m1*my
cov64 = ((Xc.T @ Xc)/32).flatten(); cross64 = ((Xc.T @ yc)/32).flatten()
f32 = torch.stack([a-1/1024,b,c-1/1024,u,v]); ref64 = torch.stack([cov64[0],cov64[1],cov64[3],cross64[0],cross64[1]])
relerr = ((f32.double()-ref64).abs()/ref64.abs()).tolist()
print(json.dumps({
 "metric":"relative L2 error vs float64 contract reference",
 "kernel_output": out.cpu().tolist(), "float64_pred": pred.cpu().tolist(),
 "relative_error": float(rel.item()), "tolerance": 0.05, "passed": bool(rel.item()<=0.05),
 "beta_f64": beta.cpu().tolist(),
 "cov_f32_relerr": relerr,
 "det_f64": float((A[0,0]*A[1,1]-A[0,1]**2).item()),
 "det_f32": float((a*c-b*b).item()),
 "all_finite": bool(torch.isfinite(out).all().item()),
}))
