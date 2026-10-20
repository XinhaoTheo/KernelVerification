import json, numpy as np, torch, sys
sys.path.insert(0, "/root/evidence_cases/case_e01")
import kernel
x, y, q = kernel.make_inputs()
out = kernel.run(x, y, q)
X = x.cpu().numpy().astype(np.float64)
Y = y.cpu().numpy().astype(np.float64)
Q = q.cpu().numpy().astype(np.float64)
mu_x = X.mean(axis=0); mu_y = Y.mean()
Xc = X - mu_x; yc = Y - mu_y
lam = 1.0/1024
A = Xc.T@Xc/32 + lam*np.eye(2)
beta = np.linalg.solve(A, Xc.T@yc/32)
pred = mu_y + (Q - mu_x)@beta
o = out.cpu().numpy().astype(np.float64)
err = np.linalg.norm(o-pred)/max(np.linalg.norm(pred),0.1)
print(json.dumps({"metric":"relative_l2_vs_float64_prediction","output":o.tolist(),"prediction":pred.tolist(),"beta":beta.tolist(),"relative_error":float(err),"tolerance":0.05,"finite":bool(np.all(np.isfinite(o))),"shape":list(out.shape),"passed":bool(err<=0.05)}))