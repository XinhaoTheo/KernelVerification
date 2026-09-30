import numpy as np, torch, json
from fractions import Fraction
import importlib.util
spec = importlib.util.spec_from_file_location("k","/root/evidence_cases/case_e17/kernel.py")
# fall back: reconstruct input generator locally
SEED=203795
rng = np.random.Generator(np.random.PCG64(SEED))
low = rng.uniform(0.25,1.75,(4,8)).astype(np.float32)
interior = np.concatenate((np.full((4,1),2.0**30,np.float32),np.full((4,1),-(2.0**30),np.float32),low),axis=1)
order = rng.permutation(10)
x = np.empty((4,12),dtype=np.float32)
x[:,0]=np.float32(2.0**80); x[:,1:-1]=interior[:,order]; x[:,-1]=np.float32(-(2.0**80))
# exact targets via Fraction of exact float32 values
def exact_row(row):
    return sum(Fraction(float(v)) for v in x[row])
targets=[float(exact_row(r)) for r in range(4)]
# load kernel module and run on GPU
import sys
sys.path.insert(0,"/root/evidence_cases/case_e17")
import kernel as kmod
xs = kmod.make_inputs("cuda")
out = kmod.run(xs[0]).detach().cpu().numpy().astype(np.float64)
rel = float(np.linalg.norm(out-np.array(targets))/max(np.linalg.norm(targets),1e-12))
per_row=[abs(out[i]-targets[i])/max(abs(targets[i]),1e-12) for i in range(4)]
print(json.dumps({"targets":targets,"output":out.tolist(),"relative_l2":rel,"tolerance":1e-5,"per_row_rel":per_row,"within_tol":rel<=1e-5}))