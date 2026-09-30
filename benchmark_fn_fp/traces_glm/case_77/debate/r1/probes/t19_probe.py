import numpy as np, json
from fractions import Fraction as Fr
SEED=203600
rng=np.random.Generator(np.random.PCG64(SEED))
low=rng.uniform(0.25,1.75,(4,8)).astype(np.float32)
interior=np.concatenate((np.full((4,1),2.0**30,np.float32),np.full((4,1),-(2.0**30),np.float32),low),axis=1)
order=rng.permutation(10)
x=np.empty((4,12),dtype=np.float32)
x[:,0]=np.float32(2.0**80)
x[:,1:-1]=interior[:,order]
x[:,-1]=np.float32(-(2.0**80))
exact=[]
for r in range(4):
    exact.append(float(sum((Fr(float(v)) for v in x[r]),Fr(0))))
f64=[]
for r in range(4):
    acc=np.float64(0.0)
    for j in range(12):
        acc=np.float64(acc+np.float64(x[r,j]))
    f64.append(float(acc))
res={"exact_fraction_target":exact,"float64_sequential":f64,
     "all_nonzero":all(v!=0.0 for v in exact),
     "float64_matches_exact":f64==exact,
     "target_norm":float(np.linalg.norm(np.array(exact)))}
print(json.dumps(res))