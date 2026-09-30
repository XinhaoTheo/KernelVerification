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
f32=np.float32
sim=[]; trace=[]
for r in range(4):
    total=f32(0.0); corr=f32(0.0)
    for j in range(12):
        v=f32(x[r,j])
        updated=f32(total+v)
        if abs(total)>=abs(v):
            lost=f32(f32(total-updated)+v)
        else:
            lost=f32(f32(v-updated)+total)
        corr=f32(corr+lost)
        total=updated
        trace.append(float(lost))
    sim.append(float(f32(total+corr)))
target=[float(sum((Fr(v) for v in x[r]),Fr(0))) for r in range(4)]
rel=float(np.linalg.norm(np.array(sim)-np.array(target))/max(np.linalg.norm(np.array(target)),1e-12))
# show how many small-magnitude lost terms were absorbed when corr was ~2^30-scale
res={"simulated_output":sim,"exact_target":target,"relative_l2_error":rel,
     "tolerance":1e-5,"passed":bool(rel<=1e-5),
     "lost_terms_first_row":trace[:12],
     "sample_recorded_output":[0.7218434810638428,1.2222856283187866,1.1249507665634155,1.3434942960739136]}
print(json.dumps(res))