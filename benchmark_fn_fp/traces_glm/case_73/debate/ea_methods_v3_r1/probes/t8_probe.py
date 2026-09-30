import numpy as np, itertools, json
SEED=194003
rng=np.random.Generator(np.random.PCG64(SEED))
center=rng.choice(np.asarray([-1,1]),size=12)
fp=rng.uniform(0.05,0.4)
flips=np.where(rng.uniform(size=(6,12))<fp,-1,1)
mag=rng.choice(np.asarray([0.5,1.0]),size=(6,12))
W=(center[None,:]*flips*mag).astype(np.float32)
b=(0.75*np.abs(W).sum(axis=1)).astype(np.float32)
c=np.full(6,0.25,dtype=np.float32)
Wd=W.astype(np.float64); bd=b.astype(np.float64); cd=c.astype(np.float64)
best=-1; best_x=None
signs=np.array(list(itertools.product([-1.0,1.0],repeat=12)))
for x in signs:
    r=float(np.sum(cd*np.maximum(Wd@x-bd,0.0)))
    if r>best: best=r; best_x=x
# aligned with W rows
aligned_r=[]
for j in range(6):
    x=np.sign(W[j]).astype(np.float64)
    aligned_r.append(float(np.sum(cd*np.maximum(Wd@x-bd,0.0))))
print(json.dumps({"seed":SEED,"vertex_enumeration_max_residual":best,
 "best_vertex":best_x.tolist(),"aligned_with_W_rows_residuals":aligned_r,
 "tolerance":1.0,"exceeds_tolerance":bool(best>1.0),
 "S_j_abs_row_sums":np.abs(W).sum(axis=1).tolist(),
 "b":b.tolist()}))