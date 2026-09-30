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
def res(x): return float(np.sum(cd*np.maximum(Wd@np.asarray(x,np.float64)-bd,0.0)))
signs=np.array(list(itertools.product([-1.0,1.0],repeat=12)),dtype=np.float64)
vals=np.array([res(x) for x in signs])
imax=int(np.argmax(vals)); vmax=vals[imax]; vx=signs[imax]
# near-vertex: flip 1 or 2 signs of the best vertex
near={}
for k in (1,2):
    ms=[]
    for comb in itertools.combinations(range(12),k):
        y=vx.copy()
        for i in comb: y[i]=-y[i]
        ms.append(res(y))
    near[f"flip_{k}_min_max"]=[float(min(ms)),float(max(ms))]
print(json.dumps({"seed":SEED,"box_vertex_max_residual":float(vmax),
 "best_vertex":vx.tolist(),"num_vertices_over_1.0":int((vals>1.0).sum()),
 "num_vertices_over_0.9":int((vals>0.9).sum()),
 "near_vertex_residual_ranges":near,"tolerance":1.0,
 "is_borderline_0.9_1.1":bool(0.9<=vmax<=1.1),
 "max_exceeds_1.0":bool(vmax>1.0)}))