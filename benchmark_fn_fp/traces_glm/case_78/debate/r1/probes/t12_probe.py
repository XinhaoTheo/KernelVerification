import numpy as np, json
SEED=203795
rng = np.random.Generator(np.random.PCG64(SEED))
low = rng.uniform(0.25,1.75,(4,8)).astype(np.float32)
interior = np.concatenate((np.full((4,1),2.0**30,np.float32),np.full((4,1),-(2.0**30),np.float32),low),axis=1)
order = rng.permutation(10)
x = np.empty((4,12),dtype=np.float32)
x[:,0]=np.float32(2.0**80); x[:,1:-1]=interior[:,order]; x[:,-1]=np.float32(-(2.0**80))
pos30 = int(np.where(order==0)[0][0]); posm30 = int(np.where(order==1)[0][0])
lo,hi = min(pos30,posm30), max(pos30,posm30)
n_between = hi-lo-1
small_between = [float(x[r,1+j]) for r in range(4) for j in range(lo+1,hi)]
# simulate plain fp32 correction sum of the ten interior values in column order
def fp32(v): return float(np.float32(v))
sim=[]
for r in range(4):
    corr = np.float32(0.0)
    for c in range(1,11):
        v = x[r,c]
        # Neumaier: total=2^80 so lost = value exactly
        lost = np.float32(v)
        corr = np.float32(corr + lost)
    sim.append(float(corr))
# simulate full kernel reduction incl. terminal cancellation
out=[]
for r in range(4):
    total = np.float32(2.0**80); corr = np.float32(0.0)
    for c in range(12):
        v = np.float32(x[r,c])
        upd = np.float32(total+v)
        if abs(total) >= abs(v):
            lost = np.float32(np.float32(total-upd)+v)
        else:
            lost = np.float32(np.float32(v-upd)+total)
        corr = np.float32(corr+lost); total = upd
    out.append(float(np.float32(total+corr)))
print(json.dumps({"order":order.tolist(),"pos30_col":pos30+1,"posm30_col":posm30+1,"n_small_between":n_between,"small_between":small_between,"sim_correction_sum":sim,"sim_kernel_out":out}))