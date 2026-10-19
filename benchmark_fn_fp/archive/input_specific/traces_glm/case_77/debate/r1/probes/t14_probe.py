import numpy as np, json, struct
SEED=203600
rng=np.random.Generator(np.random.PCG64(SEED))
low=rng.uniform(0.25,1.75,(4,8)).astype(np.float32)
interior=np.concatenate((np.full((4,1),2.0**30,np.float32),np.full((4,1),-(2.0**30),np.float32),low),axis=1)
order=rng.permutation(10)
x=np.empty((4,12),dtype=np.float32)
x[:,0]=np.float32(2.0**80)
x[:,1:-1]=interior[:,order]
x[:,-1]=np.float32(-(2.0**80))
def bits(v):
    return hex(struct.unpack('<I',struct.pack('<f',v))[0])
res={
 "first_col_values":x[:,0].tolist(),
 "first_col_bits":[bits(v) for v in x[:,0]],
 "last_col_values":x[:,-1].tolist(),
 "last_col_bits":[bits(v) for v in x[:,-1]],
 "all_finite":bool(np.isfinite(x).all()),
 "contains_2e30":bool((x==np.float32(2.0**30)).any() and (x==np.float32(-(2.0**30))).any()),
 "num_small_in_range":int(((x>=0.25)&(x<=1.75)).sum()),
 "max_abs":float(np.abs(x).max()),
 "float32_max":3.4028234663852886e+38,
 "two_pow80":2.0**80,
}
print(json.dumps(res))