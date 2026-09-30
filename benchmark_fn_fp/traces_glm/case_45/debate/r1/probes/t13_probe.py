
import numpy as np, json
rng = np.random.Generator(np.random.PCG64(782406))
x = (64.0 + rng.normal(0.0, 0.125, 128)).astype(np.float32)
x64 = x.astype(np.float64)
mean64 = x64.sum()/128
var64 = ((x64-mean64)**2).sum()/128
ref = (x64-mean64)/np.sqrt(var64+1e-5)

# kernel fp32 sequential mean
total = np.float32(0.0); squares = np.float32(0.0)
for v in x:
    v = np.float32(v)
    total = np.float32(total + v)
    squares = np.float32(squares + np.float32(v*v))
mean32 = np.float32(total/np.float32(128))

mean_abs_err = abs(float(mean32) - float(mean64))
var32 = np.float32(max(np.float32(np.float32(squares/np.float32(128)) - np.float32(mean32*mean32)), np.float32(0.0)))
denom32 = np.float32(np.sqrt(np.float32(var32+np.float32(1e-5))))
ref_denom = np.sqrt(var64+1e-5)
# output offset contribution from mean bias alone (using exact denominator)
offset_per_elem = mean_abs_err/ref_denom
out_mean_only = (x64-mean32)/ref_denom
err_mean_only = np.linalg.norm(out_mean_only-ref)/max(np.linalg.norm(ref),0.001*np.sqrt(128))
print(json.dumps({
  "mean_fp32": float(mean32), "mean_f64": float(mean64),
  "mean_abs_err": mean_abs_err,
  "mean_rel_err_vs_denom": float(mean_abs_err/ref_denom),
  "output_offset_fraction_per_element": float(offset_per_elem),
  "rel_l2_from_mean_bias_alone": float(err_mean_only),
  "denominator": float(ref_denom),
  "tolerance": 0.02
}))
