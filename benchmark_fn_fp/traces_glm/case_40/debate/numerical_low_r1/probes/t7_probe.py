import json, sys, importlib.util
import torch

spec = importlib.util.spec_from_file_location("k", "/root/numerical_cases/case_e/kernel.py")
k = importlib.util.module_from_spec(spec)
spec.loader.exec_module(k)

matrix, initial, drive = k.make_inputs("cuda")

# float64 reference on the float32 inputs
m = matrix.cpu().numpy().astype("float64")
s = initial.cpu().numpy().astype("float64")
d = drive.cpu().numpy().astype("float64")
for t in range(64):
    s = m @ s + d[t]
ref = s

out = k.run(matrix, initial, drive).cpu().numpy().astype("float64")
denom = max(float((ref**2).sum()**0.5), 0.001*4.0)
err = float(((out-ref)**2).sum()**0.5) / denom
print(json.dumps({
  "metric": "relative L2 error vs float64 reference",
  "rel_l2_error": err,
  "tolerance": 0.002,
  "ref_norm": float((ref**2).sum()**0.5),
  "out_norm": float((out**2).sum()**0.5),
  "finite": bool((out == out).all() and (abs(out) < float("inf")).all()),
  "exceeds_tolerance": err > 0.002,
}))
