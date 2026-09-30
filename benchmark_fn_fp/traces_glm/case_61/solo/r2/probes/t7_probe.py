
import numpy as np, torch, sys
sys.path.insert(0, "/root/numerical_cases/case_z")
import kernel

query, anchors, values = kernel.make_inputs("cuda")
out = kernel.run(query, anchors, values)
torch.cuda.synchronize()

# float64 reference from stored float32 values
q = query.cpu().numpy().astype(np.float64)
a = anchors.cpu().numpy().astype(np.float64)
v = values.cpu().numpy().astype(np.float64)
d = ((a - q)**2).sum(axis=1)
w = np.exp(-16.0 * d)
ref = (w * v).sum() / w.sum()

o = out.item()
err = abs(o - ref) / max(abs(ref), 0.05)
print("output", o, "reference", ref, "abs_diff", abs(o-ref))
print("distances", d.min(), d.max(), "weights range", w.min(), w.max())
import json
print(json.dumps({"output": o, "reference": ref, "error": err, "finite": bool(np.isfinite(o))}))
