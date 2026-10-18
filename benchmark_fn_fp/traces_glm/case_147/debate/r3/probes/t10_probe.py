import sys, json
sys.path.insert(0, "/root/cases/case_147")
import torch, numpy as np
from kernel import run

bits = np.arange(0, 65536, dtype=np.uint16)
fp16 = bits.view(np.float16)
vals = fp16[np.isfinite(fp16)]
n = vals.size

x = torch.from_numpy(vals.copy()).to("cuda")
out = run(x).cpu().numpy()
ref = torch.atan(torch.from_numpy(vals.astype(np.float64))).numpy()
err = np.abs(out.astype(np.float64) - ref)

mag = np.abs(vals.astype(np.float64))
branch = np.where(mag > 2.414213562373095, "upper",
         np.where(mag > 0.66, "middle", "small"))

def stats(m):
    if not m.any(): return None
    i = int(np.argmax(err[m]))
    return {"count": int(m.sum()), "max_abs_err": float(err[m].max()),
            "worst_input": float(vals[m][i]), "worst_out": float(out[m][i]),
            "worst_ref": float(ref[m][i])}

res = {
  "n_tested": int(n), "tolerance": 1e-6,
  "num_violations": int((err > 1e-6).sum()),
  "global_max_abs_err": float(err.max()),
  "worst_input": float(vals[int(np.argmax(err))]),
  "worst_out": float(out[int(np.argmax(err))]),
  "worst_ref": float(ref[int(np.argmax(err))]),
  "by_branch": {b: stats(branch == b) for b in ["small", "middle", "upper"]},
}
print(json.dumps(res))
