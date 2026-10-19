import sys, json
sys.path.insert(0, "/root/cases/case_147")
import torch, numpy as np
from kernel import run

bits = np.arange(0, 65536, dtype=np.uint16)
vals = bits.view(np.float16)
vals = vals[np.isfinite(vals)]
mag = np.abs(vals.astype(np.float64))

upper = mag > 2.414213562373095
b_up = np.abs(mag - 2.414213562373095) < 0.01
b_mid = np.abs(mag - 0.66) < 0.01
focus = upper | b_up | b_mid
fvals = vals[focus]
fmag = np.abs(fvals.astype(np.float64))

x = torch.from_numpy(fvals.copy()).to("cuda")
out = run(x).cpu().numpy()
ref = torch.atan(torch.from_numpy(fvals.astype(np.float64))).numpy()
err = np.abs(out.astype(np.float64) - ref)
f_upper = fmag > 2.414213562373095
f_bup = np.abs(fmag - 2.414213562373095) < 0.01
grp = np.where(f_upper, "upper", np.where(f_bup, "near_2.414", "near_0.66"))

def stats(m):
    if not m.any(): return None
    i = int(np.argmax(err[m]))
    return {"count": int(m.sum()), "max_abs_err": float(err[m].max()),
            "worst_input": float(fvals[m][i]), "worst_out": float(out[m][i]),
            "worst_ref": float(ref[m][i])}

big = fmag > 100  # very large magnitudes incl 65504
res = {
  "n_focus": int(fvals.size), "tolerance": 1e-6,
  "num_violations": int((err > 1e-6).sum()),
  "max_abs_err": float(err.max()),
  "worst_input": float(fvals[int(np.argmax(err))]),
  "worst_out": float(out[int(np.argmax(err))]),
  "worst_ref": float(ref[int(np.argmax(err))]),
  "fp16_grid_around_2.414": sorted(set(float(v) for v in fvals[f_bup & (fmag <= 2.5)])),
  "fp16_grid_around_0.66": sorted(set(float(v) for v in fvals[np.abs(fmag-0.66) < 0.01])),
  "very_large_magnitude_stats": stats(big),
  "by_group": {g: stats(grp == g) for g in ["upper", "near_2.414", "near_0.66"]},
}
print(json.dumps(res))
