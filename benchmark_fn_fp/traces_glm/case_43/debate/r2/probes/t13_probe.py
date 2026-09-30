
import sys, json
sys.path.insert(0, "/root/numerical_cases/case_h")
import numpy as np, torch
import kernel as K

try:
    values = K.make_inputs("cuda")[0]
    out = K.run(values)
    out_cpu = out.detach().cpu().numpy()
    # faithful fp32 left-to-right emulation from the same tensor
    v = values.detach().cpu().numpy()
    acc = np.zeros(64, dtype=np.float32)
    for j in range(128):
        acc = (acc + v[:, j]).astype(np.float32)
    max_dev = float(np.abs(out_cpu.astype(np.float64) - acc.astype(np.float64)).max())
    res = {
      "shape": list(out.shape), "dtype": str(out.dtype),
      "all_finite": bool(np.isfinite(out_cpu).all()),
      "matches_fp32_sequential_emulation": bool(max_dev == 0.0),
      "max_abs_dev_vs_emulation": max_dev,
    }
except Exception as e:
    res = {"error": f"{type(e).__name__}: {e}"}
print(json.dumps(res))
