
import json, importlib.util, traceback
import numpy as np, torch

res = {"exception": None}
try:
    spec = importlib.util.spec_from_file_location("k2", "/root/cases/case_a/kernel.py")
    k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
    x, A, B = k.make_inputs("cuda")
    out = k.run(x, A, B)
    torch.cuda.synchronize()
    o = out.detach().cpu().numpy()
    An = A.detach().cpu().numpy(); Bn = B.detach().cpu().numpy()
    res.update({
      "ran": True,
      "shape": list(o.shape), "dtype": str(out.dtype),
      "n_nan": int(np.isnan(o).sum()), "n_inf": int(np.isinf(o).sum()),
      "all_finite": bool(np.isfinite(o).all()),
      "min_rowmax_A": float(np.abs(An).max(axis=1).min()),
      "min_rowmax_B": float(np.abs(Bn).max(axis=1).min()),
      "zero_rows_A": int((np.abs(An).max(axis=1)==0).sum()),
      "zero_rows_B": int((np.abs(Bn).max(axis=1)==0).sum()),
      "out_min": float(o.min()), "out_max": float(o.max()),
      "triton_version": __import__("triton").__version__,
      "gpu": torch.cuda.get_device_name(0),
    })
except Exception as e:
    res.update({"ran": False, "exception": repr(e), "tb": traceback.format_exc()[-800:]})
print(json.dumps(res))
