
import json, importlib.util, torch
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_18/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
torch.manual_seed(0)
res = {}
for n_cols in [1, 8, 64, 127]:
    x = torch.randn(4, n_cols, device="cuda", dtype=torch.float32)
    y = k.softmax(x)   # default block_size=128
    ref = torch.softmax(x, dim=-1)
    res[str(n_cols)] = {
        "n_inf": int(torch.isinf(y).sum()),
        "n_nan": int(torch.isnan(y).sum()),
        "n_elems": int(y.numel()),
        "row_sums": [float(v) for v in y.sum(-1)],
        "sample_row0": [float(v) for v in y[0][:8]],
        "ref_row0": [float(v) for v in ref[0][:8]],
    }
print(json.dumps(res))
