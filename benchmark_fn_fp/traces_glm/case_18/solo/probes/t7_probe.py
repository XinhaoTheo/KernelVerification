import json, torch
import importlib.util
spec = importlib.util.spec_from_file_location("kernel", "/root/cases/case_18/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

res = {}
for n_cols in [128, 129, 200, 300, 1000]:
    x = torch.randn(4, n_cols, device="cuda", dtype=torch.float32)
    y = m.softmax(x)
    ref = torch.softmax(x, dim=-1)
    sums = y.sum(dim=-1)
    ref_sums = ref.sum(dim=-1)
    err = (y - ref).abs().max().item()
    res[n_cols] = {"kernel_row_sums": sums.tolist(), "max_abs_err_vs_ref": err,
                   "min": y.min().item(), "max": y.max().item()}

# dominant-entry case too
x = torch.randn(4, 130, device="cuda"); x[:, 0] += 20.0
y = m.softmax(x); ref = torch.softmax(x, dim=-1)
res["dominant_130"] = {"kernel_row_sums": y.sum(-1).tolist(),
                       "max_abs_err_vs_ref": (y-ref).abs().max().item(),
                       "tail_err": (y[:,1:]-ref[:,1:]).abs().max().item()}
print(json.dumps({"metric": "row sums + elementwise max abs error vs torch.softmax; row sums must equal 1 per contract", "result": res}))