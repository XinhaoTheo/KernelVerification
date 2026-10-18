
import torch, json, sys
sys.path.insert(0, "/root/cases/case_147")
import kernel

x = torch.tensor([65504.0, -65504.0, 65503.0, -65503.0, 60000.0, -60000.0],
                 dtype=torch.float16).to("cuda")
out = kernel.run(x)
ref = torch.atan(x.to(torch.float64))
err = (out.to(torch.float64) - ref).abs()
print(json.dumps({
    "inputs": [float(v) for v in x.to(torch.float64).tolist()],
    "outputs": [float(v) for v in out.to(torch.float64).tolist()],
    "errs": [float(v) for v in err.tolist()],
    "max_abs_err_extrema": float(err.max().item()),
    "tol": 1e-6,
}))
