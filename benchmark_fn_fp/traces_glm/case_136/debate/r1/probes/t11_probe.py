import sys, torch, json, numpy as np
sys.path.insert(0, "/root/cases/case_136")
import kernel

# underflow region |x| >= 88 including extrema
vals = [65504.0, -65504.0, 100.0, -100.0, 88.0, -88.0, 89.0, -89.0, 90.0, -90.0, 2048.0, -2048.0, 4096.0, 10000.0, -10000.0]
xs = torch.tensor(vals, dtype=torch.float32).to(torch.float16).cuda()
out = kernel.run(xs)
ref = kernel.reference(xs.cpu()).cuda()
err = (out.double() - ref.double()).abs()
print(json.dumps({
    "metric": "abs error in underflow/extrema region vs FP64 reference",
    "values": [float(v) for v in xs.double().cpu()],
    "outputs": [float(v) for v in out.double().cpu()],
    "errs": [float(v) for v in err.cpu()],
    "max_err": float(err.max()),
    "tolerance": 1e-6,
    "n_violations": int((err > 1e-6).sum()),
    "pos_65504_out": float(out[0].item()),
    "neg_65504_out": float(out[1].item()),
}))