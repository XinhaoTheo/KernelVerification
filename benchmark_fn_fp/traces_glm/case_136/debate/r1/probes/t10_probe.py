import sys, torch, json, numpy as np
sys.path.insert(0, "/root/cases/case_136")
import kernel

vals = [0.0, -0.0, 5.96e-08, -5.96e-08, 2.98e-08, -2.98e-08, 6.1e-05, -6.1e-05, 1.19e-07]
xs = torch.tensor(vals, dtype=torch.float32).to(torch.float16).cuda()
out = kernel.run(xs)
ref = kernel.reference(xs.cpu()).cuda()
err = (out.double() - ref.double()).abs()
print(json.dumps({
    "metric": "abs error at signed zeros and subnormals vs FP64 reference",
    "values": [float(v) for v in xs.double().cpu()],
    "outputs": [float(v) for v in out.double().cpu()],
    "refs": [float(v) for v in ref.double().cpu()],
    "errs": [float(v) for v in err.cpu()],
    "max_err": float(err.max()),
    "tolerance": 1e-6,
    "neg_zero_err_vs_ln2": float((out[1].double() - float(np.log(2))).abs()),
    "n_violations": int((err > 1e-6).sum()),
}))