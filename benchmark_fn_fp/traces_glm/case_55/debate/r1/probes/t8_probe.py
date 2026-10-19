import torch, math, json, importlib.util
spec = importlib.util.spec_from_file_location("kernel", "/root/cases/case_128/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

bits = torch.arange(65536, dtype=torch.int32).to(torch.uint16).view(torch.float16)
vals = bits[torch.isfinite(bits)]  # exclude NaN/Inf (out of scope)
print("sweep size:", vals.numel())
x = vals.cuda().contiguous()
out = k.run(x)
ref = k.reference(x.cpu()).cuda()
err = (out.double() - ref.double()).abs()
maxerr = err.max().item()
i = int(err.argmax())
tgt = torch.tensor([4.0,5.0,6.0,7.0,8.0,10.0,-4.0,-5.0,-6.0,-7.0,-8.0,-10.0], dtype=torch.float16).cuda()
o2 = k.run(tgt); r2 = k.reference(tgt.cpu()).cuda()
terr = (o2.double()-r2.double()).abs()
result = {
 "max_abs_err_full_sweep": maxerr,
 "worst_input_fp16": float(x[i].item()),
 "tolerance": 1e-6,
 "num_exceed_tol": int((err > 1e-6).sum().item()),
 "num_finite_swept": vals.numel(),
 "targeted_inputs": [float(v) for v in tgt.cpu()],
 "targeted_errors": [float(v) for v in terr.cpu()],
 "targeted_max_err": terr.max().item(),
 "verdict_metric": "max absolute error vs FP64 erfc reference over all finite FP16 values"
}
print("FINAL_JSON:", json.dumps(result))