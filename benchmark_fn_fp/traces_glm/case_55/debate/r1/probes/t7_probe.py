import torch, math, json, importlib.util
spec = importlib.util.spec_from_file_location("kernel", "/root/cases/case_128/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

vals = torch.arange(65536, dtype=torch.int32).to(torch.uint16).view(torch.float16)
x = vals.cuda().contiguous()
print("N:", x.numel(), "finite:", torch.isfinite(x).all().item())

out = k.run(x)
ref = k.reference(x.cpu()).cuda()
err = (out.double() - ref.double()).abs()
maxerr = err.max().item()
i = int(err.argmax())
targeted = torch.tensor([4.0,5.0,6.0,7.0,8.0,10.0,-4.0,-5.0,-6.0,-7.0,-8.0,-10.0], dtype=torch.float16).cuda()
o2 = k.run(targeted); r2 = k.reference(targeted.cpu()).cuda()
terr = (o2.double()-r2.double()).abs()
json.dump({
 "max_abs_err_full_sweep": maxerr,
 "worst_input_fp16": float(x[i].item()),
 "worst_input_repr": hex(int(torch.tensor([x[i]]).view(torch.int16).item())),
 "tolerance": 1e-6,
 "num_exceed_tol": int((err > 1e-6).sum().item()),
 "targeted_inputs": [float(v) for v in targeted.cpu()],
 "targeted_errors": [float(v) for v in terr.cpu()],
 "targeted_max_err": terr.max().item(),
 "verdict_metric": "max absolute error vs FP64 erfc reference over all 65536 finite FP16 values"
}, fp := open("/tmp/probe_c1.json","w")); fp.close()
print(open("/tmp/probe_c1.json").read())
print("FINAL_JSON:", open("/tmp/probe_c1.json").read())