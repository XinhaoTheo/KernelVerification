import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_163/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
# every finite bf16 value in [-10,10]
bits = torch.arange(0, 65536, dtype=torch.int32)
vals = bits.to(torch.uint16).view(torch.bfloat16)
vals = vals[torch.isfinite(vals) & (vals.abs() <= 10)]
x = vals.to("cuda")
out = k.run(x)
ref = torch.exp2(x.to(torch.float64))
err = (out.to(torch.float64) - ref).abs()
rel = err / ref
i = int(rel.argmax())
print(json.dumps({
 "num_legal_values": int(x.numel()),
 "max_relative_error": float(rel[i]),
 "worst_x": float(x[i]),
 "worst_out": float(out[i]),
 "worst_ref": float(ref[i]),
 "num_violations_1e-6": int((rel > 1e-6).sum()),
 "num_violations_1e-5": int((rel > 1e-5).sum()),
 "num_violations_1e-4": int((rel > 1e-4).sum()),
}))
