import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_134/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
# signed zeros and all subnormal FP16 patterns (exponent 0, mantissa != 0)
m = torch.arange(1024, dtype=torch.int32)
sub_bits = (m & 0x03FF).to(torch.uint16).view(torch.float16)
neg_sub = sub_bits.clone(); neg_sub_bits_view = neg_sub.view(torch.int16); # sign bit set
neg = (sub_bits.view(torch.int16) | torch.int16(-0x8000)).view(torch.float16)
xs = [torch.tensor([0.0], dtype=torch.float16), torch.tensor([-0.0], dtype=torch.float16),
      torch.tensor([6.104e-5], dtype=torch.float16), torch.tensor([-6.104e-5], dtype=torch.float16),
      sub_bits, neg]
x = torch.cat(xs).cuda()
out = k.run(x)
ref = k.reference(x)
err = (out.double()-ref).abs()
i = int(err.argmax())
res = {"n": x.numel(), "max_abs_err": float(err[i]), "tolerance": 1e-6,
       "num_exceed": int((err>1e-6).sum()),
       "neg_zero_out": float(out[1].item()), "neg_zero_ref": float(ref[1].item()),
       "neg_zero_signbit_out": bin(torch.tensor(out[1].item()).view(torch.int32).item() & 0x80000000 != 0),
       "min_subnormal_err": float(err[len(xs)-1])}
print(json.dumps(res))