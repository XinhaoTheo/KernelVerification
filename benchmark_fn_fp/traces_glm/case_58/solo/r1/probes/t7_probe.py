import torch, kernel, json
bits = torch.arange(0, 65536, dtype=torch.int16)
x_all = bits.view(torch.float16)
finite = torch.isfinite(x_all)
x = x_all[finite].contiguous()
out = kernel.run(x.cuda())
ref = kernel.reference(x.cuda())
err = (out.double() - ref).abs()
tol = 1e-6
bad = err > tol
worst_i = int(err.argmax())
print("worst x:", float(x[worst_i]), "out:", float(out[worst_i]), "ref:", float(ref[worst_i]))
if bad.any():
    bidx = torch.where(bad)[0].cpu()
    xb = x.cpu()[bidx]
    print("bad x range:", float(xb.min()), float(xb.max()), "n_bad:", int(bad.sum()))
    print("sample bad x:", xb[:8].tolist())
print(json.dumps({"n_tested": int(x.numel()), "n_over_tol": int(bad.sum()), "max_err": float(err.max()), "tol": 1e-6}))