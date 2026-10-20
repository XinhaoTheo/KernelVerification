import torch, json
import kernel as K

base = (torch.rand(3 + 1000, generator=torch.Generator().manual_seed(1)) * 20 - 10).to(torch.float16).cuda()
x = base[3:3+1000]  # contiguous view, storage_offset=3
assert x.is_contiguous() and x.storage_offset() == 3
out = K.run(x)
ref = torch.expm1(x.double().cpu())
err = (out.double().cpu() - ref).abs()
tol = 1e-7 + 1e-6 * ref.abs()
print(json.dumps({
    "metric": "abs_err / contract_tol for contiguous view with storage_offset=3",
    "n": x.numel(), "n_fail": int((err > tol).sum().item()),
    "max_abs_err": float(err.max().item()),
    "max_ratio": float((err/tol).max().item()),
    "storage_offset": x.storage_offset(),
}))
