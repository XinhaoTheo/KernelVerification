import torch, json
import kernel as K

# Exhaustive scan of every finite FP16 value in [-10,10]
bits = torch.arange(65536, dtype=torch.int32)
vals = bits.to(torch.int16).view(torch.float16).float()
legal = torch.isfinite(vals) & (vals.abs() <= 10.0)
xs = vals[legal].to(torch.float16)
xs, _ = torch.unique(xs, sorted=True).cpu().to(torch.float16), None
xs = xs.cuda()
out = K.run(xs)
ref = torch.expm1(xs.double().cpu()).cuda()
err = (out.double() - ref).abs()
tol = 1e-7 + 1e-6 * ref.abs()
ratio = err / tol
i = int(torch.argmax(ratio).item())
n_fail = int((err > tol).sum().item())
print(json.dumps({
    "metric": "abs_err/contract_tol vs FP64 expm1 over ALL finite FP16 values in [-10,10]",
    "n_tested": xs.numel(), "n_fail": n_fail,
    "worst_x": float(xs[i].item()), "worst_abs_err": float(err[i].item()),
    "tol_at_worst": float(tol[i].item()), "worst_ratio": float(ratio[i].item()),
    "top5": [[float(xs[j].item()), float(err[j].item()), float(ratio[j].item())]
             for j in torch.topk(ratio, 5).indices.tolist()],
}))
