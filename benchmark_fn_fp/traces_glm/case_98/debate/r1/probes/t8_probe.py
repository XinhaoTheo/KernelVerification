import torch, json, math
import kernel as K

# candidate FP16 gridpoints near +/- ln2/2 (k=0 boundary, worst |r|)
cands = []
for target in [0.346680908203125, -0.346680908203125, 0.346588134765625, -0.346588134765625,
               0.34649658203125, -0.34649658203125, 0.346771240234375, -0.346771240234375,
               0.347, -0.347, 0.35, -0.35, 0.3, -0.3]:
    t = torch.tensor(target, dtype=torch.float16)
    cands.append(t.item())
# also every FP16 gridpoint in [0.30, 0.36] and negatives
vals = set(cands)
v = torch.tensor(0.29, dtype=torch.float16)
import struct
start = torch.tensor(0.30, dtype=torch.float16).view(torch.int16).item()
end = torch.tensor(0.36, dtype=torch.float16).view(torch.int16).item()
for bits in range(start, end+1):
    vals.add(torch.tensor(bits, dtype=torch.int16).view(torch.float16).item())
for bits in range(-end, -start+1):
    vals.add(torch.tensor(bits, dtype=torch.int16).view(torch.float16).item())
xs = torch.tensor(sorted(vals), dtype=torch.float16).cuda()
out = K.run(xs)
ref = torch.expm1(xs.double().cpu()).cuda()
err = (out.double() - ref).abs()
tol = 1e-7 + 1e-6 * ref.abs()
ratio = err / tol
i = int(torch.argmax(ratio).item())
n_fail = int((err > tol).sum().item())
print(json.dumps({
    "metric": "abs_err / contract_tol vs FP64 expm1 over FP16 gridpoints near +/-0.3466",
    "n_tested": xs.numel(), "n_fail": n_fail,
    "worst_x": float(xs[i].item()), "worst_abs_err": float(err[i].item()),
    "tol_at_worst": float(tol[i].item()), "ratio": float(ratio[i].item()),
    "top5": [[float(xs[j].item()), float(err[j].item()), float(ratio[j].item())]
             for j in torch.topk(ratio, 5).indices.tolist()],
}))
