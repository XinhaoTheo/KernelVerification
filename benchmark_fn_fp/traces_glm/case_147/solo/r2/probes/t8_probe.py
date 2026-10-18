import sys, json, torch
sys.path.insert(0, "/root/cases/case_147")
import kernel as K

bits = torch.arange(65536, dtype=torch.int32)
bits[(bits & 0x7C00) == 0x7C00] = -1  # NaN/Inf excluded (exp all ones)
bits = bits[bits >= 0].to(torch.int16)
x = bits.view(torch.float16).cuda()
N = x.numel()
errs = []
for chunk_start in range(0, N, 60000):
    c = x[chunk_start:chunk_start+60000].contiguous()
    out = K.run(c)
    r = torch.atan(c.to(torch.float64))
    errs.append((out.to(torch.float64) - r).abs())
err = torch.cat(errs)
viol = (err > 1e-6).nonzero()
info = {"N": int(N), "max_abs_err": float(err.max()),
  "n_violations_1e-6": int((err > 1e-6).sum())}
if info["n_violations_1e-6"] > 0:
    i = int(err.argmax())
    info["worst_x"] = float(x[i].item())
    info["worst_out"] = float(torch.cat(errs).sub(0).sum().item()) if False else None
    info["err_at_worst"] = float(err[i].item())
    # also check signed zeros/subnormals/extrema specifically
edges = torch.tensor([0.0, -0.0, 6.104e-5, -6.104e-5, 5.96e-8, 65504.0, -65504.0],
                     dtype=torch.float16).cuda()
eout = K.run(edges.contiguous())
eref = torch.atan(edges.to(torch.float64))
eerr = (eout.to(torch.float64) - eref).abs()
info["edge_max_err"] = float(eerr.max())
info["edge_values"] = [float(v) for v in edges.cpu()]
info["edge_outs"] = [float(v) for v in eout.cpu()]
info["edge_errs"] = [float(v) for v in eerr.cpu()]
print(json.dumps(info))