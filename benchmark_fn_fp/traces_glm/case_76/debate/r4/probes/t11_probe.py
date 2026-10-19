import importlib.util, json, struct
import torch
spec = importlib.util.spec_from_file_location("kernel", "/root/cases/case_149/kernel.py")
kernel = importlib.util.module_from_spec(spec); spec.loader.exec_module(kernel)

edge = [0.0, -0.0, 65504.0, -65504.0, 16.0, 17.0, 17.5, 18.0, 18.5, 19.0, 20.0, 24.0,
        6.1e-5, -6.1e-5, 5.96e-8, -5.96e-8, 1.0, -1.0, 11.09, -11.09]
xe = torch.tensor(edge, dtype=torch.float16).to("cuda")
oute = kernel.run(xe)
refe = kernel.reference(xe)
ok_edge = bool((oute.view(torch.int16) == refe.view(torch.int16)).all())
edge_bits = [{"x": repr(v), "k": hex(int(b) & 0xFFFF), "r": hex(int(rb) & 0xFFFF)}
             for v, b, rb in zip(edge, oute.view(torch.int16).cpu().tolist(), refe.view(torch.int16).cpu().tolist())]

# masked-tail: N=4093 random full-domain
g = torch.Generator().manual_seed(7)
xr = (30.0 * (torch.rand(4093, generator=g) * 2 - 1)).to(torch.float16).to("cuda")
outr = kernel.run(xr); refr = kernel.reference(xr)
tail_ok = bool((outr.view(torch.int16) == refr.view(torch.int16)).all())
tail_mism = int((outr.view(torch.int16) != refr.view(torch.int16)).sum())

# also last-lane check: a tensor where only the final element is nonzero signal
xq = torch.zeros(4093, dtype=torch.float16); xq[-1] = 65504.0
xq = xq.to("cuda")
oq = kernel.run(xq); fq = kernel.reference(xq)
last_ok = bool((oq.view(torch.int16) == fq.view(torch.int16)).all())
print(json.dumps({"claim": "c3", "metric": "exact FP16 bit equality on edge inputs and non-multiple-of-256 N",
  "edge_inputs_ok": ok_edge, "tail_n4093_ok": tail_ok, "tail_mismatch_count": tail_mism,
  "last_lane_ok": last_ok, "edge_bits": edge_bits}))
