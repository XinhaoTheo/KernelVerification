import torch, sys, json
sys.path.insert(0, "/root/cases/case_122")
import kernel as K

errs = {}
for (h, w, steps) in [(1,1,4),(1,129,8),(129,1,4),(129,129,8),(2,3,4),(127,128,8),(3,3,8)]:
    g = torch.Generator(device="cpu").manual_seed(h*1000+w)
    x = (2.0*torch.rand((h,w), generator=g)-1.0).half().cuda()
    r = K.run(x, steps)
    ref = K.reference(x, steps)
    errs[f"{h}x{w}_s{steps}"] = (r-ref).abs().max().item()
# preservation & output properties
x = K.make_inputs()[0]
xc = x.clone()
r = K.run(x, 4)
ok_pres = torch.equal(x, xc)
ok_props = (r.dtype==torch.float32 and r.is_cuda and r.shape==x.shape and torch.isfinite(r).all().item())
print("preserve_x:", ok_pres, "output_props:", ok_props)
print("JSON:" + json.dumps({"metric":"max abs error vs FP64 reference, domain sweep","tolerance":2e-5,"errors":errs,"preserve_x":ok_pres,"output_props":bool(ok_props)}))
