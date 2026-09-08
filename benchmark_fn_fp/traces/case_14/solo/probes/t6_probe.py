
import torch, json, importlib.util, sys
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_14/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def ref(x):
    absmax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    scale = absmax/127
    q = torch.clamp(torch.round(x/scale), -127, 127)
    return q*scale, scale

torch.manual_seed(0)
res={}
for name, x in [("randn_128x256", torch.randn(128,256, device='cuda')*1.0),
                ("large_row", torch.randn(4,64, device='cuda')*10.0),
                ("small_row", torch.randn(4,64, device='cuda')*0.01)]:
    out = m.quant_dequant_int8(x)
    r, sc = ref(x)
    err = (out-r).abs().max().item()
    rel = ((out-r).abs()/ (r.abs()+1e-9)).max().item()
    # clipping check: per contract no value clipped by its own scale -> |out| <= |x| approx
    clipped = (out.abs() < x.abs()-1e-3) & (x.abs() > 4.0)
    res[name] = dict(max_abs_err=err, max_rel_err=rel,
                     kernel_scale_first=float((out[0].abs().max()/127).item()),
                     ref_scale_first=float(sc[0].item()),
                     n_clipped=int(clipped.sum().item()),
                     max_absx=float(x.abs().max().item()),
                     kernel_out_max=float(out.abs().max().item()),
                     ref_out_max=float(r.abs().max().item()))
print(json.dumps(res, indent=2))
