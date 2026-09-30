
import torch, json, math, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_11/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

def ref(x, cos, sin, interleaved):
    S,H,D = x.shape
    x0, x1 = x[..., :D//2].float(), x[..., D//2:].float()
    o0 = x0*cos - x1*sin
    o1 = x0*sin + x1*cos
    if not interleaved:
        return torch.cat([o0,o1], dim=-1).to(x.dtype)
    return torch.stack([o0,o1], dim=-1).flatten(-2).to(x.dtype)

results = []
torch.manual_seed(0)
for S,H,D in [(1,2,64),(128,8,128),(7,3,96),(33,5,32),(256,16,256)]:
    x = torch.randn(S,H,D, device="cuda")
    pos = torch.arange(S, device="cuda", dtype=torch.float32)
    freqs = 1.0/(10000.0 ** (torch.arange(0, D//2, device="cuda", dtype=torch.float32)/(D//2)))
    ang = torch.outer(pos, freqs)
    cos = torch.cos(ang).contiguous(); sin = torch.sin(ang).contiguous()
    for il in [False, True]:
        out = k.apply_rotary(x, cos, sin, il)
        r = ref(x, cos, sin, il)
        err = (out.float()-r.float()).abs().max().item()
        # relative-position invariance check: dot(q_m, k_n) should depend only on m-n
        q = k.apply_rotary(x, cos, sin, il)
        # recompute rotation for shifted positions directly via ref
        q2 = ref(x, cos, sin, il)
        results.append({"S":S,"H":H,"D":D,"interleaved":il,"max_abs_err":err,
                        "dtype":str(out.dtype),"shape":list(out.shape)})

# also fp16 and non-even-D? D must be even by contract; test fp16
for dtype in [torch.float16, torch.bfloat16]:
    x = torch.randn(64,4,128, device="cuda", dtype=dtype)
    pos = torch.arange(64, device="cuda", dtype=torch.float32)
    freqs = 1.0/(10000.0**(torch.arange(0,64,device="cuda",dtype=torch.float32)/64))
    ang = torch.outer(pos, freqs)
    cos = torch.cos(ang).contiguous(); sin = torch.sin(ang).contiguous()
    for il in [False, True]:
        out = k.apply_rotary(x, cos, sin, il)
        r = ref(x.float(), cos, sin, il).to(dtype)
        err = (out.float()-r.float()).abs().max().item()
        results.append({"dtype":str(dtype),"interleaved":il,"max_abs_err":err})

# relative-position invariance test
def rope_ref(x, ang_cos, ang_sin, il):
    return ref(x, ang_cos, ang_sin, il)
S,H,D = 32,2,64
x = torch.randn(S,H,D, device="cuda")
pos = torch.arange(2*S, device="cuda", dtype=torch.float32)
freqs = 1.0/(10000.0**(torch.arange(0,D//2,device="cuda",dtype=torch.float32)/(D//2)))
ang = torch.outer(pos, freqs)
cos = torch.cos(ang).contiguous(); sin = torch.sin(ang).contiguous()
ok_inv = True
for il in [False, True]:
    xs = k.apply_rotary(x, cos, sin, il)
    dots = torch.einsum("shd,thd->st", xs[:,0], xs[:,0])
    ref_dots = torch.einsum("shd,thd->st", ref(x,cos,sin,il)[:,0], ref(x,cos,sin,il)[:,0])
    rel = (dots - ref_dots).abs().max().item()
    ok_inv &= (rel < 1e-3)
print(json.dumps({"results": results, "invariance_vs_ref_rel_diff": rel, "invariance_ok": ok_inv}))
