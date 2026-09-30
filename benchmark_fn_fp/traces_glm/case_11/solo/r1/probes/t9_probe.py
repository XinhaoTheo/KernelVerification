import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_11/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

def ref(x, cos, sin, interleaved):
    S,H,D = x.shape
    c = cos.float().unsqueeze(1); s = sin.float().unsqueeze(1)  # (S,1,D/2)
    x0, x1 = x[..., :D//2].float(), x[..., D//2:].float()
    o0 = x0*c - x1*s
    o1 = x0*s + x1*c
    if not interleaved:
        return torch.cat([o0,o1], dim=-1).to(x.dtype)
    return torch.stack([o0,o1], dim=-1).flatten(-2).to(x.dtype)

def mk(S,D):
    pos = torch.arange(S, device="cuda", dtype=torch.float32)
    freqs = 1.0/(10000.0**(torch.arange(0,D//2,device="cuda",dtype=torch.float32)/(D//2)))
    ang = torch.outer(pos, freqs)
    return torch.cos(ang).contiguous(), torch.sin(ang).contiguous()

results = []
torch.manual_seed(0)
for S,H,D in [(1,2,64),(128,8,128),(7,3,96),(33,5,32),(256,16,256)]:
    x = torch.randn(S,H,D, device="cuda")
    cos, sin = mk(S,D)
    for il in [False, True]:
        out = k.apply_rotary(x, cos, sin, il)
        r = ref(x, cos, sin, il)
        err = (out.float()-r.float()).abs().max().item()
        results.append({"S":S,"H":H,"D":D,"interleaved":il,"max_abs_err":err})
print("PART1", json.dumps(results), flush=True)

for dtype in [torch.float16, torch.bfloat16]:
    x = torch.randn(64,4,128, device="cuda", dtype=dtype)
    cos, sin = mk(64,128)
    for il in [False, True]:
        out = k.apply_rotary(x, cos, sin, il)
        r = ref(x.float(), cos, sin, il).to(dtype)
        err = (out.float()-r.float()).abs().max().item()
        results.append({"dtype":str(dtype),"interleaved":il,"max_abs_err":err})
print("PART2", json.dumps(results[-4:]), flush=True)

# relative-position invariance: dot(q_s, k_t) depends only on s-t
S,H,D = 64,2,64
x = torch.randn(S,H,D, device="cuda")
cos, sin = mk(S,D)
inv = {}
for il in [False, True]:
    xr = k.apply_rotary(x, cos, sin, il)
    dots = torch.einsum("sd,td->st", xr[:,0], xr[:,0])
    diffs = [ (torch.diag(dots, s) - dots[0, s]).abs().max().item() for s in range(1, S) ]
    inv["interleaved" if il else "split"] = max(diffs)
print("PART3", json.dumps(inv))
print(json.dumps({"results": results, "invariance_max_dev": inv}))
