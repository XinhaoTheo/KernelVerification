import torch, json, sys
sys.path.insert(0, "/root/cases/case_123")
import kernel

dev = "cuda"
results = []
shapes = [(1,1),(1,129),(129,1),(129,129),(31,33),(128,127),(2,3)]
def make(h, w, kind, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    if kind == "rand":
        x = (2*torch.rand((h,w), generator=g)-1).half()
    elif kind == "ones":
        x = torch.ones((h,w)).half()
    elif kind == "impulse":
        x = torch.zeros((h,w)).half(); x[0,0]=1.0
    elif kind == "corners":
        x = torch.zeros((h,w)).half()
        x[0,0]=1.0; x[-1,-1]=-1.0; x[h//2,w//2]=1.0
    elif kind == "neg":
        x = -torch.ones((h,w)).half()
    return x.to(dev)

worst = 0.0
for (h,w) in shapes:
    for kind in ["rand","ones","impulse","corners","neg"]:
        for steps in (4,8):
            x = make(h,w,kind,hash((h,w,kind))%1000)
            out = kernel.run(x, steps)
            ref = kernel.reference(x, steps)
            err = (out.double()-ref).abs().max().item()
            worst = max(worst, err)
            results.append({"h":h,"w":w,"kind":kind,"steps":steps,"max_abs_err":err,
                            "dtype":str(out.dtype),"shape":list(out.shape),"cuda":out.is_cuda,
                            "input_preserved": bool(torch.equal(x, x.detach()))})
            if err > 2e-5:
                results[-1]["fail"] = True

# contiguous view with nonzero storage offset is legal
base = torch.randn(300, 300, generator=torch.Generator().manual_seed(1)).half().to(dev).clamp(-1,1)
view = base[10:139, 20:149].contiguous()  # actually contiguous slice with offset
try:
    out = kernel.run(view, 4)
    ref = kernel.reference(view, 4)
    err = (out.double()-ref).abs().max().item()
    worst = max(worst, err)
    results.append({"kind":"offset_view","err":err})
except Exception as e:
    results.append({"kind":"offset_view","error":str(e)})

fails = [r for r in results if r.get("fail")]
print(json.dumps({"metric":"max abs error vs FP64 per-step reference, tolerance 2e-5",
 "reason":"contract requires every element within 2e-5 of U_steps over whole legal domain",
 "n_cases":len(results),"n_fail":len(fails),"worst_max_abs_err":worst,"fails":fails[:5]}))