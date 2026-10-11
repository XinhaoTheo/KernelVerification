
import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_122/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev = "cuda"
results = {}
worst = 0.0; worst_case = None
for (H, W) in [(31,33),(4,4),(2,5),(129,3),(3,129),(8,128)]:
    g = torch.Generator(device="cpu").manual_seed(H*100+W)
    x = (2*torch.rand((H,W), generator=g)-1).half().to(dev)
    out = k.run(x, 4)
    ref = k.reference(x, 4)
    err = (out.double()-ref).abs()
    # boundary region error (edges and corners)
    bmask = torch.zeros_like(err, dtype=torch.bool)
    bmask[0,:]=True; bmask[-1,:]=True; bmask[:,0]=True; bmask[:,-1]=True
    bmax = err[bmask].max().item()
    imax = err[~bmask].max().item()
    results[f"{H}x{W}"] = {"max_err": err.max().item(), "boundary_max": bmax, "interior_max": imax}
    if err.max().item() > worst: worst, worst_case = err.max().item(), f"{H}x{W}"
print(json.dumps({"tolerance":2e-5,"cases":results,"worst":worst,"worst_case":worst_case,
                 "claim":"steps=4 fused reflection vs per-step zero-padded reference, boundary elements"}))
