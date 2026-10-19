
import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_122/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev = "cuda"
results = {}
for H in range(1,4):
    for W in range(1,4):
        g = torch.Generator(device="cpu").manual_seed(H*10+W)
        x = (2*torch.rand((H,W), generator=g)-1).half().to(dev)
        errs = {}
        for s in (4,8):
            out = k.run(x, s); ref = k.reference(x, s)
            errs[s] = (out.double()-ref).abs().max().item()
        results[f"{H}x{W}"] = errs
over = {k2:v for k2,v in results.items() if max(v.values())>2e-5}
print(json.dumps({"tolerance":2e-5,"tiny_cases":results,"cases_over_tolerance":over,
                 "claim":"tiny H,W in 1..3 aliasing of 9-tap window vs zero-padded reference"}))
