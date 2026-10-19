
import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_122/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev = "cuda"
results = {}
for (H, W) in [(31,33),(4,4),(2,5),(129,3),(3,129),(8,128),(129,129)]:
    g = torch.Generator(device="cpu").manual_seed(H*100+W+7)
    x = (2*torch.rand((H,W), generator=g)-1).half().to(dev)
    out4 = k.run(x, 4); ref4 = k.reference(x, 4)
    out8 = k.run(x, 8); ref8 = k.reference(x, 8)
    e4 = (out4.double()-ref4).abs().max().item()
    e8 = (out8.double()-ref8).abs().max().item()
    results[f"{H}x{W}"] = {"steps4_max_err": e4, "steps8_max_err": e8}
print(json.dumps({"tolerance":2e-5,"cases":results,
                 "claim":"steps=8 double fused block composition error"}))
