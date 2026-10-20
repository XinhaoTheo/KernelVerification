import torch, sys
sys.path.insert(0, "/root/cases/case_164")
import kernel

# all finite BF16 values in [-10,10]
vals = []
bits = torch.arange(0x0000, 0x10000, dtype=torch.int32)
bf = bits.to(torch.uint16).view(torch.bfloat16).to(torch.float64)
mask = torch.isfinite(bf) & (bf.abs() <= 10)
sub = torch.arange(0x10000, dtype=torch.int32)[mask]
allv = torch.from_numpy(sub.numpy().astype('int32')).to(torch.uint16).view(torch.bfloat16).to(torch.float64)
print("num legal values:", allv.numel())

x = torch.from_numpy(sub.numpy().astype('int32')).to(torch.uint16).view(torch.bfloat16).to('cuda')
out = kernel.run(x)
ref = torch.exp2(x.to(torch.float64).cpu())
rel = (out.cpu().to(torch.float64) - ref).abs() / ref
bad = (rel > 1e-6)
print("max rel err:", rel.max().item())
print("count violations:", int(bad.sum()))
if bad.any():
    i = rel.argmax()
    print("worst x:", x.cpu().float()[i].item(), "out:", out.cpu()[i].item(), "ref:", ref[i].item(), "rel:", rel[i].item())
    idxs = torch.nonzero(bad).flatten()[:10]
    for j in idxs:
        print("  viol x=", x.cpu().float()[j].item(), "rel=", rel[j].item())
import json
print(json.dumps({"metric":"max relative error over exhaustive BF16 domain [-10,10]","max_rel":rel.max().item(),"violations":int(bad.sum()),"n":int(x.numel())}))
