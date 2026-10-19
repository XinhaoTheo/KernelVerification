
import torch, zlib, json, sys
sys.path.insert(0, "/root/cases/case_153")
import kernel

results = {}
cases = []
for width in (64, 128):
    for seed in range(5):
        cases.append((f"rand_w{width}_s{seed}", torch.randint(0,256,(128,width),dtype=torch.uint8).cuda()))
    cases.append((f"zeros_w{width}", torch.zeros((128,width),dtype=torch.uint8).cuda()))
    cases.append((f"ff_w{width}", torch.full((128,width),255,dtype=torch.uint8).cuda()))
    cases.append((f"ramp_w{width}", torch.arange(width,dtype=torch.uint8).repeat(128,1).cuda()))
    cases.append((f"one_w{width}_b1", torch.randint(0,256,(1,width),dtype=torch.uint8).cuda()))

mism = 0; total = 0; first = None
for name, x in cases:
    out = kernel.run(x)
    ref = torch.tensor([zlib.crc32(bytes(r)) for r in x.cpu().tolist()], dtype=torch.int64, device=x.device)
    ok = torch.equal(out, ref)
    total += x.shape[0]
    if not ok:
        mism += (out != ref).sum().item()
        idx = (out != ref).nonzero()[0].item()
        first = {"case": name, "idx": idx, "kernel": out[idx].item(), "ref": ref[idx].item(),
                 "row_bytes": x[idx].cpu().tolist()[:16]}
        print(f"MISMATCH in {name} row {idx}: kernel={out[idx].item()} ref={ref[idx].item()}")
    else:
        print(f"OK {name} ({x.shape[0]} rows)")
print(json.dumps({"mismatched_rows": mism, "total_rows": total, "first_mismatch": first,
                  "metric": "exact int64 equality vs zlib.crc32, abs/rel tol=0"}))
