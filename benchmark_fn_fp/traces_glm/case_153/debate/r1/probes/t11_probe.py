import sys, json, torch, zlib
sys.path.insert(0, "/root/cases/case_153")
import kernel as K

out = {"total_rows": 0, "mismatches": 0, "per_width": {}}
for width in (64, 128):
    mism = 0; rows = 0
    for seed in range(12):
        gen = torch.Generator(device="cpu").manual_seed(seed * 977 + 5)
        x = torch.randint(0, 256, (128, width), generator=gen, dtype=torch.uint8).cuda()
        got = K.run(x).cpu()
        exp = torch.tensor([zlib.crc32(bytes(r)) & 0xFFFFFFFF for r in x.cpu().tolist()], dtype=torch.int64)
        rows += x.shape[0]
        mism += int((got != exp).sum())
    out["per_width"][f"width{width}"] = {"rows": rows, "mismatches": mism}
    out["total_rows"] += rows; out["mismatches"] += mism
print(json.dumps(out))
