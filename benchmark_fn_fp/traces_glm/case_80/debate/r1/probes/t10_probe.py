import sys, json, torch, zlib
sys.path.insert(0, "/root/cases/case_153")
import kernel as K

out = {}
for width in (64, 128):
    gen = torch.Generator(device="cpu").manual_seed(7)
    x = torch.randint(0, 256, (128, width), generator=gen, dtype=torch.uint8).cuda()
    got = K.run(x).cpu()
    exp = torch.tensor([zlib.crc32(bytes(r)) & 0xFFFFFFFF for r in x.cpu().tolist()], dtype=torch.int64)
    big = exp >= 2**31
    big_mism = int((got[big] != exp[big]).sum()) if big.any() else 0
    neg = int((got < 0).sum())
    out[f"width{width}"] = {
        "n_big_checksums": int(big.sum()),
        "big_checksum_mismatches": big_mism,
        "negative_outputs": neg,
        "total_mismatches": int((got != exp).sum()),
        "sample_big_got": int(got[big][0]) if big.any() else None,
        "sample_big_exp": int(exp[big][0]) if big.any() else None,
    }
print(json.dumps(out))
