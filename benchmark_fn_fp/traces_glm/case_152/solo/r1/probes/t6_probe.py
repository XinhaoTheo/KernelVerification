import sys, json, zlib, torch
sys.path.insert(0, "/root/cases/case_152")
import kernel

dev = "cuda"
mismatches = 0
total = 0
tested = []
for width in (64, 128):
    for batch in (1, 7, 128):
        for seed in range(40):
            x = torch.randint(0, 256, (batch, width), dtype=torch.uint8,
                              generator=torch.Generator().manual_seed(seed)).to(dev)
            out = kernel.run(x)
            ref = kernel.reference(x)
            ok = torch.equal(out, ref) and out.dtype == torch.int64 and out.shape == (batch,) \
                 and (out >= 0).all() and (out <= 0xFFFFFFFF).all()
            total += batch
            if not ok:
                mismatches += int((out != ref).sum()) if out.shape == ref.shape else batch
                if mismatches <= 5:
                    i = int((out != ref).nonzero()[0]) if out.shape == ref.shape else 0
                    print("MISMATCH", width, batch, seed, i, int(out[i]), int(ref[i]))
        tested.append((width, batch))

# targeted patterns: all zeros, all 255, single-byte-set, contiguous view with offset
edge = [
    torch.zeros(3, 64, dtype=torch.uint8),
    torch.full((3, 64), 255, dtype=torch.uint8),
    torch.full((2, 128), 255, dtype=torch.uint8),
    torch.zeros(2, 128, dtype=torch.uint8),
]
for i in range(256):
    t = torch.zeros(1, 64, dtype=torch.uint8); t[0, 63] = i; edge.append(t)
    t = torch.zeros(1, 64, dtype=torch.uint8); t[0, 0] = i; edge.append(t)
for t in edge:
    x = t.to(dev)
    out = kernel.run(x); ref = kernel.reference(x)
    total += x.shape[0]
    if not torch.equal(out, ref):
        mismatches += x.shape[0]
        print("EDGE MISMATCH", x.shape, out.tolist(), ref.tolist())

# contiguous view with nonzero storage offset
big = torch.randint(0, 256, (300, 64), dtype=torch.uint8).to(dev)
view = big[100:100+7]
out = kernel.run(view); ref = kernel.reference(view)
total += 7
if not torch.equal(out, ref):
    mismatches += 7
    print("VIEW MISMATCH")

print(json.dumps({"mismatches": mismatches, "total_rows_checked": total,
                  "shapes": tested, "verdict_metric": "exact equality vs zlib.crc32 (tol=0)"}))
