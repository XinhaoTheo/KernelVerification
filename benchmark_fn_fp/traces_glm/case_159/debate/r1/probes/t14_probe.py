import json, torch, sys
sys.path.insert(0, "/root/cases/case_159")
import kernel

dev = "cuda"
res = {"cases": 0, "mismatch": 0, "errors": []}
def check(m, b, i, tag):
    out = kernel.run(m, b, i)
    ref = kernel.reference(m, b, i)
    res["cases"] += 1
    if not torch.equal(out, ref):
        res["mismatch"] += 1
        if len(res["errors"]) < 5:
            res["errors"].append(tag)

gen = torch.Generator(device="cpu").manual_seed(123)
for N in range(1, 33):
    for B in (1, 64):
        m = torch.randint(-1, 2, (B, N, 2, 2), generator=gen, dtype=torch.int8).to(dev)
        b = torch.randint(-32768, 32768, (B, N, 2), generator=gen, dtype=torch.int16).to(dev)
        i = torch.randint(-32768, 32768, (B, 2), generator=gen, dtype=torch.int16).to(dev)
        check(m, b, i, {"kind": "rand", "N": N, "B": B})
for N in (1, 8, 9, 15, 16, 17, 24, 25, 32):
    B = 8
    for fill in (-32768, 32767):
        m = torch.ones((B, N, 2, 2), dtype=torch.int8).to(dev)
        b = torch.full((B, N, 2), fill, dtype=torch.int16).to(dev)
        i = torch.full((B, 2), fill, dtype=torch.int16).to(dev)
        check(m, b, i, {"kind": "extreme_ones", "N": N, "fill": fill})
        m = torch.zeros((B, N, 2, 2), dtype=torch.int8).to(dev)
        check(m, b, i, {"kind": "extreme_zeros", "N": N, "fill": fill})
        m = torch.ones((B, N, 2, 2), dtype=torch.int8)
        m[:, :, 0, 1] = 0; m[:, :, 1, 0] = 0
        check(m.to(dev), b, i, {"kind": "extreme_sing", "N": N, "fill": fill})
for N in (9, 17, 25, 32):
    B = 4
    m_big = torch.randint(-1, 2, (B, 64, 2, 2), generator=gen, dtype=torch.int8)
    b_big = torch.randint(-32768, 32768, (B, 64, 2), generator=gen, dtype=torch.int16)
    i_big = torch.randint(-32768, 32768, (B, 2), generator=gen, dtype=torch.int16)
    m = m_big.as_strided((B, N, 2, 2), (N*4, 4, 2, 1), storage_offset=128).to(dev)
    b = b_big.as_strided((B, N, 2), (N*2, 2, 1), storage_offset=64).to(dev)
    i = i_big.as_strided((B, 2), (2, 1), storage_offset=2).to(dev)
    check(m, b, i, {"kind": "offset_view", "N": N})
for trial in range(40):
    B = int(torch.randint(1, 65, (1,), generator=gen))
    N = int(torch.randint(1, 33, (1,), generator=gen))
    m = torch.randint(-1, 2, (B, N, 2, 2), generator=gen, dtype=torch.int8).to(dev)
    b = torch.randint(-32768, 32768, (B, N, 2), generator=gen, dtype=torch.int16).to(dev)
    i = torch.randint(-32768, 32768, (B, 2), generator=gen, dtype=torch.int16).to(dev)
    check(m, b, i, {"kind": "sweep", "N": N, "B": B})
print(json.dumps({**res, "supports_defect": res["mismatch"] > 0}))