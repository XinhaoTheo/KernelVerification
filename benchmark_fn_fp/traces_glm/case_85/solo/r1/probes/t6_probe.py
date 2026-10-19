
import torch, json
from kernel import run, reference, make_inputs, validate_inputs

dev = "cuda"
fails = []
tested = 0

# 1) default make_inputs style, many seeds, all N 1..32
for seed in range(8):
    for B in (1, 3, 7, 16, 64):
        for N in range(1, 33):
            g = torch.Generator(device="cpu").manual_seed(seed*1000 + B*100 + N)
            m = torch.randint(-1, 2, (B, N, 2, 2), generator=g, dtype=torch.int8).to(dev)
            b = torch.randint(-32768, 32768, (B, N, 2), generator=g, dtype=torch.int16).to(dev)
            i = torch.randint(-32768, 32768, (B, 2), generator=g, dtype=torch.int16).to(dev)
            out = run(m, b, i)
            ref = reference(m.cpu(), b.cpu(), i.cpu()).to(dev)
            tested += 1
            if not torch.equal(out, ref):
                bad = (out != ref).any(dim=-1).nonzero()[:5]
                fails.append(dict(kind="rand", seed=seed, B=B, N=N, bad=bad.tolist()))

# 2) extreme values: all bias/initial at int16 extremes, matrices at extremes
for B in (2, 64):
    for N in (1, 8, 9, 15, 16, 17, 23, 31, 32):
        for mv in (-1, 0, 1):
            for bv, iv in ((-32768, -32768), (32767, 32767), (-32768, 32767)):
                m = torch.full((B, N, 2, 2), mv, dtype=torch.int8).to(dev)
                b = torch.full((B, N, 2), bv, dtype=torch.int16).to(dev)
                i = torch.full((B, 2), iv, dtype=torch.int16).to(dev)
                out = run(m, b, i)
                ref = reference(m.cpu(), b.cpu(), i.cpu()).to(dev)
                tested += 1
                if not torch.equal(out, ref):
                    fails.append(dict(kind="extreme", B=B, N=N, mv=mv, bv=bv, iv=iv))

# 3) singular matrices / identity / zero mixes, random bias
g = torch.Generator().manual_seed(123)
for trial in range(20):
    for N in (1, 7, 8, 12, 16, 25, 32):
        B = 5
        m = torch.randint(-1, 2, (B, N, 2, 2), generator=g, dtype=torch.int8)
        # make half singular (rows equal)
        if trial % 2 == 0:
            m[:, :, 1] = m[:, :, 0]
        b = torch.randint(-32768, 32768, (B, N, 2), generator=g, dtype=torch.int16)
        i = torch.randint(-32768, 32768, (B, 2), generator=g, dtype=torch.int16)
        m, b, i = m.to(dev), b.to(dev), i.to(dev)
        out = run(m, b, i)
        ref = reference(m.cpu(), b.cpu(), i.cpu()).to(dev)
        tested += 1
        if not torch.equal(out, ref):
            fails.append(dict(kind="singular", trial=trial, N=N))

print(json.dumps({"tested": tested, "fails": fails[:10], "n_fails": len(fails)}))
