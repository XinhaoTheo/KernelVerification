import sys, json, torch
sys.path.insert(0, "/root/cases/case_145")
from kernel import run, reference

dev = "cuda"
results = []

def check(name, x):
    out = run(x)
    ref = reference(x)
    m = bool((out == ref).all().item())
    results.append({"case": name, "shape": list(x.shape), "match": m,
                    "mismatches": (out != ref).nonzero().flatten().tolist()[:10]})

# N=1 (BLOCK=1, tie key 0xffffffff at column 0)
x = torch.tensor([[float("-65504")], [0.0], [1.0]], dtype=torch.float32).to(torch.float16).to(dev)
check("N=1 extremes", x)

# Non-power-of-2 N with padding lanes and extremes, -65504 present
g = torch.Generator(device="cpu").manual_seed(3)
x = (torch.randn((8, 13), generator=g) * 100).to(torch.float16)
x[3, 7] = -65504.0; x[3, 2] = 65504.0; x[5, 0] = 65504.0
x = x.to(dev)
check("N=13 extremes+padding", x)

# Non-power-of-2 N with all-equal tiny subnormals (padding must never win)
x = torch.full((4, 7), 6e-8, dtype=torch.float32).to(torch.float16).to(dev)
check("N=7 all-equal subnormals", x)

# Larger mixed grid: ties, max at column 0, negative max row
g = torch.Generator(device="cpu").manual_seed(7)
x = torch.randn((64, 100), generator=g).to(torch.float16)
x[10, :] = -5.0; x[10, 50] = -1.0        # negative-max row, max at col 50
x[20, :] = 3.25;                          # full tie row -> index 0
x[30, 0] = 65504.0                        # max at column 0
x = x.to(dev)
check("N=100 mixed ties", x)

# Max value at column 0 across many rows with varied values
g = torch.Generator(device="cpu").manual_seed(11)
x = torch.randn((33, 257), generator=g).to(torch.float16)
x[:, 0] = 2048.0
x = x.to(dev)
check("N=257 max at col 0", x)

print(json.dumps({"cases": results, "all_match": all(r["match"] for r in results)}))