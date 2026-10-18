import torch, sys
sys.path.insert(0, "/root/cases/case_151")
import kernel as K

dev = "cuda" if torch.cuda.is_available() else "cpu"
results = []
def check(tag, x):
    x = x.to(dev, torch.int32).contiguous()
    out = K.run(x)
    ref = K.reference(x)
    mismatches = int((out != ref).sum())
    results.append({"tag": tag, "N": x.shape[1], "B": x.shape[0], "mismatches": mismatches})
    if mismatches:
        idx = (out != ref).nonzero()[:5].tolist()
        results[-1]["first_bad"] = idx
        results[-1]["kernel_vals"] = [int(out[tuple(i)]) for i in idx]
        results[-1]["ref_vals"] = [int(ref[tuple(i)]) for i in idx]

g = torch.Generator().manual_seed(1234)
for N in (64, 128):
    check("random_b7", torch.randint(0, 3329, (7, N), generator=g))
    check("random_b128", torch.randint(0, 3329, (128, N), generator=g))
    check("zeros_b1", torch.zeros((1, N)))
    check("allmax_b1", torch.full((1, N), 3328))
    check("allones_b3", torch.ones((3, N)))
    check("unit_e0_b1", torch.eye(1, N).to(torch.int32))  # first basis vector
    # storage-offset contiguous view
    big = torch.randint(0, 3329, (4, 2*N), generator=g)
    view = big[:, N:2*N].contiguous()
    check("offsetview", view)
print("CUDA:", torch.cuda.is_available())
import json
print(json.dumps(results))
