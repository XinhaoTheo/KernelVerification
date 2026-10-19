import torch, itertools, sys
sys.path.insert(0, "/root/cases/case_117")
import kernel

fails = []
tested = 0
max_rel = 0.0
combos = []
for m in [1, 3, 16, 17, 33, 64]:
    for n in [1, 5, 16, 33, 64]:
        for k in [1, 7, 31, 32, 33, 65, 127, 257, 512]:
            for sk in [1, 2, 4, 8]:
                combos.append((m, n, k, sk))
torch.manual_seed(0)
for (m, n, k, sk) in combos:
    gen = torch.Generator("cpu").manual_seed(m * 1000 + n * 100 + k * 10 + sk)
    a = (torch.rand((m, k), generator=gen) - .5).half().cuda()
    b = (torch.rand((k, n), generator=gen) - .5).half().cuda()
    bias = (2 * torch.rand((n,), generator=gen) - 1).float().cuda()
    out = kernel.run(a, b, bias, sk)
    ref = kernel.reference(a, b, bias, sk)
    err = (out.double() - ref.double()).abs()
    tol = 0.01 + 0.001 * ref.double().abs()
    tested += 1
    if not bool((err <= tol).all()):
        fails.append((m, n, k, sk, err.max().item(), ref.abs().max().item()))
import json
print(json.dumps({"tested": tested, "num_fail": len(fails), "fails": fails[:10]}))