import torch, sys, json
sys.path.insert(0, "/root/cases/case_117")
import kernel

fails = []
def check(a, b, bias, sk, tag):
    out = kernel.run(a, b, bias, sk)
    ref = kernel.reference(a, b, bias, sk)
    err = (out.double() - ref.double()).abs()
    tol = 0.01 + 0.001 * ref.double().abs()
    if not bool((err <= tol).all()):
        fails.append((tag, sk, err.max().item()))

torch.manual_seed(1)
for sk in (1, 2, 4, 8):
    # extreme magnitudes
    a = (torch.sign(torch.randn(64, 512)) * 0.5).half().cuda()
    b = (torch.sign(torch.randn(512, 64)) * 0.5).half().cuda()
    bias = (torch.sign(torch.randn(64)) * 1.0).float().cuda()
    check(a, b, bias, sk, "extreme")
    # zeros
    a = torch.zeros(33, 129, dtype=torch.float16).cuda()
    b = torch.zeros(129, 17, dtype=torch.float16).cuda()
    bias = torch.zeros(17, dtype=torch.float32).cuda()
    check(a, b, bias, sk, "zeros")
    # subnormals in fp16
    a = (torch.randn(17, 65) * 1e-7).half().cuda()
    b = (torch.randn(65, 33) * 1e-7).half().cuda()
    bias = (torch.randn(33) * 1e-38).float().cuda()
    check(a, b, bias, sk, "tiny")
    # cancellation-prone: same distribution signs matched
    g = torch.Generator("cpu").manual_seed(123)
    a = (torch.rand((64, 257), generator=g) - .5).half().cuda()
    b = (torch.rand((257, 64), generator=g) - .5).half().cuda()
    bias = (torch.rand((64,), generator=g) * 2 - 1).float().cuda()
    check(a, b, bias, sk, "cancel")
    # input preservation
    a0 = a.clone(); b0 = b.clone(); bias0 = bias.clone()
    kernel.run(a, b, bias, sk)
    if not (torch.equal(a, a0) and torch.equal(b, b0) and torch.equal(bias, bias0)):
        fails.append(("preserve", sk, 1.0))
    # output properties
    out = kernel.run(a, b, bias, sk)
    if out.dtype != torch.float32 or out.is_cuda == False or out.shape != (64, 64) or not bool(torch.isfinite(out).all()):
        fails.append(("output_props", sk, 1.0))

print(json.dumps({"num_fail": len(fails), "fails": fails[:10]}))