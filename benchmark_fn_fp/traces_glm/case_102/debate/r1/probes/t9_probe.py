import torch, triton, triton.language as tl, numpy as np, json

@triton.jit
def div_kernel(A, B, O, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < N
    a = tl.load(A + i, mask=m, other=1.0)
    b = tl.load(B + i, mask=m, other=1.0)
    tl.store(O + i, tl.div_rn(a, b), mask=m)

# mantissa-like values in [1,2) crossed with estimate^2-like values
g = torch.Generator().manual_seed(1)
a = torch.rand(1 << 20, generator=g) + 1.0     # [1,2)
e = torch.rand(1 << 20, generator=g) * 0.1 + 1.0
b = (e * e).to(torch.float32)
a = a.to(torch.float32)
a, b = a.cuda(), b.cuda()
out = torch.empty_like(a)
N = a.numel()
div_kernel[(triton.cdiv(N, 256),)](a, b, out, N=N, BLOCK=256, num_warps=4, enable_fp_fusion=False)
# correctly-rounded FP32 division reference (numpy division is round-to-nearest)
ref = (a.cpu().double() / b.cpu().double()).to(torch.float32).cuda()
exact = (out == ref)
mismatch = (~exact).sum().item()
rel = ((out.double() - ref.double()).abs() / ref.double().abs())
print(json.dumps({"n": N, "bit_exact_matches": int(exact.sum().item()), "mismatches": int(mismatch),
                 "max_rel_dev_from_correct_rounding": float(rel.max().item()),
                 "max_abs_dev": float((out-ref).abs().max().item()),
                 "div_rn_correctly_rounded": mismatch == 0}))