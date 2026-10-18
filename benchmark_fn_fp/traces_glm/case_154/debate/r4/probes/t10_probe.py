import torch, triton, triton.language as tl, json

@triton.jit
def _divtest(X, Y, OUT, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < N
    x = tl.load(X + i, mask=m, other=1.0)
    y = tl.load(Y + i, mask=m, other=1.0)
    tl.store(OUT + i, tl.div_rn(x, y), mask=m)

N = 1 << 20
g = torch.Generator().manual_seed(3)
x = (torch.rand(N, generator=g) * 2 - 1) * 1e3
y = (torch.rand(N, generator=g) * 2 - 1) * 1e3
x[(x == 0)] = 1.5  # div by anything nonzero
y[(y == 0)] = 2.5
x = x.to("cuda"); y = y.to("cuda")
out = torch.empty_like(x)
_divtest[(triton.cdiv(N, 1024),)](x, y, out, N=N, BLOCK=1024, enable_fp_fusion=False)
torch.cuda.synchronize()

# correctly-rounded target: round exact (FP64) quotient to FP32
exact = (x.to(torch.float64) / y.to(torch.float64))
cr = exact.to(torch.float32)
mismatch = (out != cr)
# ulp distance of out from exact
reldiff = ((out.to(torch.float64) - exact).abs() / exact.abs())
ulp32 = torch.ldexp(torch.ones((), dtype=torch.float64, device="cuda"),
                    (out.to(torch.float64).abs().log2().floor() - 23))
# simpler: report mismatch count and max relative error
print(json.dumps({
    "metric": "div_rn_correct_rounding_vs_round_of_fp64_exact",
    "n": N, "mismatch_count": int(mismatch.sum()),
    "max_rel_err": float(reldiff.max()),
    "mismatch_examples": [
        {"x": float(x[j]), "y": float(y[j]), "out": float(out[j]), "correct_rounded": float(cr[j])}
        for j in mismatch.nonzero()[:5].flatten().tolist()
    ]}))
