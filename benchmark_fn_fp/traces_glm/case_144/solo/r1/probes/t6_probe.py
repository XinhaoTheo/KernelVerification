import sys, torch
sys.path.insert(0, "/root/cases/case_144")
import kernel

dev = "cuda"
fails = []

def check(name, x):
    out = kernel.run(x)
    ref = kernel.reference(x)
    if not torch.equal(out, ref):
        fails.append((name, (out != ref).nonzero().flatten()[:5].tolist(),
                      out[~torch.eq(out,ref)][:5].tolist(), ref[~torch.eq(out,ref)][:5].tolist()))

# random exhaustive-ish sweep over all fp16 bit patterns per row
g = torch.Generator().manual_seed(0)
# random rows with N=4096 (max), various M
x = torch.randn(64, 4096, generator=g).to(torch.float16).to(dev)
check("randn 64x4096", x)

# all bit patterns except NaN/Inf, many rows, N=512
bits = torch.arange(65536, dtype=torch.int64)
bits = bits[~(((bits & 0x7c00) == 0x7c00))]  # exclude nan/inf exponent all ones
torch.manual_seed(1)
perm = bits[torch.randperm(bits.numel(), generator=torch.Generator().manual_seed(1))]
x = perm[:512*64].reshape(64,512)
v = torch.zeros(64,512, dtype=torch.float16); v.view(torch.int16).copy_(x.to(torch.int16))
v = v.to(dev)
check("all-bit-patterns 64x512", v)

# signed zeros: first zero of either sign wins; row max zero
x = torch.zeros(8, 16, dtype=torch.float16)
for r in range(8):
    x[r, 3] = -0.0; x[r, 5] = 0.0; x[r, 9] = -0.0
    # max is 0; first zero at col 3 (neg zero) -> expect 3
check("signed zeros all-zero rows", x.to(dev))
x2 = x.clone(); x2[:, 2] = -5.0  # max still zero
check("signed zeros with negatives", x2.to(dev))

# max is +0 vs -0 first, plus subnormals
x = torch.full((4, 32), -65504.0, dtype=torch.float16)
x[0, 7] = -0.0; x[0, 7] = 0.0  # +0 at 7
x[1, 2] = -0.0; x[1, 5] = 0.0
x[2, 0] = 6.1035e-5  # smallest subnormal max? row all -65504 so max is subnormal at 0
x[3, 31] = -5.96e-8  # negative subnormal, row is -65504 -> max is the negative subnormal at 31
check("zeros/subnormals extreme", x.to(dev))

# +/-65504 ties
x = torch.zeros(16, 1024, dtype=torch.float16)
for r in range(16):
    x[r, :900] = -65504.0
    cols = [5, 100, 250, 999]
    for c in cols: x[r, c] = 65504.0
check("65504 ties", x.to(dev))

# ties among repeated max at various positions incl padding lanes
x = torch.randn(33, 257, generator=torch.Generator().manual_seed(2)).to(torch.float16)
x[0, 0] = x[0, 256] = x[0, 128] = 10.0
x[1] = 1.0  # all equal -> expect 0
check("ties N=257", x.to(dev))

# N=1
x = torch.randn(5, 1).to(torch.float16).to(dev); check("N=1", x)

# storage-offset view
big = torch.randn(1, 3000, generator=torch.Generator().manual_seed(3)).to(torch.float16).to(dev)
view = big[0, 100:100+1024].reshape(8, 128)
assert view.is_contiguous()
check("storage-offset view", view)

# dense negative values, mixed subnormals
torch.manual_seed(4)
x = (torch.randn(128, 2048, generator=torch.Generator().manual_seed(4)) * 1e-4).to(torch.float16).to(dev)
check("tiny subnormal-ish 128x2048", x)

# verify run output dtype/device/preserve
x = torch.randn(4, 8).to(torch.float16).to(dev); xc = x.clone()
o = kernel.run(x)
info = dict(dtype=str(o.dtype), device=str(o.device), shape=tuple(o.shape), preserved=bool(torch.equal(x, xc)), fresh=o.data_ptr()!=x.data_ptr())
print("RESULT", "FAIL" if fails else "PASS", fails if fails else "", info)
