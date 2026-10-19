
import torch, json, importlib.util, sys
spec = importlib.util.spec_from_file_location("kern", "/root/cases/case_163/kernel.py")
kern = importlib.util.module_from_spec(spec); spec.loader.exec_module(kern)

def exact_dft(x):  # x fp16 [B,N] -> fp64 components [B,N/2+1,2]
    xd = x.to(torch.float64)
    N = x.shape[1]
    j = torch.arange(N, dtype=torch.float64)
    k = torch.arange(N//2+1, dtype=torch.float64)
    phase = torch.outer(k, j) * (-2*torch.pi/N)
    re = xd @ torch.cos(phase).T
    im = xd @ torch.sin(phase).T
    return torch.stack([re, im], dim=-1)

results = []
worst_ratio = 0.0
fails = 0
def check(name, x):
    global worst_ratio, fails
    out = kern.run(x)
    ref = exact_dft(x)
    assert out.shape == (x.shape[0], x.shape[1]//2+1, 2), out.shape
    assert out.dtype == torch.float32 and out.is_contiguous()
    d = (out.to(torch.float64) - ref).abs()
    tol = 1e-4 + 1e-6*ref.abs()
    bad = d > tol
    nonfinite = (~torch.isfinite(out)).sum().item()
    ratio = (d/tol).max().item()
    worst_ratio = max(worst_ratio, ratio)
    n_bad = bad.sum().item() + nonfinite
    if n_bad: fails += 1
    results.append({"name": name, "shape": list(x.shape), "violations": int(n_bad),
                    "nonfinite": int(nonfinite), "max_ratio": ratio,
                    "max_abs_err": d.max().item(),
                    "max_abs_exact": ref.abs().max().item()})

torch.manual_seed(0)
for N in (64,128,256):
    for B in (1,3,256):
        check(f"rand_B{B}_N{N}", torch.randn(B,N,dtype=torch.float16,device="cuda"))
# extreme / special values
N=256
check("all_max", torch.full((2,N), 65504.0, dtype=torch.float16, device="cuda"))
check("all_min", torch.full((2,N), -65504.0, dtype=torch.float16, device="cuda"))
check("subnormal", (torch.rand(4,N,device="cuda")*6e-8 - 3e-8).to(torch.float16))
check("signed_zeros", torch.zeros(2,N,dtype=torch.float16,device="cuda"))
z = torch.zeros(2,N,dtype=torch.float16,device="cuda"); z[0,:] = -0.0; check("neg_zero_row", z)
check("mixed_extremes", (torch.randn(8,N,device="cuda")*20000).to(torch.float16))
check("impulse", torch.zeros(2,N,dtype=torch.float16,device="cuda")[:, :].index_fill_(1, torch.tensor([0,1,17,63],device="cuda"), 65504.0))
check("alternating", ((-1.0)**torch.arange(N,device="cuda")).to(torch.float16).repeat(3,1).to(torch.float16))
# nonzero storage offset
base = torch.randn(2, N+5, dtype=torch.float16, device="cuda")
check("storage_offset", base[:, 5:].contiguous().clone())  # contiguous full tensor already
vo = torch.randn(2*N+7, dtype=torch.float16, device="cuda")
off = vo[3:3+2*N].view(2,N)
check("storage_offset_view", off)
# input preserved
x0 = torch.randn(4,N,dtype=torch.float16,device="cuda"); x0c = x0.clone()
o = kern.run(x0)
results.append({"input_preserved": bool(torch.equal(x0, x0c)),
                "output_separate": o.data_ptr() != x0.data_ptr()})
# reference() agreement sanity
ref_lib = kern.reference(x0)
results.append({"lib_ref_max_diff_vs_exact": (ref_lib - exact_dft(x0).to(torch.float64)).abs().max().item()})

print(json.dumps({"fails": fails, "worst_ratio": worst_ratio, "results": results}))
