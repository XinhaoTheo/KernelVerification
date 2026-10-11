import torch, json, sys
sys.path.insert(0, "/root/cases/case_116")
import kernel

dev = "cuda"
M, N, K = 4, 4, 512
split_k = 4
seg = K // split_k  # 128 rows of B per split
a = torch.full((M, K), 0.5, dtype=torch.float16, device=dev)

best = None
for seed in range(60):
    g = torch.Generator(device="cpu").manual_seed(seed)
    # splits 0-2: strongly positive B rows (random FP16 grain, values 0.25..0.5)
    pos = (0.25 + 0.25 * torch.rand((3 * seg, N), generator=g)).half()
    # split 3: strongly negative rows chosen so each column's total is near 0
    neg = -(0.25 + 0.25 * torch.rand((seg, N), generator=g)).half()
    b = torch.cat([pos, neg], dim=0).to(dev)
    bias = torch.zeros(N, dtype=torch.float32, device=dev)
    ref = kernel.reference(a, b, bias, split_k)
    # cancel target: subtract per-column excess via bias? bias must be |x|<=1;
    # instead scale split-3 rows to approximately cancel each column
    col_sums_pos = torch.matmul(a[:, :3*seg].double(), b[:3*seg].double())  # [M,N] ~ big
    # choose split-3 rows so column sums cancel: set each neg row constant per column
    # target col j = 0.5*sum_k b[k,j]; make split3 = -colsum_pos[:,j]/(0.5*seg)
    need = -torch.matmul(a[0, :3*seg].double(), b[:3*seg].double().cpu()) / (0.5 * seg)
    need = need.clamp(-0.5, 0.5).to(torch.float16)
    b3 = need.unsqueeze(0).expand(seg, N).contiguous().to(dev)
    b = torch.cat([b[:3*seg], b3], dim=0)
    ref = kernel.reference(a, b, bias, split_k)
    if ref.abs().max() > 0.05:
        continue  # want near-zero targets
    out = kernel.run(a, b, bias, split_k)
    err = (out - ref).abs()
    tol = 0.01 + 0.001 * ref.abs()
    viol = int((err > tol).sum())
    rec = {"seed": seed, "max_abs_err": float(err.max()),
           "max_ref_abs": float(ref.abs().max()), "violations": viol}
    if best is None or rec["max_abs_err"] > best["max_abs_err"]:
        best = rec
print(json.dumps({
    "metric": "max abs err vs FP64 ref; partials large (~24), target near zero, split_k=4",
    "reason": "c1: FP16 scratch rounding of large non-representable partials should exceed 0.01 base tolerance when target cancels",
    "best": best,
    "claims_confirmed_if": "best.violations > 0"
}))