import torch, json, sys
sys.path.insert(0, "/root/cases/case_116")
import kernel

dev = "cuda"
M, N, K, split_k = 1, 4, 512, 4
seg = K // split_k  # 128

a = torch.full((M, K), 0.5, dtype=torch.float16)  # CPU first
best = None
for seed in range(80):
    g = torch.Generator(device="cpu").manual_seed(seed)
    # splits 0-2: random positive grain in [0.25,0.5] -> partial ~ 24-30 each
    pos = (0.25 + 0.25 * torch.rand((3 * seg, N), generator=g)).half()
    pos_d = pos.double()
    col_pos = torch.matmul(a.double(), pos_d).flatten()  # [N], magnitude ~22-28
    # split 3: constant per-column value so each column's total cancels
    need = -col_pos / (0.5 * seg)  # values within +-0.5
    if need.abs().max() > 0.5:
        continue
    b3 = need.to(torch.float16).unsqueeze(0).expand(seg, N).contiguous()
    b = torch.cat([pos, b3], dim=0).to(dev)
    a_d = a.to(dev)
    bias = torch.zeros(N, dtype=torch.float32, device=dev)
    ref = kernel.reference(a_d, b, bias, split_k)
    if float(ref.abs().max()) > 0.05:
        continue
    # per-split partial magnitudes
    partials = []
    for s in range(split_k):
        block = b[s*seg:(s+1)*seg].double()
        partials.append(float(torch.matmul(a_d.double(), block).abs().max()))
    out = kernel.run(a_d, b, bias, split_k)
    err = (out - ref).abs()
    tol = 0.01 + 0.001 * ref.abs()
    viol = int((err > tol).sum())
    rec = {"seed": seed, "max_abs_err": float(err.max()),
           "max_ref_abs": float(ref.abs().max()),
           "min_partial_abs": min(partials), "max_partial_abs": max(partials),
           "violations": viol, "numel": out.numel()}
    if best is None or rec["max_abs_err"] > best["max_abs_err"]:
        best = rec
print(json.dumps({
    "metric": "max abs err vs FP64 ref; large non-representable partials, near-zero target, split_k=4",
    "reason": "c1: FP16 scratch rounding of large partials should exceed 0.01 base tolerance when target cancels",
    "best": best,
    "a_partial_reps": "random-grain [0.25,0.5] values, partials ~24-30, not FP16-representable sums",
    "claims_confirmed_if": "best.violations > 0"
}))