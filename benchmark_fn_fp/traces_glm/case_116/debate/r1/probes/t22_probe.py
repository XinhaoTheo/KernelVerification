import torch, json, sys
sys.path.insert(0, "/root/cases/case_116")
import kernel

M, N, K, split_k = 1, 4, 512, 4
seg = K // split_k  # 128 rows of B per split
a = torch.full((M, K), 0.5, dtype=torch.float16)  # CPU

best = None
tested = 0
for seed in range(200):
    g = torch.Generator(device="cpu").manual_seed(seed)
    # splits 0-2: random-grain positive values in [0.25,0.5] -> partials ~24-32, not FP16-representable
    pos = (0.25 + 0.25 * torch.rand((3 * seg, N), generator=g)).half()
    a_pos = a[:, :3 * seg].double()
    col_pos = torch.matmul(a_pos, pos.double()).flatten()  # [N], ~22-30
    # split 3: constant per-column value chosen so the full-K column sum cancels
    need = -col_pos / (0.5 * seg)  # within +-0.5
    if need.abs().max() > 0.5:
        continue
    b3 = need.to(torch.float16).unsqueeze(0).expand(seg, N).contiguous()
    b = torch.cat([pos, b3], dim=0)  # CPU [512, 4] FP16
    a_d = a.to("cuda"); b_d = b.to("cuda")
    bias = torch.zeros(N, dtype=torch.float32, device="cuda")
    ref = kernel.reference(a_d, b_d, bias, split_k)
    if float(ref.abs().max()) > 0.05:
        continue
    partials = [float(torch.matmul(a_d.double(), b_d[s*seg:(s+1)*seg].double()).abs().max())
                for s in range(split_k)]
    out = kernel.run(a_d, b_d, bias, split_k)
    # split_k invariance cross-check: FP32 scratch path
    out1 = kernel.run(a_d, b_d, bias, 1)
    err = (out - ref).abs()
    tol = 0.01 + 0.001 * ref.abs()
    viol = int((err > tol).sum())
    tested += 1
    rec = {"seed": seed, "max_abs_err": float(err.max()),
           "max_ref_abs": float(ref.abs().max()),
           "min_partial_abs": min(partials), "max_partial_abs": max(partials),
           "sk1_vs_sk4_maxdiff": float((out - out1).abs().max()),
           "violations": viol, "numel": out.numel()}
    if best is None or rec["max_abs_err"] > best["max_abs_err"]:
        best = rec
print(json.dumps({
    "metric": "max abs err vs FP64 ref; large non-representable partials (~24-32), near-zero target, split_k=4",
    "reason": "c1: FP16 scratch rounding of large partials should exceed 0.01 base tolerance when target cancels; also cross-check split_k invariance",
    "best": best, "cases_tested": tested,
    "claims_confirmed_if": "best is not None and best.violations > 0"
}))