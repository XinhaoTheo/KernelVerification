import torch, sys
sys.path.insert(0, "/root/cases/case_17")
from kernel import chunked_cumsum

torch.manual_seed(0)
res = {}
for seqlen in [100, 64, 130]:
    x = torch.randn(3, seqlen, device="cuda", dtype=torch.float32)
    out = chunked_cumsum(x, chunk=64)
    ref = torch.cumsum(x, dim=1)
    diff = (out - ref).abs()
    tail = seqlen - (seqlen // 64) * 64
    res[seqlen] = {
        "max_abs_err_full": diff.max().item(),
        "max_abs_err_tail": diff[:, -tail:].max().item() if tail else None,
        "max_abs_err_body": diff[:, :seqlen - tail].max().item() if tail else diff.max().item(),
        "tail_len": tail,
    }
print(res)
