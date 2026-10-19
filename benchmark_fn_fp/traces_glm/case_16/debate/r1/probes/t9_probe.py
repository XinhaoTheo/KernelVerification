import torch, sys, json
sys.path.insert(0, "/root/cases/case_17")
from kernel import chunked_cumsum

torch.manual_seed(1)
x = torch.randn(2, 32, device="cuda", dtype=torch.float32)
out = chunked_cumsum(x, chunk=64)
ref = torch.cumsum(x, dim=1)
print(json.dumps({
    "seqlen": 32, "chunk": 64, "shape": list(out.shape),
    "max_abs_err": float((out - ref).abs().max()),
    "out_is_all_zero": bool((out == 0).all()),
    "ref_nonzero_count": int((ref != 0).sum()),
    "out_nonzero_count": int((out != 0).sum()),
    "allclose": bool(torch.allclose(out, ref, atol=1e-6)),
    "sample_out": out[0, :4].tolist(),
    "sample_ref": ref[0, :4].tolist(),
}))