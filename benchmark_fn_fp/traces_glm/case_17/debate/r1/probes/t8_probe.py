import torch, sys, json
sys.path.insert(0, "/root/cases/case_17")
from kernel import chunked_cumsum

torch.manual_seed(0)
x = torch.randn(3, 100, device="cuda", dtype=torch.float32)
out = chunked_cumsum(x, chunk=64)
ref = torch.cumsum(x, dim=1)
tail = slice(64, 100)  # positions written by full chunk vs. trailing partial chunk
written = slice(0, 64)
err_tail = (out[:, tail] - ref[:, tail]).abs()
print(json.dumps({
    "seqlen": 100, "chunk": 64, "shape": list(out.shape), "dtype": str(out.dtype),
    "tail_indices": [64, 99],
    "max_abs_err_written": float((out[:, written] - ref[:, written]).abs().max()),
    "max_abs_err_tail": float(err_tail.max()),
    "tail_is_all_zero": bool((out[:, tail] == 0).all()),
    "ref_tail_allclose": bool(torch.allclose(out[:, tail], ref[:, tail], atol=1e-5)),
    "sample_out_tail": out[0, 64:70].tolist(),
    "sample_ref_tail": ref[0, 64:70].tolist(),
}))