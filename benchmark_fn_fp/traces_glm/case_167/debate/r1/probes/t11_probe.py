import json, sys, torch
sys.path.insert(0, '/root/cases/case_167')
import kernel

N = 500
big = torch.randint(0, 256, (N + 5,), dtype=torch.uint8, device='cuda')
view = big[3:3 + N]
assert view.is_contiguous() and view.storage_offset() == 3
before = big.clone()
out = kernel.run(view)
ref = kernel.reference(view)
mism = int((out != ref).sum())
input_mutated = int((big != before).sum())
shares = out.data_ptr() in (big.data_ptr(), view.data_ptr())
fresh_shape = tuple(out.shape) == (N,) and out.dtype == torch.uint8 and out.is_contiguous()
# also verify reference of a wrong-offset read would differ (sanity that values vary)
print(json.dumps({"mismatch_count": mism, "input_mutated_bytes": input_mutated, "shares_storage_with_input": shares, "fresh_shape_dtype_contig": fresh_shape}))