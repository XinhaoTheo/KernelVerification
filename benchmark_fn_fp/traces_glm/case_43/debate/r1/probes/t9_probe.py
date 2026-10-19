import torch, json, sys
sys.path.insert(0, "/root/cases/case_116")
import kernel

dev = "cuda"
# Adversarial cancellation: K=512, split_k=4. Row i of A alternates so each
# 128-element split produces a large partial (~ up to 128*0.5*0.5=32 magnitude)
# but total sum is ~0.
M, N, K, split_k = 4, 4, 512, 4
a = torch.full((M, K), 0.5, dtype=torch.float16, device=dev)
# B rows: split 0..3 blocks of 128; within each element product 0.25; make
# each split partial large but splits cancel across split_k=4 partials?
# Simpler: per split partial magnitude = sum over 128 elems of 0.25 = 32.
# To make partial large but total near 0, alternate sign by split block.
signs = torch.tensor([1., 1., -1., -1.], device=dev)  # split blocks
b_blocks = []
for s in range(split_k):
    block = torch.full((K//split_k, N), 0.5, dtype=torch.float16, device=dev)
    if signs[s] < 0:
        block = -block
    b_blocks.append(block)
b = torch.cat(b_blocks, dim=0)
bias = torch.zeros(N, dtype=torch.float32, device=dev)

ref = kernel.reference(a, b, bias, split_k)
out = kernel.run(a, b, bias, split_k)
err = (out - ref).abs()
tol = 0.01 + 0.001 * ref.abs()
viol = err > tol
print(json.dumps({
    "metric": "max abs err vs FP64 ref, cancellation inputs",
    "max_abs_err": float(err.max()), "tol_at_max": float(tol.flatten()[int(err.argmax())]),
    "max_ref_abs": float(ref.abs().max()),
    "num_violations": int(viol.sum()), "numel": out.numel(),
    "dtype_out": str(out.dtype), "shape": list(out.shape),
    "split_k": split_k, "K": K,
    "ref_sample": ref.flatten()[:4].tolist(), "out_sample": out.flatten()[:4].tolist()
}))