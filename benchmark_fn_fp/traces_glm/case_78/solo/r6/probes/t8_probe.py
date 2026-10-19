import torch, sys, json
sys.path.insert(0, "/root/cases/case_151")
import kernel

big = torch.randint(0, 3329, (4, 130), dtype=torch.int32).cuda()
view = big[1:3, 2:66]  # contiguous, nonzero storage offset
assert view.is_contiguous() and view.storage_offset() != 0
out = kernel.run(view)
ref = kernel.reference(view)
mismatch = int((out != ref).sum().item())
print(json.dumps({"metric": "exact match for contiguous view with storage offset 2",
                  "storage_offset": view.storage_offset(), "mismatch": mismatch,
                  "shape": tuple(view.shape), "all_exact": mismatch == 0}))