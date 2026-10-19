import torch, sys, json
sys.path.insert(0, "/root/cases/case_151")
import kernel

# contiguous view with nonzero storage offset: row-slice of a 3x64 tensor
big = torch.randint(0, 3329, (3, 64), dtype=torch.int32).cuda()
view = big[1:3]
assert view.is_contiguous() and view.storage_offset() == 64
out = kernel.run(view)
ref = kernel.reference(view)
mismatch = int((out != ref).sum().item())
# note: c1 already confirmed kernel values are wrong generally; check whether
# the view is read from the correct memory (compare kernel on view vs kernel on a clone of same data)
clone = view.clone()
same_as_clone = bool((out == kernel.run(clone)).all().item())
print(json.dumps({"metric": "kernel(view) vs kernel(clone-of-view) and reference",
                  "storage_offset": view.storage_offset(),
                  "mismatch_vs_reference": mismatch,
                  "identical_to_clone_run": same_as_clone,
                  "note": "identical_to_clone_run=True proves storage offset handled correctly; value error is the already-confirmed c1 bug"}))