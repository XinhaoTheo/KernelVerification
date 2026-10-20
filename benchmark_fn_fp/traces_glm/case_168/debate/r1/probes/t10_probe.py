import torch, sys, json
sys.path.insert(0, '/root/cases/case_168')
from kernel import run, reference

base = torch.arange(300, dtype=torch.uint8, device='cuda')
base_clone = base.clone()
view = base[3:259]
assert view.storage_offset() == 3 and view.is_contiguous()
out = run(view)
ref = reference(view)
input_preserved = torch.equal(base, base_clone)
alias = out.data_ptr() == base.data_ptr() or out.storage().data_ptr() == base.storage().data_ptr()
values_correct = torch.equal(out, ref)
# also check bytes around the view (outside region) unchanged
print(json.dumps({"metric": "exact equality on nonzero-offset view + backing preservation + storage aliasing",
                  "offset": view.storage_offset(),
                  "values_correct": bool(values_correct),
                  "input_backing_preserved": bool(input_preserved),
                  "output_shares_storage_with_input": bool(alias),
                  "output_fresh_contig": out.is_contiguous() and out.dtype==torch.uint8 and out.shape==view.shape}))