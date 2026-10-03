import sys, torch, json
sys.path.insert(0, "/root/cases/case_110")
import kernel

dev = "cuda" if torch.cuda.is_available() else "cpu"
g = torch.Generator().manual_seed(1)
state = torch.randn((2,8,4), generator=g).to(dev)
events = torch.randn((2,2,4), generator=g).to(dev)
ev_bits_before = events.view(torch.int32).clone()
st_storage_before = state.data_ptr()
expected_bits = kernel.reference(state, events).view(torch.int32).clone()
out = kernel.run(state, events)
torch.cuda.synchronize()
same_obj = (out is state)
same_ptr = (state.data_ptr() == st_storage_before)
bits_ok = bool((state.view(torch.int32) == expected_bits).all().item())
events_ok = bool((events.view(torch.int32) == ev_bits_before).all().item())

# is_neg / resolve_neg surface: does any in-domain contiguous tensor have is_neg True?
is_neg_present = hasattr(torch.Tensor, "is_neg") or hasattr(state, "is_neg")
neg_tests = {}
try:
    neg_tests["plain_is_neg"] = bool(state.is_neg())
    neg_tests["resolve_neg_same_ptr"] = (state.resolve_neg().data_ptr() == state.data_ptr())
    neg_tests["resolve_neg_same_obj"] = (state.resolve_neg() is state)
except Exception as e:
    neg_tests["error"] = repr(e)
# try plausible neg-view constructions
constructions = {}
for name, t in [
    ("flip_dim1", torch.randn((2,8,4), generator=g).to(dev).flip(1)),
    ("neg_view_via_stride0", torch.randn((2,8,4), generator=g).to(dev)),
]:
    try:
        constructions[name] = {
            "is_contiguous": t.is_contiguous(),
            "is_neg": bool(t.is_neg()),
            "is_contiguous_and_is_neg": t.is_contiguous() and bool(t.is_neg()),
        }
    except Exception as e:
        constructions[name] = {"error": repr(e)}
# flip is contiguous? no; check whether a contiguous tensor with is_neg True is constructible
contig_neg_found = False
try:
    for t in [torch.randn((2,8,4)).to(dev), torch.zeros((2,8,4)).to(dev)]:
        if t.is_contiguous() and t.is_neg():
            contig_neg_found = True
except Exception:
    pass
result = {
    "returned_same_object": same_obj,
    "storage_ptr_unchanged": same_ptr,
    "result_bits_match_reference": bits_ok,
    "events_preserved": events_ok,
    "is_neg_api_present": is_neg_present,
    "neg_tests": neg_tests,
    "constructions": constructions,
    "contig_neg_view_found": contig_neg_found,
}
print(json.dumps(result))
