import torch, importlib.util, json
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_148/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
res = {}
gen = torch.Generator().manual_seed(1)
def ref_cpu(x):
    return k.reference(x).to("cpu")
for n in [1, 131072, 4093, 256, 257]:
    x = (3*torch.randn(n, generator=gen)).to(torch.float16).cuda()
    xb = x.clone()
    out = k.run(x)
    r = ref_cpu(x)
    ok_bits = bool(torch.equal(out.view(torch.int16).cpu(), r.view(torch.int16).cpu()))
    fresh = out.data_ptr() != x.data_ptr()
    preserved = bool(torch.equal(x, xb))
    res[f"N={n}"] = {"bits_match": ok_bits, "fresh_output": fresh, "input_preserved": preserved,
                     "shape": list(out.shape), "dtype": str(out.dtype), "device": str(out.device)}
# storage-offset contiguous view
big = (3*torch.randn(300, generator=gen)).to(torch.float16).cuda()
view = big[5:5+100]  # contiguous, nonzero storage offset
assert view.is_contiguous() and view.storage_offset() > 0
try:
    out = k.run(view)
    r = ref_cpu(view)
    res["offset_view"] = {"accepted": True, "bits_match": bool(torch.equal(out.view(torch.int16).cpu(), r.view(torch.int16).cpu())),
                          "storage_offset": view.storage_offset()}
except Exception as e:
    res["offset_view"] = {"accepted": False, "error": str(e)}
print(json.dumps(res))
