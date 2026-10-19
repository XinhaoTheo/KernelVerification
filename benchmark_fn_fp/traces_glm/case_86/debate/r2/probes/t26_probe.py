import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_159/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev = "cuda"
res = {}
def try_case(name, fn):
    try:
        res[name] = {"raised": False, "detail": fn()}
    except Exception as ex:
        res[name] = {"raised": True, "error_type": type(ex).__name__, "detail": repr(ex)[:200]}
def offset_view():
    g = torch.Generator(device="cpu").manual_seed(5)
    big_m = torch.randint(-1, 2, (64, 32, 2, 2), generator=g, dtype=torch.int8).to(dev)
    big_b = torch.randint(-32768, 32768, (64, 32, 2), generator=g, dtype=torch.int16).to(dev)
    big_i = torch.randint(-32768, 32768, (64, 2), generator=g, dtype=torch.int16).to(dev)
    m = big_m[13:14, 5:15]   # [1,10,2,2] contiguous view, nonzero storage offset
    b = big_b[13:14, 5:15]   # [1,10,2] contiguous view, nonzero offset
    i = big_i[13:14]        # [1,2] contiguous view, nonzero offset
    info = k.validate_inputs(m, b, i)
    out = k.run(m, b, i); ref = k.reference(m, b, i)
    return {"info": info, "exact": bool(torch.equal(out, ref)),
            "shape": list(out.shape), "dtype": str(out.dtype),
            "m_offset_nonzero": bool(m.storage_offset() != 0)}
try_case("offset_views", offset_view)
print(json.dumps(res))