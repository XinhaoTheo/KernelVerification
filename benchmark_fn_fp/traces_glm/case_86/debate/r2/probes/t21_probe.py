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
def f2():
    g = torch.Generator(device="cpu").manual_seed(5)
    big_m = torch.randint(-1, 2, (64, 32, 2, 2), generator=g, dtype=torch.int8).to(dev)
    big_b = torch.randint(-32768, 32768, (64, 32, 2), generator=g, dtype=torch.int16).to(dev)
    big_i = torch.randint(-32768, 32768, (64, 2), generator=g, dtype=torch.int16).to(dev)
    m = big_m[13, 5:15]; b = big_b[13, 5:15]; i = big_i[13]
    info = k.validate_inputs(m, b, i)
    out = k.run(m, b, i); ref = k.reference(m, b, i)
    return {"info": info, "exact": bool(torch.equal(out, ref)), "shape": list(out.shape), "dtype": str(out.dtype)}
try_case("offset_views", f2)
def f3():
    B, N = 2, 9
    m = torch.ones(B, N, 2, 2, dtype=torch.int8, device=dev)
    b = torch.full((B, N, 2), -32768, dtype=torch.int16, device=dev)
    i = torch.full((B, 2), 32767, dtype=torch.int16, device=dev)
    info = k.validate_inputs(m, b, i)
    return {"info": info, "exact": bool(torch.equal(k.run(m, b, i), k.reference(m, b, i)))}
try_case("extreme_int16", f3)
def f4():
    m, b, i = k.make_inputs()
    return {"is_neg": bool(m.is_neg()), "is_conj": bool(m.is_conj()), "is_inference": bool(m.is_inference())}
try_case("flag_values", f4)
print(json.dumps(res))