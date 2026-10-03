import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_107/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
results = {}
# K == L configs (L <= 128), plus edge K=1, L=2, D=1
for (B,L,D,K) in [(2,8,4,8),(1,128,7,128),(3,2,1,2),(4,3,1024,3),(1,128,1024,64)]:
    g = torch.Generator("cpu").manual_seed(7)
    st = torch.randn((B,L,D), generator=g).to("cuda")
    ev = torch.randn((B,K,D), generator=g).to("cuda")
    ref = k.reference(st.clone(), ev.clone())
    out = k.run(st, ev)
    torch.cuda.synchronize()
    results[f"B{B}L{L}D{D}K{K}"] = {
        "bitwise_equal_reference": bool(torch.equal(out.cpu().view(torch.int32), ref.cpu().view(torch.int32))),
        "max_abs_err": float((out.cpu()-ref.cpu()).abs().max()),
        "L_minus_K": L-K,
    }
print(json.dumps(results))