import json, torch, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_106/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

results = {}
geoms = [(2,64,64,32), (1,64,64,63), (1,2048,1,128), (4,16,32,8), (1,4096,1024,2)]
for (B,L,D,K) in geoms:
    g = torch.Generator("cpu").manual_seed(42)
    state = torch.randn((B,L,D), generator=g).cuda()
    ev = torch.randn((B,K,D), generator=g).cuda()
    ref = k.reference(state, ev)
    ev_bits = ev.view(torch.int32).clone()
    out = k.run(state, ev)
    torch.cuda.synchronize()
    ok_bits = torch.equal(out.view(torch.int32), ref.view(torch.int32))
    ok_vals = torch.equal(out, ref)
    nm = (out.view(torch.int32) != ref.view(torch.int32)).sum().item()
    ev_ok = torch.equal(ev.view(torch.int32), ev_bits)
    same_ptr = out.data_ptr() == state.data_ptr()
    results[f"{(B,L,D,K)}"] = {"numel": B*L*D, "bit_exact": ok_bits, "val_exact": ok_vals,
                               "mismatch_count": nm, "events_preserved_bits": ev_ok,
                               "same_storage": same_ptr}
print(json.dumps(results))