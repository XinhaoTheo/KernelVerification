import torch, json
import importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_107/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

def bits(t): return t.view(torch.int32).cpu()

results = {}
for (B,L,D,K) in [(2,8,4,2),(1,64,7,3),(3,5,16,5),(2,129,3,1)]:
    g = torch.Generator("cpu").manual_seed(42)
    st = torch.randn((B,L,D), generator=g).to("cuda")
    ev = torch.randn((B,K,D), generator=g).to("cuda")
    # plant signed zeros in both sources at known positions
    n_st = st.numel()
    stf = st.cpu().flatten(); evf = ev.cpu().flatten()
    for i in range(0, n_st, 7): stf[i] = -0.0
    for i in range(0, evf.numel(), 3): evf[i] = 0.0 if i%2 else -0.0
    st.copy_(stf.view_as(st)); ev.copy_(evf.view_as(ev))
    neg_zero_in_state = int((bits(st)==0x80000000).sum())
    neg_zero_in_ev = int((bits(ev)==0x80000000).sum())
    pre_state = st.clone(); pre_ev = ev.clone()
    ref = k.reference(pre_state, pre_ev)
    out = k.run(st, ev)
    torch.cuda.synchronize()
    exact = bool(torch.equal(bits(out.cpu()), bits(ref.cpu())))
    # check specifically that planted -0.0 survived
    nz_out_state_region = int((bits(out.cpu())==0x80000000).sum())
    results[f"B{B}L{L}D{D}K{K}"] = {
        "bitwise_equal_reference": exact,
        "neg_zero_planted_state": neg_zero_in_state,
        "neg_zero_planted_events": neg_zero_in_ev,
        "neg_zero_in_output": nz_out_state_region,
    }
print(json.dumps(results))