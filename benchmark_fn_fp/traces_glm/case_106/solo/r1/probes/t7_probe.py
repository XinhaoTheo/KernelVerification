import sys, torch
sys.path.insert(0, "/root/cases/case_106")
import kernel

results = []
cases = [(4,2048,1024,2),(1,4096,1024,128),(2,64,7,5),(1,2,1,2),(3,100,33,64),(2,4096,1,1),(1,3,1024,3)]
for (B,L,D,K) in cases:
    for seed in (0,1):
        g = torch.Generator(device="cpu").manual_seed(seed)
        state = torch.randn((B,L,D), generator=g).cuda()
        ev = torch.randn((B,K,D), generator=g).cuda()
        ref = kernel.reference(state.clone(), ev)
        out = kernel.run(state, ev)
        torch.cuda.synchronize()
        ok_ret = out is state
        same_storage = out.data_ptr() == state.data_ptr()
        ev_ok = torch.equal(ev, ev.clone())
        ok_bits = torch.equal(state.view(torch.int32), ref.view(torch.int32).cuda())
        # repeated call on resulting state
        ref2 = kernel.reference(state.clone(), ev)
        out2 = kernel.run(state, ev)
        torch.cuda.synchronize()
        ok2 = torch.equal(state.view(torch.int32), ref2.view(torch.int32).cuda())
        results.append(dict(B=B,L=L,D=D,K=K,seed=seed,ok_ret=ok_ret,same_storage=same_storage,ev_ok=ev_ok,ok_bits=ok_bits,ok_repeat=ok2))
import json
print(json.dumps(results))
bad = [r for r in results if not (r["ok_ret"] and r["same_storage"] and r["ev_ok"] and r["ok_bits"] and r["ok_repeat"])]
print("FAILURES:", len(bad))
