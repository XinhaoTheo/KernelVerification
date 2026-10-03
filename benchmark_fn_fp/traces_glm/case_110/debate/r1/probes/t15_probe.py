import sys, torch, json
sys.path.insert(0, "/root/cases/case_110")
import kernel

dev = "cuda" if torch.cuda.is_available() else "cpu"
cases = [ (1,2,1,1), (1,2,1,2), (2,2,3,1), (1,128,5,128), (1,3,1,3),
          (3,129,7,128), (1,5,1024,2), (2,1025,3,17), (1,2048,1024,2) ]
fails = []
for (B,L,D,K) in cases:
    g = torch.Generator().manual_seed(B*1000+L+D+K)
    state = torch.randn((B,L,D), generator=g).to(dev)
    events = torch.randn((B,K,D), generator=g).to(dev)
    expected_bits = kernel.reference(state, events).view(torch.int32).clone()
    out = kernel.run(state, events)
    torch.cuda.synchronize()
    got_bits = state.view(torch.int32)
    mm = (got_bits != expected_bits).sum().item()
    same_obj = out is state
    fails.append({"B":B,"L":L,"D":D,"K":K,"bit_mismatches":mm,
                  "numel":state.numel(),"numel_mod_1024":state.numel()%1024,
                  "returned_same_object":same_obj})
    if mm:
        idx = (got_bits != expected_bits).nonzero()[:5].tolist()
        fails[-1]["first_mismatch_indices"] = idx
# chained calls on same state
g = torch.Generator().manual_seed(7)
state = torch.randn((2,10,4), generator=g).to(dev)
all_ok = True
for step in range(3):
    K = [1,5,10][step]
    ev = torch.randn((2,K,4), generator=g).to(dev)
    ref = kernel.reference(state, ev)
    kernel.run(state, ev)
    torch.cuda.synchronize()
    if not (state.view(torch.int32) == ref.view(torch.int32)).all().item():
        all_ok = False
result = {"cases": fails, "all_edge_cases_pass": all(f["bit_mismatches"]==0 for f in fails),
          "chained_calls_pass": all_ok}
print(json.dumps(result))
