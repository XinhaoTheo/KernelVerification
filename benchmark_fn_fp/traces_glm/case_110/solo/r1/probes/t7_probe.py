
import torch, json, sys
sys.path.insert(0, "/root/cases/case_110")
import kernel

def bits(t):
    return t.view(torch.uint32)

torch.cuda.init()
failures = []
tested = 0

configs = [
    (1, 2, 1, 1), (1, 2, 1, 2),          # K=L, K=min(L,128)
    (4, 2, 1024, 2),
    (1, 4096, 1024, 128),                 # max size, K=128
    (2, 2048, 1024, 2),                   # example config
    (3, 100, 7, 37), (1, 129, 3, 128),    # K=128 but L=129 -> K<L
    (2, 3, 5, 1), (4, 17, 13, 16),
    (1, 4096, 1, 128),
]

for (B, L, D, K) in configs:
    gen = torch.Generator("cpu").manual_seed(B*1000+L+D+K)
    state = torch.randn((B, L, D), generator=gen)
    ev = torch.randn((B, K, D), generator=gen)
    # inject signed zeros / specials
    state.view(-1)[::7] = 0.0
    state.view(-1)[::11] = -0.0
    ev.view(-1)[::5] = -0.0
    ev.view(-1)[::9] = 0.0
    state = state.to("cuda"); ev = ev.to("cuda")
    pre_state = state.clone()
    pre_ev = ev.clone()
    expected = kernel.reference(pre_state, pre_ev)
    out = kernel.run(state, ev)
    torch.cuda.synchronize()
    tested += 1
    ok_same = out is state
    ok_bits = torch.equal(bits(state), bits(expected))
    ok_ev = torch.equal(bits(ev), bits(pre_ev))
    if not (ok_same and ok_bits and ok_ev):
        mism = (bits(state) != bits(expected)).sum().item()
        failures.append(dict(cfg=[B,L,D,K], same_storage=ok_same, bitwise=ok_bits,
                             mismatch_count=mism, ev_preserved=ok_ev))
        continue
    # repeated calls on mutated state
    for step in range(3):
        g2 = torch.Generator("cpu").manual_seed(step+1)
        ev2 = torch.randn((B, max(1, min(L, K+1, 128)), D), generator=g2).to("cuda")
        pre2 = state.clone(); pe2 = ev2.clone()
        exp2 = kernel.reference(pre2, pe2)
        out2 = kernel.run(state, ev2)
        torch.cuda.synchronize()
        tested += 1
        if not (out2 is state and torch.equal(bits(state), bits(exp2)) and torch.equal(bits(ev2), bits(pe2))):
            mism = (bits(state) != bits(exp2)).sum().item()
            failures.append(dict(cfg=[B,L,D,K], repeat=step, bitwise=torch.equal(bits(state), bits(exp2)),
                                 mismatch_count=mism, ev_preserved=torch.equal(bits(ev2), bits(pe2))))
            break

print(json.dumps({"tested_calls": tested, "failures": failures, "verdict_ok": not failures}))
