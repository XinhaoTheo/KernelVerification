import json, torch, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_106/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

# Single-program geometry: B=1, L=8, D=64 -> 512 elements <= 1024 (no cross-program race).
# Fill so that result elements map to -0.0 sources from both state tail and events.
B, L, D, K = 1, 8, 64, 3
state = torch.zeros((B, L, D), dtype=torch.float32, device="cuda")
state.view(torch.int32).fill_(0x80000000)  # -0.0 everywhere
ev = torch.zeros((B, K, D), dtype=torch.float32, device="cuda")
ev.view(torch.int32).fill_(0x80000000)    # -0.0 events
ref = k.reference(state, ev)
ref_bits = ref.view(torch.int32).clone()
out = k.run(state, ev)
torch.cuda.synchronize()
out_bits = out.view(torch.int32)
negzero_src = ((ref_bits == 0x80000000)).sum().item()
negzero_out = ((out_bits == 0x80000000)).sum().item()
poszero_out = ((out_bits == 0)).sum().item()
bit_exact = torch.equal(out_bits, ref_bits)
print(json.dumps({"geometry": [B,L,D,K], "negzero_in_expected": negzero_src,
                  "negzero_in_out": negzero_out, "poszero_in_out": poszero_out,
                  "bit_exact": bit_exact}))