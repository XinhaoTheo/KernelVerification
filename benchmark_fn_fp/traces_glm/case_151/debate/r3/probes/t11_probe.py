import json, torch, sys
sys.path.insert(0, '/root/cases/case_151')
import kernel

out = {}
width = 128  # LOG_N=7 odd, exercises the tail path
for seed in (0, 1, 2):
    torch.manual_seed(seed)
    x = torch.randint(0, 3329, (16, width), dtype=torch.int32, device='cuda')
    ref = kernel.reference(x)
    got = kernel.run(x)
    exact = bool((ref == got).all())
    out[f'seed{seed}_exact'] = exact
    out[f'seed{seed}_mismatch_count'] = (ref != got).sum().item()
    if not exact:
        out[f'seed{seed}_first_ref'] = ref[0][:4].tolist()
        out[f'seed{seed}_first_got'] = got[0][:4].tolist()
        break
print(json.dumps(out))