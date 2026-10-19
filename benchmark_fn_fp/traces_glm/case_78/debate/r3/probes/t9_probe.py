import json, torch, sys
sys.path.insert(0, '/root/cases/case_151')
import kernel

out = {}
for width in (64, 128):
    torch.manual_seed(1234 + width)
    x = torch.randint(0, 3329, (8, width), dtype=torch.int32, device='cuda')
    ref = kernel.reference(x)
    got = kernel.run(x)
    exact = bool((ref == got).all())
    mism = (ref != got).sum().item()
    out[f'n{width}_exact'] = exact
    out[f'n{width}_mismatch_count'] = mism
    if not exact:
        d = (got - ref) % 3329
        out[f'n{width}_sample_ref'] = ref[0][:8].tolist()
        out[f'n{width}_sample_got'] = got[0][:8].tolist()
print(json.dumps(out))