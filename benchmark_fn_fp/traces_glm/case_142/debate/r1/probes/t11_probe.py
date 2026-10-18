import json, torch, sys
sys.path.insert(0, "/root/cases/case_142")
import kernel

res = {}
for N in (32, 64):
    B = 4; off = 5
    try:
        base = torch.randint(-32768, 32768, (B*N + off,), dtype=torch.int32).cuda().to(torch.int16)
        x = base[off:off + B*N].view(B, N)
        assert x.is_contiguous() and x.storage_offset() == off
        out = kernel.run(x).cpu()
        ref = kernel.reference(x).cpu()
        res[f"N{N}_off{off}"] = {"mismatch": int((out != ref).sum().item()), "total": B*N,
                                 "storage_offset": int(x.storage_offset())}
    except Exception as e:
        res[f"N{N}_off{off}"] = {"error": repr(e)}
print(json.dumps(res))
