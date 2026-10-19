import json, torch, sys
sys.path.insert(0, "/root/cases/case_142")
import kernel

res = {}
for N in (32, 64, 128):
    for B in (1, 5):
        g = torch.Generator("cpu").manual_seed(1000 + N + B)
        x = torch.randint(-32768, 32768, (B, N), generator=g, dtype=torch.int32).cuda().to(torch.int16)
        try:
            out = kernel.run(x).cpu()
            ref = kernel.reference(x.cpu()).cpu() if False else kernel.reference(x).cpu()
            mismatch = int((out != ref).sum().item())
            # check if mismatches are pure sign flips or exact equal
            sign_only = mismatch > 0 and torch.equal(out, -ref)
            res[f"N{N}_B{B}"] = {"mismatch": mismatch, "total": B*N, "sign_flip_only": bool(sign_only)}
        except Exception as e:
            res[f"N{N}_B{B}"] = {"error": repr(e)}
print(json.dumps(res))
