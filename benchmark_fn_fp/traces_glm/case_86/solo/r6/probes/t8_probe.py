import sys, json
sys.path.insert(0, "/root/cases/case_159")
import torch, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_159/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev = "cuda"
g = torch.Generator(device="cpu").manual_seed(1234)
fails = []; tests = 0
for N in range(1, 33):
    for B in (1,):
        for mode in ("rand", "extreme", "ones", "zero-mat"):
            matrix = torch.randint(-1, 2, (B, N, 2, 2), generator=g, dtype=torch.int8)
            if mode == "ones": matrix[:] = 1
            if mode == "zero-mat": matrix[:, :, 0, 1] = 0
            if mode == "extreme":
                bias = torch.full((B, N, 2), -32768, dtype=torch.int16); bias[..., ::2] = 32767
                initial = torch.full((B, 2), 32767, dtype=torch.int16)
            else:
                bias = torch.randint(-32768, 32768, (B, N, 2), generator=g, dtype=torch.int16)
                initial = torch.randint(-32768, 32768, (B, 2), generator=g, dtype=torch.int16)
            m, b_, i_ = matrix.to(dev), bias.to(dev), initial.to(dev)
            out = k.run(m, b_, i_)
            ref = k.reference(m.cpu(), b_.cpu(), i_.cpu())
            if not torch.equal(out.cpu(), ref):
                bad = (out.cpu() != ref)
                first = [tuple(t) for t in bad.nonzero()[:3].tolist()]
                fails.append((N, mode, first, out.flatten()[bad.flatten()][0].item(), ref.flatten()[bad.flatten()][0].item()))
            tests += 1
print(json.dumps({"metric": "exact equality", "tests": tests, "fails_N_mode": fails}))
