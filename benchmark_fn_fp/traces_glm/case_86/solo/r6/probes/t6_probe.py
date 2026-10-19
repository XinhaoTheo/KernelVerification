import sys, json
sys.path.insert(0, "/root/cases/case_159")
import torch, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_159/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev = "cuda"
g = torch.Generator(device="cpu").manual_seed(1234)
fails = []
tests = 0
maxabs = 0
for N in range(1, 33):
    for B in (1, 3, 64):
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
                fails.append((N, B, mode))
                d = (out.cpu() - ref).abs().max().item()
                maxabs = max(maxabs, d)
            tests += 1
# contiguous view with nonzero storage offset
base = torch.zeros(64, 64, 2, 2, dtype=torch.int8)
mv = base[:32, 5:37]
mv.random_(-1, 2)
biasv = torch.zeros(64, 64, 2, dtype=torch.int16); biasv[:32, 5:37].random_(-32768, 32768)
iv = torch.zeros(64, 2, dtype=torch.int16); iv.random_(-32768, 32768)
out = k.run(mv.cuda(), biasv[:32, 5:37].cuda(), iv[:32].cuda())
ref = k.reference(mv.cpu(), biasv[:32, 5:37].cpu(), iv[:32].cpu())
view_ok = torch.equal(out.cpu(), ref)
print(json.dumps({"metric": "exact integer equality vs reference", "tests": tests, "failures": fails, "max_abs_diff_on_fail": maxabs, "view_with_offset_ok": view_ok}))
