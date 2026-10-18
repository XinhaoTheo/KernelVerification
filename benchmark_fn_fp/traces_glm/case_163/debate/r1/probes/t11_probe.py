import torch, sys, os, json
sys.path.insert(0, os.environ.get("CASE_DIR", "/root/cases/case_163"))
import kernel
dev = "cuda"
res = []
for N in (64, 128, 256):
    # nonzero storage offset: slice of a larger buffer, contiguous view
    big = torch.randn((300, N + 33), generator=torch.Generator().manual_seed(2)).to(torch.float16).to(dev)
    view = big[5:5+8, 3:3+N]  # contiguous? rows stride N+33 -> not contiguous; need contiguous slice
    # use a flat storage slice instead
    flat = torch.randn(4096, generator=torch.Generator().manual_seed(3)).to(torch.float16).to(dev)
    sub = flat[17:17 + 8 * N].view(8, N)  # contiguous, nonzero offset
    sub_c = sub.clone()
    before = flat.clone()
    out = kernel.run(sub)
    torch.cuda.synchronize()
    storage_preserved = torch.equal(flat, before)
    out_sep = out.data_ptr() != flat.data_ptr() and out.storage_offset() == 0 and out.is_contiguous()
    # correctness under offset too
    ref = torch.fft.rfft(sub.to(torch.float64), dim=1, norm="backward")
    ref = torch.stack((ref.real, ref.imag), dim=-1)
    err = (out.double() - ref).abs()
    tol = 1e-4 + 1e-6 * ref.abs()
    ok = int((err > tol).sum().item()) == 0
    res.append({"N": N, "storage_preserved": bool(storage_preserved),
                "out_sep": bool(out_sep), "correct": bool(ok),
                "out_shape": list(out.shape), "out_dtype": str(out.dtype)})
print(json.dumps({"claim": "c3", "results": res}))