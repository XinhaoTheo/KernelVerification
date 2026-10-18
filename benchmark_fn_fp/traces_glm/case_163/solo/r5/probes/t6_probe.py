
import json, torch, importlib.util
spec = importlib.util.spec_from_file_location("kernel", "/root/cases/case_163/kernel.py")
kernel = importlib.util.module_from_spec(spec); spec.loader.exec_module(kernel)
dev = "cuda"
results = {}
def check(tag, values):
    before = values.clone()
    out = kernel.run(values)
    ref = kernel.reference(values).to(torch.float64)
    err = (out.to(torch.float64) - ref).abs()
    tol = 1e-4 + 1e-6*ref.abs()
    ok_shape = tuple(out.shape) == (values.shape[0], values.shape[1]//2+1, 2)
    preserved = torch.equal(before, values)
    results[tag] = {
        "max_err": float(err.max()),
        "n_violations": int((err > tol).sum()),
        "finite": bool(torch.isfinite(out).all()),
        "shape_ok": bool(ok_shape),
        "contig": out.is_contiguous(),
        "dtype": str(out.dtype),
        "input_preserved": bool(preserved),
        "fresh_storage": out.data_ptr() != values.data_ptr(),
    }

for n in (64, 128, 256):
    g = torch.Generator(device="cpu").manual_seed(n)
    v = torch.randn((5, n), generator=g).to(torch.float16).to(dev)
    check(f"randn_N{n}", v)
    # alternating signs at max magnitude
    v = torch.where(torch.arange(n, device=dev) % 2 == 0,
                    torch.full((n,), 65504., device=dev),
                    torch.full((n,), -65504., device=dev)).to(torch.float16).unsqueeze(0).repeat(3,1)
    check(f"altmax_N{n}", v)
    # subnormals
    v = (torch.randn((3, n), generator=torch.Generator().manual_seed(99)) * 6e-8).to(torch.float16).to(dev)
    check(f"subnorm_N{n}", v)
    # single spike (impulse)
    v = torch.zeros((2, n), dtype=torch.float16, device=dev); v[0,7] = 65504.; v[1,3] = -1.0
    check(f"impulse_N{n}", v)
    # all signed zeros
    v = torch.zeros((2, n), dtype=torch.float16, device=dev); v[1,:] = -0.0
    check(f"zeros_N{n}", v)

# storage offset: legal per contract
base = torch.randn((4, 128*2+4), generator=torch.Generator().manual_seed(5)).to(torch.float16).to(dev)
v = base[1:3, 4:4+128].contiguous()
check("offset_like_N128", v)
v2 = base.as_strided((2,128), (128,1), storage_offset=7)
check("offset_view_N128", v2)

print(json.dumps(results, indent=1))
print("ANY_FAIL", any((not r["finite"]) or r["n_violations"]>0 or not r["shape_ok"] or not r["contig"] or not r["input_preserved"] or not r["fresh_storage"] for r in results.values()))
