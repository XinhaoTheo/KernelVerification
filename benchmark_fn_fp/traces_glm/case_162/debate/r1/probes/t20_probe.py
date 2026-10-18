import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("kernel", "/root/cases/case_162/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
results = {}
for N in (64, 128, 256):
    for offset in (3, 17, 257):
        B = 4
        g = torch.Generator(device="cpu").manual_seed(offset)
        base = torch.randn(offset + B * N, generator=g).to(torch.float16).cuda()
        vals = base[offset:offset + B*N].reshape(B, N)  # contiguous view, nonzero storage_offset
        assert vals.is_contiguous() and vals.storage_offset() == offset
        try:
            out = k.run(vals)
        except Exception as e:
            results[f"N{N}_off{offset}"] = {"exception": repr(e)}
            continue
        ref = torch.fft.rfft(vals.double().cpu(), dim=1)
        refc = torch.stack((ref.real, ref.imag), dim=-1).cuda()
        tol = 1e-4 + 1e-6*refc.abs()
        err = (out.double() - refc).abs()
        results[f"N{N}_off{offset}"] = {
            "max_abs_err": err.max().item(),
            "tol_violations": int((err > tol).sum()),
            "nonfinite_out": bool((~torch.isfinite(out)).any()),
            "input_preserved": bool(torch.equal(vals, base[offset:offset + B*N].reshape(B, N))),
        }
print(json.dumps(results))