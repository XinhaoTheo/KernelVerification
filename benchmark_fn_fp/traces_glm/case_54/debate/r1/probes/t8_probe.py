import torch, json

dev = "cuda" if torch.cuda.is_available() else "cpu"
import sys
sys.path.insert(0, "/root/cases/case_127")
import kernel as K

# all finite FP16 values
bits = torch.arange(65536, dtype=torch.int32)
fp16_all = torch.tensor(bits.view(torch.uint16).numpy(), dtype=torch.uint16).view(torch.float16)
finite = torch.isfinite(fp16_all)
x_all = fp16_all[finite].to(dev)
print("num_finite_values:", x_all.numel(), "N<=131072:", x_all.numel() <= 131072)

out = K.run(x_all) if dev == "cuda" else None
if dev == "cuda":
    ref = torch.erf(x_all.to(torch.float64))
    err = (out.to(torch.float64) - ref).abs()
    i = int(err.argmax())
    print(json.dumps({
        "n_values": x_all.numel(),
        "max_abs_err": float(err.max()),
        "tolerance": 1e-6,
        "num_violations": int((err > 1e-6).sum()),
        "worst_x": float(x_all[i]),
        "worst_kernel_out": float(out[i]),
        "worst_ref_erf": float(ref[i]),
        "out_finite": bool(torch.isfinite(out).all().item()),
    }))
else:
    print(json.dumps({"error": "no cuda"}))
