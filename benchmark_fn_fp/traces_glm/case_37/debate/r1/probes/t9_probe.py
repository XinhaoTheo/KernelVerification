import json, math, importlib.util, traceback
import numpy as np
import torch

result = {"claim": "c2", "probe": "per-branch quantization error decomposition on fixed seed-921000 workload"}
try:
    spec = importlib.util.spec_from_file_location("case_b_kernel", "/root/cases/case_b/kernel.py")
    kmod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kmod)

    gpu = torch.cuda.is_available()
    result["gpu_available"] = bool(gpu)
    device = "cuda" if gpu else "cpu"
    if gpu:
        result["gpu_device"] = torch.cuda.get_device_name(0)

    x, A, B = kmod.make_inputs(device=device)

    # fp64 references per branch and combined
    xd, Ad, Bd = x.double(), A.double(), B.double()
    refA = Ad @ xd
    refB = Bd @ xd
    ref = refA + refB

    # fp32 emulation of the kernel's per-row quantization (kernel.py lines 12-17)
    sa = A.abs().amax(dim=1, keepdim=True) / 7.0
    sb = B.abs().amax(dim=1, keepdim=True) / 7.0
    qa = torch.clamp(torch.floor(A / sa + 0.5), -7.0, 7.0)
    qb = torch.clamp(torch.floor(B / sb + 0.5), -7.0, 7.0)
    Aq, Bq = qa * sa, qb * sb

    # per-branch outputs: fp64 accumulation of fp32-quantized matrices (structural error),
    # and fp32 accumulation (kernel-like)
    outA64 = Aq.double() @ xd
    outB64 = Bq.double() @ xd
    outA32 = (Aq * x[None, :]).sum(dim=1)
    outB32 = (Bq * x[None, :]).sum(dim=1)

    errA = outA64 - refA
    errB = outB64 - refB
    errA32 = outA32.double() - refA
    errB32 = outB32.double() - refB

    nA = float(errA.norm()); nB = float(errB.norm())
    nA32 = float(errA32.norm()); nB32 = float(errB32.norm())
    refA_n = float(refA.norm()); refB_n = float(refB.norm())
    comb = errA + errB
    comb_n = float(comb.norm())
    quad_n = math.sqrt(nA ** 2 + nB ** 2)
    cov = float((errA * errB).mean() - errA.mean() * errB.mean())
    corr = cov / (float(errA.std(unbiased=False)) * float(errB.std(unbiased=False)))
    cosine = float(torch.dot(errA, errB) / (nA * nB))

    result.update({
        "errA_norm": nA, "errB_norm": nB,
        "errA32_norm": nA32, "errB32_norm": nB32,
        "refA_norm": refA_n, "refB_norm": refB_n,
        "rel_errA": nA / refA_n, "rel_errB": nB / refB_n,
        "combined_err_norm_structural": comb_n,
        "quadrature_norm": quad_n,
        "quadrature_ratio": comb_n / quad_n,
        "errA_errB_correlation": corr,
        "errA_errB_cosine": cosine,
        "errA_mean": float(errA.mean()), "errB_mean": float(errB.mean()),
        "errA_std": float(errA.std()), "errB_std": float(errB.std()),
        "errA_absmax": float(errA.abs().max()), "errB_absmax": float(errB.abs().max()),
        "fp32acc_minus_fp64acc_norm_a": float((outA32.double() - outA64).norm()),
        "fp32acc_minus_fp64acc_norm_b": float((outB32.double() - outB64).norm()),
    })

    # fidelity: emulated combined output vs actual kernel output
    if gpu:
        try:
            out = kmod.run(x, A, B)
            torch.cuda.synchronize()
            err_kernel = out.double() - ref
            result.update({
                "kernel_ran": True,
                "kernel_combined_err_norm": float(err_kernel.norm()),
                "kernel_vs_structural_ratio": float(err_kernel.norm()) / comb_n,
                "emu32_vs_kernel_max_absdiff": float(((outA32 + outB32).double() - out.double()).abs().max()),
            })
        except Exception as e:
            result["kernel_ran"] = False
            result["kernel_error"] = f"{type(e).__name__}: {e}"
    else:
        result["kernel_ran"] = False
        result["kernel_error"] = "cuda unavailable; kernel not executed"
except Exception as e:
    result["probe_error"] = f"{type(e).__name__}: {e}"
    result["traceback"] = traceback.format_exc()[-2000:]

print(json.dumps(result))