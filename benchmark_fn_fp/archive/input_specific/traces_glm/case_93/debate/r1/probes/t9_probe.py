import os, sys, json, traceback, glob
import numpy as np

def find_kernel():
    cands = [
        "/root/cases/case_a/kernel.py",
        os.path.join(os.getcwd(), "kernel.py"),
        os.path.join(os.getcwd(), "case_a", "kernel.py"),
        os.path.join(os.getcwd(), "..", "kernel.py"),
    ]
    for c in cands:
        if os.path.exists(c):
            return c
    hits = sorted(glob.glob("/root/cases/*/kernel.py"))
    if hits:
        return hits[0]
    raise FileNotFoundError("kernel.py not found; tried %r" % (cands,))

def load_module(kpath):
    # Normal import when torch+triton are available; otherwise stub triton so
    # that make_inputs (pure NumPy/torch) can still be used on CPU.
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("kernel_mod_c2", kpath)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod, "normal_import"
    except Exception:
        import types
        tl = types.ModuleType("triton.language")
        tl.constexpr = object()
        tr = types.ModuleType("triton")
        class _J:
            def __call__(self, fn):
                return fn
        tr.jit = _J()
        tr.language = tl
        sys.modules["triton"] = tr
        sys.modules["triton.language"] = tl
        with open(kpath) as f:
            src = f.read()
        ns = {"__name__": "kernel_src_c2"}
        exec(compile(src, kpath, "exec"), ns)
        return ns, "stub_triton"

def quant_scheme(M, half_up=True):
    # Exact scheme from kernel.py lines 12-17:
    # s = max(|row|)/7; codes = clip(floor(v/s + 0.5), -7, 7); dequant = codes*s
    s = np.max(np.abs(M), axis=1, keepdims=True) / 7.0
    if half_up:
        codes = np.clip(np.floor(M / s + 0.5), -7.0, 7.0)
    else:
        codes = np.clip(np.round(M / s), -7.0, 7.0)  # round-half-even, informative
    return codes * s, s, codes

def quant_scheme_fp32(M):
    M32 = M.astype(np.float32)
    s32 = np.max(np.abs(M32), axis=1, keepdims=True) / np.float32(7.0)
    c32 = np.clip(np.floor(M32 / s32 + np.float32(0.5)), np.float32(-7.0), np.float32(7.0))
    return (c32 * s32).astype(np.float64), c32

def main():
    kpath = find_kernel()
    mod, mode = load_module(kpath)
    make_inputs = mod["make_inputs"] if isinstance(mod, dict) else mod.make_inputs
    import torch
    x_t, a_t, b_t = make_inputs("cpu")
    xd = x_t.numpy().astype(np.float64)
    ad = a_t.numpy().astype(np.float64)
    bd = b_t.numpy().astype(np.float64)

    ref = ad @ xd + bd @ xd
    refa = ad @ xd
    refb = bd @ xd
    refn = float(np.linalg.norm(ref))
    floor_den = 0.001 * (64 ** 0.5)
    den = max(refn, floor_den)

    qa, sa, ca = quant_scheme(ad)
    qb, sb, cb = quant_scheme(bd)
    emul = qa @ xd + qb @ xd
    E_emul = float(np.linalg.norm(emul - ref) / den)

    qa_he, _, _ = quant_scheme(ad, half_up=False)
    qb_he, _, _ = quant_scheme(bd, half_up=False)
    E_he = float(np.linalg.norm((qa_he @ xd + qb_he @ xd) - ref) / den)

    qa32, ca32 = quant_scheme_fp32(ad)
    qb32, cb32 = quant_scheme_fp32(bd)
    E_fp32 = float(np.linalg.norm((qa32 @ xd + qb32 @ xd) - ref) / den)
    code_flips = int(np.count_nonzero(ca != ca32) + np.count_nonzero(cb != cb32))

    res = {
        "probe": "c2_float64_scheme_emulation",
        "metric": "E_emul = ||emulation - float64 unquantized ref||_2 / max(||ref||_2, 0.001*sqrt(64))",
        "metric_reason": "isolates inherent quantization-scheme lossiness under ideal float64 arithmetic from Triton implementation deviation; contract threshold E <= 0.1 on the same fixed workload",
        "kernel_path": kpath,
        "load_mode": mode,
        "scheme": "per-row s=max|row|/7; codes=clip(floor(v/s+0.5),-7,7); dequant=code*s; y=(qa@x)+(qb@x)",
        "E_emul_float64": E_emul,
        "E_emul_half_even": E_he,
        "E_emul_fp32arith": E_fp32,
        "code_flips_fp32_vs_fp64": code_flips,
        "contract_threshold": 0.1,
        "scheme_too_lossy_for_budget": bool(E_emul > 0.1),
        "ref_l2": refn,
        "denominator": den,
        "rms_row_err_emul": float(np.sqrt(np.mean((emul - ref) ** 2))),
        "max_abs_row_err_emul": float(np.max(np.abs(emul - ref))),
        "branch_rel_err_a": float(np.linalg.norm(qa @ xd - refa) / max(np.linalg.norm(refa), floor_den)),
        "branch_rel_err_b": float(np.linalg.norm(qb @ xd - refb) / max(np.linalg.norm(refb), floor_den)),
        "saturation_codes_a": int(np.count_nonzero(np.abs(ca) == 7)),
        "saturation_codes_b": int(np.count_nonzero(np.abs(cb) == 7)),
        "zero_codes_a": int(np.count_nonzero(ca == 0)),
        "zero_codes_b": int(np.count_nonzero(cb == 0)),
        "scale_a_min_mean_max": [float(np.min(sa)), float(np.mean(sa)), float(np.max(sa))],
        "scale_b_min_mean_max": [float(np.min(sb)), float(np.mean(sb)), float(np.max(sb))],
    }

    # Secondary: real Triton kernel vs the float64 emulation (gap > ~0.02 would
    # indicate an implementation bug on top of scheme lossiness).
    cmp_info = {"kernel_run": "not_attempted"}
    try:
        if torch.cuda.is_available():
            if mode == "normal_import":
                gmod = mod
            else:
                sys.modules.pop("triton", None)
                sys.modules.pop("triton.language", None)
                import importlib.util
                spec = importlib.util.spec_from_file_location("kernel_mod_c2_gpu", kpath)
                gmod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(gmod)
            xg, ag, bg = gmod.make_inputs("cuda")
            same_inputs = bool(
                torch.equal(x_t, xg.detach().cpu())
                and torch.equal(a_t, ag.detach().cpu())
                and torch.equal(b_t, bg.detach().cpu())
            )
            out = gmod.run(xg, ag, bg)
            torch.cuda.synchronize()
            outd = out.detach().cpu().numpy().astype(np.float64)
            E_kernel = float(np.linalg.norm(outd - ref) / den)
            cmp_info = {
                "kernel_run": "ok",
                "inputs_identical_cpu_cuda": same_inputs,
                "E_kernel": E_kernel,
                "abs_gap_E_kernel_vs_emul": abs(E_kernel - E_emul),
                "gap_threshold_for_impl_bug": 0.02,
                "max_elem_gap_kernel_vs_emul": float(np.max(np.abs(outd - emul))),
                "rms_elem_gap_kernel_vs_emul": float(np.sqrt(np.mean((outd - emul) ** 2))),
            }
    except Exception as e:
        cmp_info = {"kernel_run": "failed", "error": repr(e), "traceback": traceback.format_exc()}
    res["kernel_comparison"] = cmp_info
    print(json.dumps(res))

try:
    main()
except Exception as e:
    print(json.dumps({"probe": "c2_float64_scheme_emulation", "error": repr(e), "traceback": traceback.format_exc()}))