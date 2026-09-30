import json, os, sys, glob, importlib.util, traceback
import torch

RESULT = {
    "claim_id": "c1",
    "metric": "count of distinct int32 bit patterns of blocked_sum(x)[0] across repeated calls on the same in-domain tensor; c1 confirmed iff any fill yields more than 1 distinct pattern",
    "domain": "N=65536 fp32, every element k*2^-7 with |k|<=127, default block=1024 (64 programs, one fp32 atomic_add each)",
}

def find_kernel_path():
    for c in ["/root/cases/case_35/kernel.py",
              os.path.join(os.getcwd(), "kernel.py"),
              "kernel.py",
              os.path.join(os.getcwd(), "case_35", "kernel.py")]:
        if os.path.isfile(c):
            return c
    for h in sorted(glob.glob("/root/cases/*/kernel.py")):
        if "case_35" in h:
            return h
    return None

def collect_ptx(obj, depth=0, acc=None):
    if acc is None:
        acc = []
    if depth > 5:
        return acc
    asm = getattr(obj, "asm", None)
    if isinstance(asm, dict) and "ptx" in asm:
        acc.append(asm["ptx"])
        return acc
    if isinstance(obj, dict):
        for v in obj.values():
            collect_ptx(v, depth + 1, acc)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            collect_ptx(v, depth + 1, acc)
    return acc

def build_fills():
    N = 65536
    f = {}
    f["all_plus_127"] = torch.full((N,), 127, dtype=torch.int16)
    f["all_minus_127"] = torch.full((N,), -127, dtype=torch.int16)
    f["all_zero"] = torch.zeros(N, dtype=torch.int16)
    hf = torch.empty(N, dtype=torch.int16)
    hf[: N // 2] = 127
    hf[N // 2 :] = -127
    f["half_plus_then_half_minus"] = hf
    ab = torch.empty(N, dtype=torch.int16)
    abv = ab.view(64, 1024)
    abv[0::2, :] = 127
    abv[1::2, :] = -127
    f["alternating_blocks_pm127"] = ab
    cb = torch.empty(N, dtype=torch.int16)
    cbv = cb.view(64, 1024)
    cbv[:, :512] = 127
    cbv[:, 512:] = -127
    f["blockwise_half_cancel"] = cb
    kk = torch.randint(1, 128, (N // 2,), dtype=torch.int16,
                       generator=torch.Generator().manual_seed(7))
    pc = torch.empty(N, dtype=torch.int16)
    pc[0::2] = kk
    pc[1::2] = -kk
    f["paired_random_cancel"] = pc
    f["rand_k_seed1234"] = torch.randint(-127, 128, (N,), dtype=torch.int16,
                                         generator=torch.Generator().manual_seed(1234))
    f["rand_k_seed999"] = torch.randint(-127, 128, (N,), dtype=torch.int16,
                                        generator=torch.Generator().manual_seed(999))
    return f

try:
    p = find_kernel_path()
    if p is None:
        raise RuntimeError("kernel.py not found in probe environment")
    spec = importlib.util.spec_from_file_location("case35_kernel_mod", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    blocked_sum = mod.blocked_sum
    RESULT["kernel_path"] = p
    import triton
    RESULT["triton_version"] = triton.__version__
    RESULT["torch_version"] = torch.__version__
    if not torch.cuda.is_available():
        RESULT["gpu_available"] = False
        RESULT["error"] = "no CUDA device available; Triton kernel cannot be executed"
        print(json.dumps(RESULT))
        sys.exit(0)
    RESULT["gpu_available"] = True
    RESULT["gpu_name"] = torch.cuda.get_device_name(0)
    RESULT["gpu_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
    dev = torch.device("cuda")
    N = 65536
    SCALE = 2.0 ** -7
    REPS = 200
    fills = build_fills()
    per_fill = {}
    mismatch_any = False
    for name, k in fills.items():
        x = (k.to(torch.float32) * SCALE).to(dev)
        ongrid = bool(torch.equal((x.to(torch.float64) * 128.0).round().to(torch.int16).cpu(), k))
        pats = []
        first_val = None
        for i in range(REPS):
            out = blocked_sum(x)
            if i == 0:
                first_val = float(out.item())
            pats.append(int(out.view(torch.int32).item()))
        d = sorted(set(pats))
        per_fill[name] = {
            "on_grid_verified": ongrid,
            "reps": REPS,
            "n_distinct_bit_patterns": len(d),
            "distinct_patterns_hex": [hex(v) for v in d[:6]],
            "first_value_float": first_val,
            "bitwise_repeatable_across_calls": len(d) == 1,
        }
        if len(d) > 1:
            mismatch_any = True
    RESULT["per_fill"] = per_fill
    RESULT["total_kernel_calls"] = REPS * len(fills)
    RESULT["any_run_to_run_bit_mismatch"] = mismatch_any
    try:
        ptxs = []
        for attr in ("device_caches", "cache"):
            c = getattr(mod._blocked_sum_kernel, attr, None)
            if c is not None:
                ptxs.extend(collect_ptx(c))
        ptxs = list(dict.fromkeys(ptxs))
        if ptxs:
            atom_lines = sorted({ln.strip() for ptx in ptxs for ln in ptx.splitlines()
                                 if "atom" in ln.lower()})
            RESULT["ptx_modules_found"] = len(ptxs)
            RESULT["ptx_atomic_lines"] = atom_lines[:12]
            RESULT["ptx_atomic_lowering_native_add"] = any(
                ".add." in ln.lower() and "cas" not in ln.lower() for ln in atom_lines)
            RESULT["ptx_atomic_lowering_cas_loop"] = any("cas" in ln.lower() for ln in atom_lines)
        else:
            RESULT["ptx_modules_found"] = 0
            RESULT["ptx_note"] = "compiled PTX not reachable via kernel cache attributes; atomic lowering not inspected"
    except Exception as e:
        RESULT["ptx_inspection_error"] = type(e).__name__ + ": " + str(e)
    RESULT["observed_signal"] = ("run-to-run bit mismatch observed on at least one in-domain fill"
                                 if mismatch_any else
                                 "all repeated calls returned bitwise-identical output on every in-domain fill")
except Exception as e:
    RESULT["error"] = type(e).__name__ + ": " + str(e)
    RESULT["traceback"] = traceback.format_exc()[-1500:]
print(json.dumps(RESULT))