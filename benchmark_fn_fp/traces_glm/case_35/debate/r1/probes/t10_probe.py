import json, os, sys, glob, importlib.util, traceback
import torch

RESULT = {
    "claim_id": "c2",
    "metric": "bitwise equality of blocked_sum(x)[0] against the exactly-computed total (int64 coefficient sum scaled by 2^-7, cast once to fp32); c2 confirmed iff any in-domain fill mismatches",
    "domain": "N=65536 fp32, every element k*2^-7 with |k|<=127, default block=1024",
    "static_bounds": {
        "max_per_block_partial_coeff": 127 * 1024,
        "max_running_total_coeff": 127 * 65536,
        "fp32_exact_int_coeff_limit": 2 ** 24,
        "per_block_partial_exact_in_fp32": 127 * 1024 < 2 ** 24,
        "running_total_exact_in_fp32": 127 * 65536 < 2 ** 24,
    },
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
    fills = build_fills()
    per_fill = {}
    any_mismatch = False
    for name, k in fills.items():
        x = (k.to(torch.float32) * SCALE).to(dev)
        coeff = int(k.to(torch.int64).sum().item())
        expected = torch.tensor(coeff, dtype=torch.float32) * SCALE
        ebits = int(expected.reshape(1).view(torch.int32).item())
        efloat = float(expected.item())
        vals = []
        bits_list = []
        for _ in range(3):
            o = blocked_sum(x)
            vals.append(float(o.item()))
            bits_list.append(int(o.view(torch.int32).item()))
        obits = bits_list[0]
        ov = vals[0]
        tbits = int(x.sum().reshape(1).view(torch.int32).item())
        per_fill[name] = {
            "coeff_sum": coeff,
            "abs_coeff_sum": abs(coeff),
            "abs_coeff_lt_2p24": bool(abs(coeff) < 2 ** 24),
            "expected_bits_hex": hex(ebits),
            "expected_float": efloat,
            "returned_bits_hex": hex(obits),
            "returned_float": ov,
            "bitwise_equal_to_exact_golden": obits == ebits,
            "repeated_call_bits_hex": [hex(b) for b in bits_list],
            "torch_fp32_sum_bits_hex": hex(tbits),
            "torch_fp32_sum_matches_golden": tbits == ebits,
        }
        if obits != ebits:
            any_mismatch = True
    RESULT["per_fill"] = per_fill
    RESULT["any_fill_mismatches_exact_golden"] = any_mismatch
    RESULT["observed_signal"] = ("returned bits differ from the exact golden bits on at least one in-domain fill"
                                 if any_mismatch else
                                 "returned bits equal the exact golden bits on every in-domain fill")
except Exception as e:
    RESULT["error"] = type(e).__name__ + ": " + str(e)
    RESULT["traceback"] = traceback.format_exc()[-1500:]
print(json.dumps(RESULT))