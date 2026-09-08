
import subprocess, sys, json

script = r'''
import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("kmod", "/root/cases/case_03/kernel.py")
kmod = importlib.util.module_from_spec(spec); spec.loader.exec_module(kmod)

def build(M, K, N, group_size, zcode_mode, seed):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(M, K, generator=g)
    q = torch.randint(0, 16, (K, N), generator=g, dtype=torch.int32)
    packed = torch.zeros(K // 8, N, dtype=torch.int32)
    for j in range(8):
        packed |= (q[j::8] << (4 * j))
    ng = (K + group_size - 1) // group_size
    g_idx = (torch.arange(K) // group_size).to(torch.int32)
    if zcode_mode == "all15":
        zcode = torch.full((ng, N), 15, dtype=torch.int32)
    elif zcode_mode == "none15":
        zcode = torch.randint(0, 15, (ng, N), generator=g, dtype=torch.int32)
    else:  # mixed: every 3rd column is 15
        zcode = torch.randint(0, 15, (ng, N), generator=g, dtype=torch.int32)
        zcode[:, ::3] = 15
    qzeros = torch.zeros(ng, N // 8, dtype=torch.int32)
    for j in range(8):
        qzeros |= (zcode[:, j::8] << (4 * j))
    scales = (torch.rand(ng, N, generator=g) * 0.1 + 0.05)
    gl = g_idx.long()
    z_unwrapped = (zcode + 1).float()           # kernel semantics: zeros + 1, no mask
    z_wrapped = ((zcode + 1) & 15).float()      # upstream AutoGPTQ: (zeros + 1) & maxq
    z_raw = zcode.float()                       # problem.txt literal: zero, no +1
    ref_unwrapped = a @ ((q.float() - z_unwrapped[gl]) * scales[gl])
    ref_wrapped   = a @ ((q.float() - z_wrapped[gl]) * scales[gl])
    ref_raw       = a @ ((q.float() - z_raw[gl]) * scales[gl])
    return (a.cuda(), packed.cuda(), scales.cuda(), qzeros.cuda(), g_idx.cuda(),
            ref_unwrapped.cuda(), ref_wrapped.cuda(), ref_raw.cuda(),
            float(scales.max()), float(scales.min()), int((zcode == 15).sum()), zcode.numel())

def run(tag, M, K, N, group_size, zcode_mode, seed=0):
    a, packed, scales, qz, g_idx, r_un, r_wr, r_raw, smax, smin, n15, ntot = build(M, K, N, group_size, zcode_mode, seed)
    out = {"tag": tag, "M": M, "K": K, "N": N, "group_size": group_size,
           "zcode_mode": zcode_mode, "num_zero_codes_eq_15": n15, "num_zero_codes": ntot,
           "scale_min": smin, "scale_max": smax}
    try:
        c = kmod.gptq_matmul(a, packed, scales, qz, g_idx, bits=4)
        torch.cuda.synchronize()
        out["err_vs_unwrapped_zero_plus_1"] = float((c - r_un).abs().max())
        out["err_vs_wrapped_zero_plus_1_and_maxq"] = float((c - r_wr).abs().max())
        out["err_vs_raw_zero_no_plus1"] = float((c - r_raw).abs().max())
        out["ref_unwrapped_absmax"] = float(r_un.abs().max())
        out["ref_wrapped_absmax"] = float(r_wr.abs().max())
        out["ok"] = True
    except Exception as e:
        out["ok"] = False
        out["exception"] = repr(e)[:300]
    print(json.dumps(out), flush=True)

run("none15_control", 32, 128, 64, 64, "none15", seed=21)
run("all15", 32, 128, 64, 64, "all15", seed=22)
run("mixed15", 32, 128, 64, 64, "mixed", seed=23)
print(json.dumps({"tag": "done"}), flush=True)
'''

p = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=480)
res = []
for l in p.stdout.splitlines():
    l = l.strip()
    if not l:
        continue
    try:
        res.append(json.loads(l))
    except Exception:
        pass
print("STDERR_TAIL:", p.stderr[-1200:])
print(json.dumps({"returncode": p.returncode, "results": res,
                  "reached_done": any(r.get("tag") == "done" for r in res)}))
