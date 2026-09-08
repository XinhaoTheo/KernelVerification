
import subprocess, sys, json

script = r'''
import torch, json, math, importlib.util, traceback
spec = importlib.util.spec_from_file_location("kmod", "/root/cases/case_03/kernel.py")
kmod = importlib.util.module_from_spec(spec); spec.loader.exec_module(kmod)

def build(M, K, N, group_size, seed):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(M, K, generator=g)
    q = torch.randint(0, 16, (K, N), generator=g, dtype=torch.int32)
    packed = torch.zeros(K // 8, N, dtype=torch.int32)
    for j in range(8):
        packed |= (q[j::8] << (4 * j))
    ng = (K + group_size - 1) // group_size
    g_idx = (torch.arange(K) // group_size).to(torch.int32)
    # zero codes restricted to 0..14 so the (zeros+1)&maxq wrap question cannot confound
    zcode = torch.randint(0, 15, (ng, N), generator=g, dtype=torch.int32)
    qzeros = torch.zeros(ng, N // 8, dtype=torch.int32)
    for j in range(8):
        qzeros |= (zcode[:, j::8] << (4 * j))
    scales = (torch.rand(ng, N, generator=g) * 0.1 + 0.05)
    zeros_eff = (zcode + 1).float()
    deq = (q.float() - zeros_eff[g_idx.long()]) * scales[g_idx.long()]
    ref = a @ deq
    return (a.cuda(), packed.cuda(), scales.cuda(), qzeros.cuda(), g_idx.cuda(), ref.cuda())

def run(tag, M, K, N, group_size, seed=0):
    a, packed, scales, qzeros, g_idx, ref = build(M, K, N, group_size, seed)
    out = {"tag": tag, "M": M, "K": K, "N": N, "group_size": group_size,
           "num_pid_k": math.ceil(K / 16), "K_mult_of_16": (K % 16 == 0)}
    try:
        c = kmod.gptq_matmul(a, packed, scales, qzeros, g_idx, bits=4)
        torch.cuda.synchronize()
        err = (c - ref).abs()
        out["max_abs_err"] = float(err.max())
        out["rel_err"] = float(err.max() / ref.abs().max())
        out["ref_absmax"] = float(ref.abs().max())
        out["ok"] = True
    except Exception as e:
        out["ok"] = False
        out["exception"] = repr(e)[:300]
    print(json.dumps(out), flush=True)
    return out

run("control_K128", 32, 128, 64, 64, seed=1)
run("tail_K72", 32, 72, 64, 64, seed=2)
print(json.dumps({"tag": "done"}), flush=True)
'''

p = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=480)
lines = [l for l in p.stdout.splitlines() if l.strip()]
res = []
for l in lines:
    try:
        res.append(json.loads(l))
    except Exception:
        pass
print("STDOUT_RAW:", p.stdout[-2000:])
print("STDERR_TAIL:", p.stderr[-1500:])
print(json.dumps({"returncode": p.returncode, "results": res,
                  "reached_done": any(r.get("tag") == "done" for r in res)}))
