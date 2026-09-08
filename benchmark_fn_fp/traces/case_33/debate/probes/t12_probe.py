
import json, subprocess, sys, textwrap

child = textwrap.dedent('''
import json, importlib.util, torch
spec = importlib.util.spec_from_file_location("kmod", "/root/cases/case_33/kernel.py")
kmod = importlib.util.module_from_spec(spec); spec.loader.exec_module(kmod)

def to_i32(x64):
    return ((x64 + 2**31) % 2**32 - 2**31).to(torch.int32)

def pack_k(q):
    K, N = q.shape
    out = torch.zeros(K//8, N, dtype=torch.int64, device=q.device)
    for r in range(K//8):
        for j in range(8):
            out[r] |= (q[r*8+j] & 15) << (4*j)
    return to_i32(out)

def pack_n(z):
    G, N = z.shape
    out = torch.zeros(G, N//8, dtype=torch.int64, device=z.device)
    for c in range(N//8):
        for j in range(8):
            out[:, c] |= (z[:, c*8+j] & 15) << (4*j)
    return to_i32(out)

res = {}
M, N, K, gs = 32, 32, 16, 32
ng = -(-K//gs)
res["contract_groups"] = ng
res["wrapper_num_groups"] = K // gs
ar = torch.arange(K)
gw = torch.clamp(ar // gs, max=(K // gs) - 1)
res["wrapper_g_idx_unique"] = sorted(set(gw.tolist()))
res["contract_g_idx_unique"] = sorted(set((ar // gs).tolist()))

torch.manual_seed(0)
q = torch.randint(0, 16, (K, N), device='cuda', dtype=torch.int64)
zq = torch.randint(0, 16, (ng, N), device='cuda', dtype=torch.int64)
scales = (0.05 * (1.0 + torch.rand(ng, N, device='cuda'))).float()
bp = pack_k(q); zp = pack_n(zq)
a = torch.randn(M, K, device='cuda', dtype=torch.float32)

gmap = (torch.arange(K, device='cuda') // gs)
deq = (q.float() - (zq[gmap].float() + 1.0)) * scales[gmap]
C_ref = a @ deq

try:
    C_k = kmod.gptq_matmul(a, bp, scales, zp, gs, 4)
    torch.cuda.synchronize()
    e = (C_k - C_ref).abs().max().item()
    res["launched"] = True
    res["error"] = None
    res["max_abs_err_vs_contract_ref"] = e
    res["max_abs_contract_ref"] = C_ref.abs().max().item()
    res["rel_err"] = e / max(C_ref.abs().max().item(), 1e-30)
    res["allclose_contract"] = bool(torch.allclose(C_k, C_ref, rtol=1e-3, atol=1e-3))
    res["out_finite"] = bool(torch.isfinite(C_k).all().item())
    res["out_sample"] = C_k[0, :4].tolist()
    res["ref_sample"] = C_ref[0, :4].tolist()
except Exception as ex:
    res["launched"] = False
    res["error"] = type(ex).__name__ + ": " + str(ex)[:400]

print(json.dumps(res))
''')

p = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True, timeout=300)
lines = [l for l in p.stdout.strip().splitlines() if l.strip()]
try:
    inner = json.loads(lines[-1]) if lines else {}
except Exception:
    inner = {}
out = {"child_returncode": p.returncode,
       "child_crashed": p.returncode != 0,
       "stderr_tail": p.stderr.strip()[-600:],
       "result": inner}
print(json.dumps(out))
