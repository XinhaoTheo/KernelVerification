
import subprocess, sys, json

CHILD = r'''
import json, sys, torch, importlib.util
spec = importlib.util.spec_from_file_location("kern", "/root/cases/case_03/kernel.py")
kern = importlib.util.module_from_spec(spec); spec.loader.exec_module(kern)

M, N, K = 32, 32, 32
bits, ipb, group_size = 4, 8, 8
torch.manual_seed(1)
dev = "cuda"
a = torch.randn(M, K, device=dev, dtype=torch.float32)
b_packed = torch.randint(-2**31, 2**31-1, (K//ipb, N), device=dev, dtype=torch.int32)
num_groups = K//group_size
scales = (torch.rand(num_groups, N, device=dev)*0.1+0.01).float()
g_idx = (torch.arange(K, device=dev)//group_size).to(torch.int32)
k_idx = torch.arange(K, device=dev); n_idx = torch.arange(N, device=dev)
q = (b_packed[k_idx//ipb, :].to(torch.int64) >> ((k_idx % ipb)*bits)[:,None]) & 15
g = g_idx.long()

# ---- Case A: caller follows the DOCSTRING (unpacked (num_groups, N) zeros, values 0..15)
zeros_unpacked = torch.randint(0, 16, (num_groups, N), device=dev, dtype=torch.int32)
ref_unpacked_plus1 = (a @ ((q - (zeros_unpacked.to(torch.int64)[g,:]+1)).float() * scales[g,:]))
ref_unpacked_raw   = (a @ ((q -  zeros_unpacked.to(torch.int64)[g,:]   ).float() * scales[g,:]))
outA = kern.gptq_matmul(a, b_packed, scales, zeros_unpacked, g_idx, bits=bits); torch.cuda.synchronize()
# what the kernel actually computes with this tensor: reads element [g, n//8] shifted by (n%8)*4
zA = ((zeros_unpacked[:, n_idx//ipb].to(torch.int64) >> ((n_idx % ipb)*bits)[None,:]) & 15) + 1
ref_kernel_semantics_A = a @ ((q - zA[g,:]).float() * scales[g,:])

# ---- Case B: caller follows the KERNEL BODY (packed (num_groups, N//8) zeros)
zeros_packed = torch.randint(-2**31, 2**31-1, (num_groups, N//ipb), device=dev, dtype=torch.int32)
zB = ((zeros_packed[:, n_idx//ipb].to(torch.int64) >> ((n_idx % ipb)*bits)[None,:]) & 15) + 1
ref_packed = a @ ((q - zB[g,:]).float() * scales[g,:])
outB = kern.gptq_matmul(a, b_packed, scales, zeros_packed, g_idx, bits=bits); torch.cuda.synchronize()

def stats(o, r):
    e = (o-r).abs(); s = float(r.abs().max())
    return {"max_abs_err": float(e.max()), "ref_absmax": s, "max_rel_err": float(e.max())/s,
            "frac_off_1pct": float((e > 0.01*s).float().mean())}

print("RESULT " + json.dumps({
  "docstring_layout_vs_ref_plus1": stats(outA, ref_unpacked_plus1),
  "docstring_layout_vs_ref_raw":   stats(outA, ref_unpacked_raw),
  "docstring_layout_vs_kernel_packed_semantics": stats(outA, ref_kernel_semantics_A),
  "packed_layout_vs_packed_ref":   stats(outB, ref_packed),
  "zeros_unpacked_shape": list(zeros_unpacked.shape),
  "zeros_packed_shape": list(zeros_packed.shape),
}))
'''
p = subprocess.run([sys.executable, "-c", CHILD], capture_output=True, text=True, timeout=300)
line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
print(json.dumps(json.loads(line[0][7:]) if line else
      {"crashed": True, "returncode": p.returncode, "stderr_tail": p.stderr.strip().splitlines()[-8:]}))
