
import subprocess, sys, json

CHILD = r'''
import json, sys, torch, importlib.util
spec = importlib.util.spec_from_file_location("kern", "/root/cases/case_03/kernel.py")
kern = importlib.util.module_from_spec(spec); spec.loader.exec_module(kern)

K = int(sys.argv[1])
M, N = 32, 32
bits, ipb, group_size = 4, 8, 8
torch.manual_seed(0)
dev = "cuda"
a = torch.randn(M, K, device=dev, dtype=torch.float32)
b_packed = torch.randint(-2**31, 2**31-1, (K//ipb, N), device=dev, dtype=torch.int32)
num_groups = (K + group_size - 1)//group_size
scales = (torch.rand(num_groups, N, device=dev)*0.1+0.01).float()
zeros_packed = torch.randint(-2**31, 2**31-1, (num_groups, N//ipb), device=dev, dtype=torch.int32)
g_idx = (torch.arange(K, device=dev)//group_size).to(torch.int32)

k_idx = torch.arange(K, device=dev)
n_idx = torch.arange(N, device=dev)
q = (b_packed[k_idx//ipb, :].to(torch.int64) >> ((k_idx % ipb)*bits)[:,None]) & 15
z = ((zeros_packed[:, n_idx//ipb].to(torch.int64) >> ((n_idx % ipb)*bits)[None,:]) & 15) + 1
g = g_idx.long()
deq = (q - z[g,:]).float() * scales[g,:]
ref = a @ deq

out = kern.gptq_matmul(a, b_packed, scales, zeros_packed, g_idx, bits=bits)
torch.cuda.synchronize()
err = (out-ref).abs()
scale_ref = float(ref.abs().max())
print("RESULT " + json.dumps({
  "K": K, "num_k_iters": (K+15)//16,
  "max_abs_err": float(err.max()),
  "ref_absmax": scale_ref,
  "max_rel_err": float(err.max())/scale_ref,
  "frac_elems_off_1pct": float((err > 0.01*scale_ref).float().mean()),
  "out_sample": [float(x) for x in out[0,:4]],
  "ref_sample": [float(x) for x in ref[0,:4]],
}))
'''

results = {}
for K in (32, 24, 40):
    p = subprocess.run([sys.executable, "-c", CHILD, str(K)], capture_output=True, text=True, timeout=300)
    line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
    if line:
        results[str(K)] = json.loads(line[0][7:])
    else:
        results[str(K)] = {"K": K, "crashed": True, "returncode": p.returncode,
                           "stderr_tail": p.stderr.strip().splitlines()[-6:]}
print(json.dumps(results))
