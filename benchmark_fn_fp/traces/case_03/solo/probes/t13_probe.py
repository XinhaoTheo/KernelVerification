
import subprocess, json, sys, textwrap

SRC = r'''
import torch, json, importlib.util, sys
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_03/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
torch.manual_seed(0); dev='cuda'
K = int(sys.argv[1]); gs = int(sys.argv[2]); M=32; N=32; bits=4
ipw = 32//bits
num_groups = (K+gs-1)//gs
g_idx = (torch.arange(K, device=dev)//gs).to(torch.int32)
q = torch.randint(0, 2**bits, (K,N), device=dev, dtype=torch.int32)
packed = torch.zeros((K//ipw, N), device=dev, dtype=torch.int32)
for k in range(K):
    packed[k//ipw] |= (q[k] << ((k % ipw)*bits))
scales = (torch.rand((num_groups,N), device=dev)*0.1+0.01).float()
zq = torch.randint(0, 2**bits, (num_groups,N), device=dev, dtype=torch.int32)
qzeros = torch.zeros((num_groups, N//ipw), device=dev, dtype=torch.int32)
for n in range(N):
    qzeros[:, n//ipw] |= (zq[:, n] << ((n % ipw)*bits))
a = torch.randn((M,K), device=dev, dtype=torch.float32)
deq = (q.float() - (zq[g_idx.long()]+1).float()) * scales[g_idx.long()]
ref = a @ deq
try:
    c = m.gptq_matmul(a, packed, scales, qzeros, g_idx, bits=4)
    torch.cuda.synchronize()
    err = (c-ref).abs()
    print(json.dumps(dict(K=K, gs=gs, status="ok", max_abs=err.max().item(),
        ref_absmax=ref.abs().max().item(), rel=(err.max()/ref.abs().max()).item(),
        frac_off=(err>1e-3).float().mean().item())))
except Exception as e:
    print(json.dumps(dict(K=K, gs=gs, status="exception", err=repr(e)[:200])))
'''
open("/tmp/one.py","w").write(SRC)
out = {}
for K, gs in [(64,32),(48,32),(40,32),(72,32),(24,16),(56,32)]:
    r = subprocess.run([sys.executable, "/tmp/one.py", str(K), str(gs)],
                       capture_output=True, text=True, timeout=200)
    line = [l for l in r.stdout.strip().splitlines() if l.startswith("{")]
    out[f"K{K}_gs{gs}"] = json.loads(line[-1]) if line else {"rc": r.returncode, "stderr": r.stderr[-300:]}
print("RESULT " + json.dumps(out))
