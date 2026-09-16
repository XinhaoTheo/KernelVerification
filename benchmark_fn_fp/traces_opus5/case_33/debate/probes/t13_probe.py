
import json, subprocess, sys, textwrap, os

child = textwrap.dedent('''
import importlib.util, json
import numpy as np, torch
spec = importlib.util.spec_from_file_location("kmod", "/root/cases/case_33/kernel.py")
kmod = importlib.util.module_from_spec(spec); spec.loader.exec_module(kmod)

M,K,N,gs = 64,72,64,8
g = torch.Generator().manual_seed(3)
ng = (K+gs-1)//gs
q = torch.randint(0,16,(K,N),generator=g,dtype=torch.int64)
z = torch.randint(1,17,(ng,N),generator=g,dtype=torch.int64)
s = torch.rand((ng,N),generator=g)*0.1+0.01
a = torch.randn((M,K),generator=g)
qn = q.numpy().astype(np.uint32)
bp = np.zeros((K//8,N),dtype=np.uint32)
for k in range(K):
    bp[k//8] |= qn[k] << (4*(k%8))
b = torch.from_numpy(bp.view(np.int32).copy()).cuda()
zn = (z.numpy()-1).astype(np.uint32)
qz = np.zeros((ng,N//8),dtype=np.uint32)
for n in range(N):
    qz[:,n//8] |= zn[:,n] << (4*(n%8))
qzp = torch.from_numpy(qz.view(np.int32).copy()).cuda()

gidx = torch.arange(K)//gs
deq = (q.double() - z[gidx].double()) * s[gidx].double()
R = a.double() @ deq

out = kmod.gptq_matmul(a.cuda(), b, s.cuda(), qzp, gs, 4)
torch.cuda.synchronize()
out = out.double().cpu()
err = (out-R).abs().max().item()
print("CHILD_JSON"+json.dumps({
 "config":{"M":M,"K":K,"N":N,"group_size":gs,"ceil_groups":ng,"floor_groups":K//gs,
           "K_mult_of_16":K%16==0,"K_mult_of_gs":K%gs==0,"loop_iters":-(-K//16),"k_cols_touched":16*(-(-K//16))},
 "ref_max_abs":R.abs().max().item(),
 "max_abs_err_vs_contract_ref":err,
 "rel_err":err/R.abs().max().item(),
 "finite":bool(torch.isfinite(out).all().item())
}))
''')
p = "/tmp/c2_child.py"
open(p,"w").write(child)
r = subprocess.run([sys.executable,p],capture_output=True,text=True,timeout=420,
                   env={**os.environ,"CUDA_LAUNCH_BLOCKING":"1"})
child_json = None
for line in r.stdout.splitlines():
    if line.startswith("CHILD_JSON"):
        child_json = json.loads(line[len("CHILD_JSON"):])
print(json.dumps({
 "returncode": r.returncode,
 "child_result": child_json,
 "stderr_tail": r.stderr[-1500:],
 "metric_reason": "K=72 keeps clamp a no-op (K%gs==0) so any deviation or crash isolates the missing K-dimension mask"
}))
