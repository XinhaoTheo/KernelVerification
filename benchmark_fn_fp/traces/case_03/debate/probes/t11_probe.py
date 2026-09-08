
import json, os, subprocess, sys, textwrap

SCRIPT = r'''
import sys, json, numpy as np, torch
sys.path.insert(0,'/root/cases/case_03')
from kernel import gptq_matmul
K = int(sys.argv[1]); M, N, gs, bits = 32, 32, 8, 4
ng = (K + gs - 1)//gs
rows = (K + 7)//8
rng = np.random.default_rng(2)
q = rng.integers(0,16,size=(K,N)).astype(np.uint32)
packed = np.zeros((rows,N),dtype=np.uint32)
for k in range(K):
    packed[k//8] |= (q[k] << ((k%8)*4))
packed_t = torch.from_numpy(packed.view(np.int32).copy()).cuda()
scales = torch.from_numpy((rng.random((ng,N))*0.1+0.05).astype(np.float32)).cuda()
zeros = torch.zeros((ng,N),dtype=torch.int32,device='cuda')  # neutralize c1/c2: kernel zero == 1 everywhere
g_idx = (torch.arange(K,device='cuda')//gs).to(torch.int32)
a = torch.from_numpy(rng.standard_normal((M,K)).astype(np.float32)).cuda()
qf = torch.from_numpy(q.astype(np.float32)).cuda()
sc_full = scales[g_idx.long()]
ref = a @ ((qf - 1.0) * sc_full)   # kernel's own convention, so only K-masking can differ
c = gptq_matmul(a, packed_t, scales, zeros, g_idx, bits=bits)
torch.cuda.synchronize()
err = (c - ref).abs().max().item()
absmax = ref.abs().max().item()
print(json.dumps({"K":K,"packed_rows":rows,"num_groups":ng,"k_iters":(K+15)//16,
                  "max_abs_err":err,"ref_absmax":absmax,"rel_err":err/absmax,
                  "nonfinite_out":bool(not torch.isfinite(c).all().item()),
                  "match_1e-3":bool(err<1e-3)}))
'''
path = '/tmp/c3_run.py'
open(path,'w').write(SCRIPT)

results = {}
for K in (32, 24, 40):
    try:
        p = subprocess.run([sys.executable, path, str(K)], capture_output=True, text=True, timeout=200)
        line = [l for l in p.stdout.strip().splitlines() if l.strip()]
        parsed = None
        if line:
            try: parsed = json.loads(line[-1])
            except Exception: parsed = None
        results[f"K{K}"] = {"returncode": p.returncode, "parsed": parsed,
                            "stderr_tail": p.stderr.strip()[-400:] if p.stderr else ""}
    except subprocess.TimeoutExpired:
        results[f"K{K}"] = {"returncode": "timeout", "parsed": None, "stderr_tail": ""}

print(json.dumps({"block_size_k":16,"group_size":8,"results":results}))
