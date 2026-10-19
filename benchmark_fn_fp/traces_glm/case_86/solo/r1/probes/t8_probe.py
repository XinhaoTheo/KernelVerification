import sys, json, random
sys.path.insert(0, "/root/cases/case_159")
import torch
from kernel import run, reference

device="cuda"
results={}
def test(B,N,seed,extreme=False):
    g=torch.Generator().manual_seed(seed)
    if extreme:
        matrix=torch.randint(0,2,(B,N,2,2),generator=g,dtype=torch.int8)*2-1
        bias=torch.randint(0,2,(B,N,2),generator=g,dtype=torch.int16)*65535-32768
        initial=torch.randint(0,2,(B,2),generator=g,dtype=torch.int16)*65535-32768
    else:
        matrix=torch.randint(-1,2,(B,N,2,2),generator=g,dtype=torch.int8)
        bias=torch.randint(-32768,32768,(B,N,2),generator=g,dtype=torch.int16)
        initial=torch.randint(-32768,32768,(B,2),generator=g,dtype=torch.int16)
    m,b,i=matrix.to(device),bias.to(device),initial.to(device)
    out=run(m,b,i); ref=reference(m,b,i)
    eq=torch.equal(out,ref)
    return eq,(out-ref).abs().max().item() if not eq else 0

fails={}
# reproduce original failure: B=64 N=25 seed 1025
eq,md=test(64,25,1025)
fails["B64_N25_seed1025"]={"eq":eq,"maxabsdiff":md}
# vary seeds at B=64 N=25
for seed in range(2000,2015):
    eq,md=test(64,25,seed)
    fails[f"B64_N25_s{seed}"]=eq
# vary B at N=25
for B in [1,2,4,8,16,32,64]:
    eq,md=test(B,25,1025)
    fails[f"B{B}_N25_s1025"]=eq
# all N 1..32 at B=4, seed 1025
for N in range(1,33):
    eq,md=test(4,N,1025)
    fails[f"B4_N{N}_s1025"]=eq
nfail=sum(1 for k,v in fails.items() if v is False)
print(json.dumps({"metric":"exact equality vs reference","n_checks":len(fails),"n_fail":nfail,"results":fails}))
