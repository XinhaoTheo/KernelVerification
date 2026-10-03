
import torch, json, sys
sys.path.insert(0, "/root/cases/case_107")
from kernel import run, reference

torch.cuda.synchronize()
def make(B,L,D,K,seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    s = torch.randn((B,L,D), generator=g)
    e = torch.randn((B,K,D), generator=g)
    # include zeros and signed zeros
    s[0,0,0] = 0.0
    if D>1: e[0,0,1] = -0.0
    return s.cuda(), e.cuda()

cases = [
    (1,2,1,1),(1,2,1,2),(4,2,1024,1),(4,2,1024,2),
    (1,3,7,1),(1,3,7,2),(1,3,7,3),(2,5,3,4),(1,7,1,7),
    (3,17,13,16),(2,64,33,64),(1,4096,1,128),(1,4096,3,128),
    (4,1024,5,128),(2,1000,7,1),(1,4096,1024,2),(1,4096,1024,128),
    (3,33,65,32),(1,1,1,1),(1,2,1,128),(2,129,4,128),(1,2048,1024,3),
]
fails=[]
for (B,L,D,K) in cases:
    try:
        s,e = make(B,L,D,K,hash((B,L,D,K))%10000)
    except Exception as ex:
        continue  # illegal K maybe
    s_pre = s.clone(); e_pre = e.clone()
    ref = reference(s_pre, e_pre)
    ret = run(s, e)
    torch.cuda.synchronize()
    ok_bits = torch.equal(s.view(torch.int32), ref.view(torch.int32))
    ok_ret = (ret is s) and (s.data_ptr()==s_pre.data_ptr()) if False else (ret is s)
    ok_ev = torch.equal(e, e_pre)
    if not (ok_bits and ok_ret and ok_ev):
        fails.append({"case":[B,L,D,K],"bits":ok_bits,"ret_same":ret is s,"ev_ok":ok_ev,
                      "n_diff":(s.view(torch.int32)!=ref.view(torch.int32)).sum().item()})
# repeated calls
B,L,D=2,50,11
s,e = make(B,L,D,5,123)
ok_rep=True
for i in range(5):
    K = [5,1,3,11,2][i]
    e = torch.randn((B,K,D), device="cuda")
    ref = reference(s, e)
    s = run(s, e)
    torch.cuda.synchronize()
    if not torch.equal(s.view(torch.int32), ref.view(torch.int32)):
        ok_rep=False; break
print(json.dumps({"metric":"exact FP32 bit equality vs reference() over shape sweep + in-place + event preservation + repeated calls","n_cases":len(cases)+1,"fails":fails,"repeated_ok":ok_rep}))
