
import torch, json, importlib.util, random
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_05/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
EPS = k._TIE_EPS
dev='cuda'

def ref(s, p):
    v = s[s > p]
    if v.numel()==0: return 0
    m = v.min()
    return int(((v-m).abs() < EPS).sum().item())

random.seed(7); torch.manual_seed(7)
mismatch=[]; n=0
dists=["randn","uniform","quant","quant_fine","exp","clustered","bimodal"]
for trial in range(600):
    e = random.randint(0,12)
    N = 2**e
    d = random.choice(dists)
    if d=="randn": s = torch.randn(N)
    elif d=="uniform": s = torch.rand(N)*random.choice([1.0,10.0,1e3])
    elif d=="quant": s = torch.randint(0,random.choice([2,3,5,10]),(N,)).float()
    elif d=="quant_fine": s = torch.randint(0,50,(N,)).float()*random.choice([1e-4,5e-4,1e-3,2e-3])
    elif d=="exp": s = torch.distributions.Exponential(1.0).sample((N,))
    elif d=="clustered":
        base = torch.randn(1).item()
        s = base + torch.randn(N)*random.choice([1e-4,1e-3,1e-2])
    else:
        s = torch.cat([torch.randn(N//2 if N>1 else 1)*0.001, torch.randn(N - (N//2 if N>1 else 1))+5.0]) if N>1 else torch.randn(1)
    s = s.float()
    # pivot choices
    pc = random.random()
    if pc < 0.3: p = float(s[random.randrange(N)])
    elif pc < 0.6: p = float(s.median())
    elif pc < 0.8: p = float(s.min())-random.random()
    else: p = float(s.max())-random.random()*abs(float(s.max()))-1e-6
    sd = s.to(dev).contiguous()
    kv = k.count_tied_at_boundary(sd, p)
    rv = ref(sd, p)
    n+=1
    if kv != rv:
        mismatch.append({"trial":trial,"N":N,"dist":d,"pivot":p,"kernel":kv,"ref":rv})
print(json.dumps({"metric":"exact integer count vs ref under EPS=1e-3","n_cases":n,"n_mismatch":len(mismatch),"mismatches":mismatch[:10]}))
